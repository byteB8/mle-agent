"""Solution-tree search agent (AIDE-style).

Instead of one long tool-using conversation, every node is a complete training script.
The harness runs it in the sandbox, parses the validation score it prints, and checks
the submission format. The LLM is asked for exactly one of three operations per call:

    draft    a new solution from scratch (sees short summaries of earlier attempts)
    improve  one atomic change to the current best node
    debug    fix a broken node, given its error output

Each call is short and self-contained, so contexts stay small (cheap, and clean training
data for phase 2), and a bad idea costs one node instead of derailing the whole episode.
"""
from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .agent import Budget, EpisodeState, accepted_path, env_facts
from .llm import LLM, LLMError
from .sandbox import DockerSandbox, truncate
from .tasks import Task, read_table, write_table
from .trace import Tracer

SCORE_RE = re.compile(r"VALIDATION_SCORE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
OPEN_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*)\Z", re.S)   # last block with no closing fence


def extract_code(text: str) -> str:
    """Last fenced python block; falls back to an unterminated one (reply cut off at the token limit)."""
    blocks = CODE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    m = OPEN_CODE_RE.search(text)
    return m.group(1).strip() if m else ""


# One per initial draft (order shuffled per seed), so the search starts from genuinely different model families.
MODEL_FAMILIES = [
    "gradient-boosted decision trees (LightGBM, XGBoost or CatBoost)",
    "a linear model (logistic / ridge / linear regression) with thorough feature preprocessing: scaling, one-hot "
    "encoding, TF-IDF for text, interaction or polynomial features",
    "a neural network (scikit-learn MLPClassifier / MLPRegressor) on standardised features",
    "a random-forest or extra-trees ensemble",
    "a k-nearest-neighbours or support-vector-machine model on scaled features",
]

SYSTEM = """You are an expert machine-learning engineer solving a Kaggle-style task by writing complete Python scripts.

Every script you write is run by a harness as `python solution.py` in its own directory, inside a sandbox:
- Python 3.11 with numpy, pandas, scikit-learn, scipy, lightgbm, xgboost, catboost. NO internet, no pip install.
- Task files are read-only under /data. {cpus} CPUs, {memory} RAM{gpu_note}.
- The script must finish within {node_timeout} seconds or it is killed.

Requirements for every script:
1. Self-contained: load data from /data, train, predict.
2. Evaluate with a proper held-out validation (hold-out split or k-fold CV) using the task metric ({metric},
   {direction}), and print exactly one line `VALIDATION_SCORE: <float>` with that score.
3. Write predictions for the test set to `submission.csv` in the current directory, with exactly the columns and
   ids of /data/sample_submission.csv.
4. Keep printed output short.

Answer format: first a few sentences describing your plan, then the full script in ONE ```python code block."""

# Harness-scored mode: the script never sees validation labels, so it cannot leak or misreport its score.
SYSTEM_HARNESS_VALID = """You are an expert machine-learning engineer solving a Kaggle-style task by writing complete Python scripts.

Every script you write is run by a harness as `python solution.py` in its own directory, inside a sandbox:
- Python 3.11 with numpy, pandas, scikit-learn, scipy, lightgbm, xgboost, catboost. NO internet, no pip install.
- {cpus} CPUs, {memory} RAM{gpu_note}. The script must finish within {node_timeout} seconds or it is killed.

Inputs (use exactly these paths):
- /work/input/{train}  labelled training data
- /work/input/{valid}  validation rows, features only: the harness holds the labels and scores your predictions
- /work/input/{test}   test rows, features only
- /data/sample_submission.csv  the required output format (other task files under /data are read-only)

Requirements for every script:
1. Train only on /work/input/{train} (use internal cross-validation there if you want to tune).
2. Write `valid_predictions.csv` (for {valid}) and `submission.csv` (for {test}) in the current directory, both
   with exactly the columns of /data/sample_submission.csv and the ids of the corresponding input file.
3. The harness computes the official validation score ({metric}, {direction}) from valid_predictions.csv. You may
   print your own estimate as `VALIDATION_SCORE: <float>`, but it is not used. Keep printed output short.

Answer format: first a few sentences describing your plan, then the full script in ONE ```python code block."""


@dataclass
class Node:
    id: int
    parent: int | None
    op: str
    plan: str = ""
    code: str = ""
    output: str = ""
    score: float | None = None      # harness-computed when available, else self-reported
    self_score: float | None = None # what the script printed (logged, never trusted in harness mode)
    buggy: bool = True
    error: str = ""
    debug_depth: int = 0
    exec_s: float = 0.0
    preflight_s: float = 0.0
    family: str = ""                # model family of the draft this node descends from (diverse drafts)
    children: list[int] = field(default_factory=list)

    def summary(self) -> str:
        res = f"score {self.score:.5f}" if not self.buggy else f"FAILED ({self.error[:80]})"
        return f"- node {self.id} [{self.op}{'' if self.parent is None else f' of {self.parent}'}] {res}: {self.plan[:200]}"


def error_signature(error: str) -> str:
    """Stable key for an error: the exception line with numbers masked, e.g.
    "TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'"."""
    line = re.sub(r"^exit \d+: ", "", error.strip())
    return re.sub(r"\d+", "N", line)[:160]


@dataclass
class Lesson:
    signature: str
    count: int = 1              # how many nodes hit it
    fix: str = ""               # how a debug node got past it (the model's own explanation), if any

    def render(self) -> str:
        how = f" Fix that worked: {self.fix}" if self.fix else " (no working fix found yet)"
        return f"- `{self.signature}` (hit by {self.count} script(s)).{how}"


@dataclass
class SearchConfig:
    num_drafts: int = 3
    debug_prob: float = 0.5
    max_debug_depth: int = 3
    node_timeout_s: int = 600
    max_memory_nodes: int = 15      # how many earlier attempts are summarised in a draft/improve prompt
    harness_valid: bool = True      # harness holds out and scores a validation split (when the task allows it)
    valid_frac: float = 0.2
    lessons: bool = False           # share error->fix lessons across branches
    max_lessons: int = 8
    lesson_min_count: int = 2       # an unfixed error becomes a lesson once this many nodes hit it
    preflight_rows: int = 0         # >0: first run each script on this many training rows to catch crashes cheaply
    preflight_min_rows: int = 20_000  # ... only when the training set is at least this large
    preflight_timeout_s: int = 90
    search_max_rows: int = 0        # >0: cap training rows during search; the chosen script is refit on full data
    max_reserve_frac: float = 0.25  # most of the budget the final refit may hold back
    refit_max_ratio: float = 4.0    # final refit trains on at most this multiple of the search rows (time permitting)
    diverse_drafts: bool = False    # assign each draft a different model family
    family_rescue: bool = False     # with diverse drafts: make every family produce a valid node before exploiting,
    family_attempts: int = 3        # ... spending at most this many debug/redraft nodes per family
    runner_up_prob: float = 0.3     # ... and sometimes improve the best node of the runner-up family instead


class TreeSearchAgent:
    def __init__(self, llm: LLM, task: Task, sandbox: DockerSandbox, tracer: Tracer, budget: Budget,
                 config: SearchConfig | None = None, temperature: float = 0.7, seed: int | None = None,
                 use_env_facts: bool = False):
        self.llm, self.task, self.sandbox, self.tracer, self.budget = llm, task, sandbox, tracer, budget
        self.cfg = config or SearchConfig()
        self.temperature, self.seed, self.use_env_facts = temperature, seed, use_env_facts
        self.rng = random.Random(seed)
        self.families = MODEL_FAMILIES[:]
        self.rng.shuffle(self.families)
        self.state = EpisodeState()
        self.nodes: list[Node] = []
        self.data_preview = ""
        self.valid_labels = None    # set in harness-valid mode; never written under /work
        self.lessons: dict[str, Lesson] = {}
        self.preflight_ids = None   # (valid ids, test ids) of the pre-flight sample, when pre-flight is on
        self.full_ratio = 1.0       # all training rows / search training rows
        self.search_rows = 0

    # ---------- scoring ----------
    def _better(self, a: float, b: float) -> bool:
        return a > b if self.task.higher_is_better else a < b

    def best(self) -> Node | None:
        good = [n for n in self.nodes if not n.buggy]
        if not good:
            return None
        best = good[0]
        for n in good[1:]:
            if self._better(n.score, best.score):
                best = n
        return best

    # ---------- policy ----------
    def _best_of(self, nodes: list[Node]) -> Node | None:
        best = None
        for n in nodes:
            if not n.buggy and (best is None or self._better(n.score, best.score)):
                best = n
        return best

    def _rescue(self) -> tuple[str, Node | None, str] | None:
        """First drafted family that has no valid node yet and attempts left: debug its newest broken leaf,
        or re-draft it when there is nothing to debug (e.g. the reply had no code)."""
        seen = []
        for d in (n for n in self.nodes if n.parent is None and n.family):
            if d.family in seen:
                continue
            seen.append(d.family)
            fam_nodes = [n for n in self.nodes if n.family == d.family]
            if any(not n.buggy for n in fam_nodes) or len(fam_nodes) - 1 >= self.cfg.family_attempts:
                continue
            leaves = [n for n in fam_nodes if n.buggy and not n.children and n.code
                      and n.debug_depth < self.cfg.max_debug_depth]
            return ("debug", leaves[-1], d.family) if leaves else ("draft", None, d.family)
        return None

    def select(self) -> tuple[str, Node | None, str]:
        """Returns (operation, parent node, model family for a draft)."""
        drafts = [n for n in self.nodes if n.parent is None]
        if len(drafts) < self.cfg.num_drafts:
            return "draft", None, self._next_family()
        if self.cfg.diverse_drafts and self.cfg.family_rescue:
            rescue = self._rescue()
            if rescue:
                return rescue
        if self.rng.random() < self.cfg.debug_prob:
            debuggable = [n for n in self.nodes if n.buggy and not n.children
                          and n.debug_depth < self.cfg.max_debug_depth and n.code]
            if debuggable:
                return "debug", self.rng.choice(debuggable), ""
        best = self.best()
        if best and self.cfg.family_rescue and self.rng.random() < self.cfg.runner_up_prob:
            others = [self._best_of([n for n in self.nodes if n.family == f])
                      for f in {n.family for n in self.nodes if n.family and n.family != best.family}]
            runner_up = self._best_of([n for n in others if n is not None])
            if runner_up:
                return "improve", runner_up, ""
        return ("improve", best, "") if best else ("draft", None, self._next_family())

    # ---------- prompts ----------
    def _memory(self) -> str:
        done = self.nodes[-self.cfg.max_memory_nodes:]
        return "\n".join(n.summary() for n in done) if done else "(none yet)"

    def _learn(self, node: Node) -> None:
        """Update lessons after a node ran: count failures, and record the fix when a debug node succeeds."""
        if node.buggy and node.error:
            sig = error_signature(node.error)
            if sig in self.lessons:
                self.lessons[sig].count += 1
            else:
                self.lessons[sig] = Lesson(sig)
        if node.op == "debug" and not node.buggy:
            parent = self.nodes[node.parent]
            lesson = self.lessons.get(error_signature(parent.error))
            if lesson is not None and not lesson.fix:
                lesson.fix = " ".join(node.plan.split())[:300]

    def _pitfalls(self) -> str:
        if not self.cfg.lessons:
            return ""
        shown = [l for l in self.lessons.values() if l.fix or l.count >= self.cfg.lesson_min_count]
        shown = sorted(shown, key=lambda l: (not l.fix, -l.count))[:self.cfg.max_lessons]
        if not shown:
            return ""
        return ("\n# Known pitfalls in this environment (learned from earlier scripts; avoid them)\n"
                + "\n".join(l.render() for l in shown) + "\n")

    def _next_family(self) -> str:
        if not self.cfg.diverse_drafts:
            return ""
        drafts = sum(n.op == "draft" for n in self.nodes)
        return self.families[drafts % len(self.families)]

    def _prompt(self, op: str, parent: Node | None, family: str = "") -> str:
        family = family if op == "draft" else ""
        head = (f"# Task\n{self.task.description}\n\n# Data preview\n{self.data_preview}\n"
                + self._pitfalls())
        if op == "draft":
            return (head + f"\n# Earlier attempts\n{self._memory()}\n\n"
                    "Write a NEW solution. Prefer an approach that differs from the earlier attempts; "
                    "a simple, correct, fast solution is better than an ambitious broken one."
                    + (f"\n\nModel family for this draft: {family}. Other drafts explore other families, so stay "
                       "within this one and make it as strong as you can." if family else ""))
        if op == "improve":
            return (head + f"\n# Earlier attempts\n{self._memory()}\n\n"
                    f"# Current best solution (node {parent.id}, validation {parent.score:.5f})\n"
                    f"```python\n{parent.code}\n```\n\n# Its output\n{truncate(parent.output, 3000)}\n\n"
                    "Propose ONE specific, atomic improvement likely to raise the validation score (e.g. a feature, "
                    "a better validation-driven hyperparameter choice, a different model, an ensemble). Do not repeat "
                    "an idea that already failed. Then give the full updated script.")
        return (head + f"\n# Broken script (node {parent.id})\n```python\n{parent.code}\n```\n\n"
                f"# Its output / error\n{truncate(parent.output, 4000)}\n\n"
                "Find the root cause, fix it, and give the full corrected script. If an error says an argument is "
                "unexpected, the installed library version differs from what you remember: check the signature.")

    # ---------- execution ----------
    def _remaining(self) -> float:
        return self.budget.time_limit_s - self.state.elapsed()

    def _preflight(self, node: Node) -> str | None:
        """Run the script on the small sample in input_small/. Returns an error, or None if it passed or was
        inconclusive (timeout: probably just slow, so the full run decides)."""
        rel = f"nodes/{node.id}_pf"
        self.sandbox.write_file(f"{rel}/solution.py", node.code.replace("/work/input/", "/work/input_small/"))
        timeout = int(max(10, min(self.cfg.preflight_timeout_s, self._remaining() - self._final_reserve())))
        r = self.sandbox.exec(f"cd {rel} && python solution.py", timeout=timeout)
        node.preflight_s = r.duration
        if r.timed_out:
            return None
        if r.exit_code != 0:
            node.output = f"[harness] pre-flight run on a {self.cfg.preflight_rows}-row training sample failed:\n{r.output}"
            return f"exit {r.exit_code}: " + (r.output.strip().splitlines() or [""])[-1][:200]
        valid_ids, test_ids = self.preflight_ids
        problems = ["submission.csv: " + p for p in self.task.validate_submission(
            self.sandbox.host_path(f"{rel}/submission.csv"), expected_ids=test_ids)]
        problems += ["valid_predictions.csv: " + p for p in self.task.validate_submission(
            self.sandbox.host_path(f"{rel}/valid_predictions.csv"), expected_ids=valid_ids)]
        if problems:
            node.output = f"[harness] pre-flight run produced invalid output:\n{r.output}"
            return "invalid output: " + "; ".join(problems)
        return None

    def _run_node(self, node: Node) -> None:
        rel = f"nodes/{node.id}"
        self.sandbox.write_file(f"{rel}/solution.py", node.code)
        if self.preflight_ids is not None:
            err = self._preflight(node)
            if err:
                node.error = f"pre-flight: {err}"
                return
        # a node may not eat into the time kept back for the final refit
        timeout = int(max(10, min(self.cfg.node_timeout_s, self._remaining() - self._final_reserve())))
        r = self.sandbox.exec(f"cd {rel} && python solution.py", timeout=timeout)  # cwd is /work
        node.exec_s = r.duration
        node.output = r.output
        scores = SCORE_RE.findall(r.output)
        node.self_score = float(scores[-1]) if scores else None
        sub = self.sandbox.host_path(f"{rel}/submission.csv")
        harness = self.valid_labels is not None
        if r.timed_out:
            node.error = f"killed after {timeout}s timeout"
        elif r.exit_code != 0:
            node.error = f"exit {r.exit_code}: " + (r.output.strip().splitlines() or [""])[-1][:200]
        elif not harness and not scores:
            node.error = "no VALIDATION_SCORE line printed"
        else:
            problems = ["submission.csv: " + p for p in self.task.validate_submission(sub)]
            if harness:
                vp = self.sandbox.host_path(f"{rel}/valid_predictions.csv")
                problems += ["valid_predictions.csv: " + p for p in
                             self.task.validate_submission(vp, expected_ids=self.valid_labels[self.task.id_col])]
            if problems:
                node.error = "invalid output: " + "; ".join(problems)
            else:
                try:
                    node.score = self.task.grade(vp, answers=self.valid_labels) if harness else node.self_score
                    node.buggy = False
                except Exception as e:  # e.g. non-numeric predictions
                    node.error = f"could not score predictions: {e}"
        if harness and not node.buggy:
            node.output += f"\n[harness] validation {self.task.metric} = {node.score:.5f}"
        if node.buggy and node.error not in node.output:
            node.output += f"\n[harness] {node.error}"

    def _step(self) -> bool:
        op, parent, family = self.select()
        prompt = self._prompt(op, parent, family)
        family = family or (parent.family if parent else "")   # descendants keep their draft's family
        try:
            c = self.llm.chat([{"role": "system", "content": self.system}, {"role": "user", "content": prompt}],
                              temperature=self.temperature, max_tokens=8192,
                              seed=None if self.seed is None else self.seed * 100_003 + len(self.nodes))
        except LLMError as e:
            self.tracer.log("llm_error", error=str(e))
            return False
        self.state.step += 1
        self.state.prompt_tokens = c.prompt_tokens
        self.state.total_prompt_tokens += c.prompt_tokens
        self.state.completion_tokens += c.completion_tokens

        text = c.message.get("content") or ""
        node = Node(id=len(self.nodes), parent=None if parent is None else parent.id, op=op,
                    plan=text.split("```")[0].strip()[:600], code=extract_code(text), family=family,
                    debug_depth=(parent.debug_depth + 1) if op == "debug" else 0)
        if parent is not None:
            parent.children.append(node.id)
        if node.code:
            self._run_node(node)
        else:
            node.error = "no ```python code block in the reply"
            node.output = f"[harness] {node.error}"
        self.nodes.append(node)
        self._learn(node)
        best = self.best()
        self.tracer.log("node", step=self.state.step, id=node.id, parent=node.parent, op=op, family=family,
                        plan=node.plan,
                        code=node.code, output=truncate(node.output, 4000), score=node.score,
                        self_score=node.self_score, buggy=node.buggy,
                        error=node.error, exec_s=round(node.exec_s, 1), preflight_s=round(node.preflight_s, 1),
                        prompt=prompt, response=text,
                        prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens,
                        llm_latency=round(c.latency, 2), best_id=best.id if best else None,
                        best_score=best.score if best else None)
        return True

    def _final_reserve(self) -> float:
        """Time kept back to re-run the best script on the full training set: its search runtime scaled by
        how much more data the refit sees, capped at a fraction of the budget."""
        best = self.best()
        if not best or self.valid_labels is None:
            return 0.0
        return min(self.cfg.max_reserve_frac * self.budget.time_limit_s,
                   1.5 * best.exec_s * self._planned_ratio() + 30)

    def _planned_ratio(self) -> float:
        return min(self.full_ratio, self.cfg.refit_max_ratio)

    def _out_of_budget(self) -> str | None:
        if self.state.step >= self.budget.max_steps:
            return "max_steps"
        if self._remaining() <= 30 + self._final_reserve():   # not enough time for a meaningful node
            return "time_limit"
        if self.state.total_prompt_tokens + self.state.completion_tokens >= self.budget.max_total_tokens:
            return "token_limit"
        return None

    def _setup_harness_valid(self) -> None:
        seed = self.seed or 0
        train, valid_x, valid_y = self.task.split_train(seed=seed, valid_frac=self.cfg.valid_frac)
        full_rows = len(train) + len(valid_x)          # what the final refit trains on
        cap = self.cfg.search_max_rows
        if cap and len(train) > cap:                   # multi-fidelity: search on a subsample
            train = train.sample(n=cap, random_state=seed)
        if cap and len(valid_x) > cap // 2:
            keep = valid_x.sample(n=cap // 2, random_state=seed).index
            valid_x, valid_y = valid_x.loc[keep], valid_y.loc[keep]
        self.full_ratio = full_rows / len(train)
        self.search_rows = len(train)
        inp = self.sandbox.host_path("input")
        inp.mkdir(parents=True, exist_ok=True)
        write_table(train, inp / self.input_names["train"])
        write_table(valid_x, inp / self.input_names["valid"])
        shutil.copyfile(self.task.public_dir / self.task.test_file, inp / self.input_names["test"])
        valid_y.to_csv(self.sandbox.workdir.parent / "valid_labels.csv", index=False)  # audit copy, outside /work
        self.valid_labels = valid_y
        if self.cfg.preflight_rows and len(train) >= self.cfg.preflight_min_rows:
            self._setup_preflight(train, valid_x)

    def _setup_preflight(self, train, valid_x) -> None:
        """input_small/: a few training rows (at least one per class when the target is categorical) and the
        first rows of valid/test, so a script can be smoke-tested in seconds."""
        seed, n = self.seed or 0, self.cfg.preflight_rows
        label = self.task.label_col or (self.task.target_cols[0] if len(self.task.target_cols) == 1 else None)
        parts = []
        if label is not None and train[label].nunique() <= 100:
            parts.append(train.groupby(label, group_keys=False).head(1))
        rest = train.drop(parts[0].index) if parts else train
        parts.append(rest.sample(n=max(0, min(len(rest), n - sum(len(p) for p in parts))), random_state=seed))
        small_train = pd.concat(parts)
        small_valid = valid_x.head(200)
        small_test = read_table(self.task.public_dir / self.task.test_file).head(200)
        inp = self.sandbox.host_path("input_small")
        inp.mkdir(parents=True, exist_ok=True)
        write_table(small_train, inp / self.input_names["train"])
        write_table(small_valid, inp / self.input_names["valid"])
        write_table(small_test, inp / self.input_names["test"])
        self.preflight_ids = (small_valid[self.task.id_col], small_test[self.task.id_col])

    def run(self) -> dict:
        sb = self.sandbox
        ext = Path(self.task.train_file).suffix
        self.input_names = {"train": f"train{ext}", "valid": f"valid{ext}", "test": f"test{Path(self.task.test_file).suffix}"}
        harness = self.cfg.harness_valid and self.task.can_split
        if harness:
            self._setup_harness_valid()
        template = SYSTEM_HARNESS_VALID if harness else SYSTEM
        self.system = template.format(cpus=sb.cpus, memory=sb.memory,
                                    gpu_note=f", GPU(s) {sb.gpus}" if sb.gpus else ", no GPU",
                                    node_timeout=self.cfg.node_timeout_s, metric=self.task.metric,
                                    direction="higher is better" if self.task.higher_is_better else "lower is better",
                                    **self.input_names)
        if harness and self.full_ratio > 1.5:
            self.system += (f"\n\nNote: during the search, {self.input_names['train']} is a {1 / self.full_ratio:.0%} "
                            "subsample of the training data. The script you end up with is re-run once on more data "
                            f"(up to ~{self._planned_ratio():.0f}x more rows), so don't hard-code row counts and keep the "
                            "runtime scalable.")
        if self.use_env_facts:
            self.system += env_facts(sb)
        where = "/work/input/* /data/sample_submission.csv" if harness else "/data/*"
        preview = sb.exec(f"for f in {where}; do echo \"== $f ($(wc -l < \"$f\") lines)\"; "
                          "head -c 1500 \"$f\" | head -n 4; echo; done", timeout=60)
        self.data_preview = truncate(preview.output, 6000)
        self.tracer.log("episode_start", task=self.task.name, model=self.llm.model, agent="tree",
                        budget=vars(self.budget), config=vars(self.cfg), system=self.system, harness_valid=harness,
                        data_preview=self.data_preview)

        stop_reason = None
        while not (stop_reason := self._out_of_budget()):
            if not self._step():
                stop_reason = "llm_error"
                break

        result = self._finish(stop_reason)
        self.tracer.log("episode_end", **result)
        return result

    def _final_refit(self, best: Node) -> bool:
        """Re-run the chosen script on more training data (validation rows folded back in, and for a subsampled
        search up to `refit_max_ratio` x the search rows), sized to the time actually left: runtime is assumed to
        grow linearly with rows, with a 1.5x safety factor. Returns True if it produced a valid submission at
        nodes/final/submission.csv; skips (False) when there isn't time for a meaningfully larger fit."""
        affordable = (self._remaining() - 20) / (1.5 * max(best.exec_s, 1.0))
        ratio = min(self._planned_ratio(), affordable)
        if ratio < 1.1:
            self.tracer.log("final_refit", best_id=best.id, ok=False, skipped=True, affordable_ratio=round(affordable, 2))
            return False
        full = read_table(self.task.public_dir / self.task.train_file)
        rows = int(min(len(full), self.search_rows * ratio))
        dst = self.sandbox.host_path(f"input/{self.input_names['train']}")
        if rows >= len(full):
            shutil.copyfile(self.task.public_dir / self.task.train_file, dst)
        else:
            write_table(full.sample(n=rows, random_state=self.seed or 0), dst)
        self.sandbox.write_file("nodes/final/solution.py", best.code)
        timeout = int(max(10, self._remaining()))  # more data can take longer than a search node
        r = self.sandbox.exec("cd nodes/final && python solution.py", timeout=timeout)
        ok = r.exit_code == 0 and not self.task.validate_submission(self.sandbox.host_path("nodes/final/submission.csv"))
        self.tracer.log("final_refit", best_id=best.id, ok=ok, exit=r.exit_code, exec_s=round(r.duration, 1),
                        rows=rows, ratio=round(rows / max(self.search_rows, 1), 2), output=truncate(r.output, 2000))
        return ok

    def _finish(self, stop_reason: str) -> dict:
        best = self.best()
        final = accepted_path(self.sandbox)
        score, source = None, None
        if best is not None:
            source = "tree_best"
            chosen = f"nodes/{best.id}/submission.csv"
            if self.valid_labels is not None and self._final_refit(best):
                chosen, source = "nodes/final/submission.csv", "tree_best_refit"
            shutil.copyfile(self.sandbox.host_path(chosen), final)
            try:
                score = self.task.grade(final)
            except Exception as e:
                self.tracer.log("grade_error", error=str(e))
        ops = [n.op for n in self.nodes]
        return {"stop_reason": stop_reason, "score": score, "metric": self.task.metric,
                "higher_is_better": self.task.higher_is_better, "valid_submission": final.exists(),
                "submission_source": source, "early_submits": 0,
                "val_mode": "harness" if self.valid_labels is not None else "self_reported",
                "full_ratio": round(self.full_ratio, 2), "preflight": self.preflight_ids is not None,
                "draft_families": [n.family for n in self.nodes if n.op == "draft" and n.family],
                "valid_families": sorted({n.family for n in self.nodes if n.family and not n.buggy}),
                "best_family": best.family if best else None,
                "preflight_rejects": sum(n.error.startswith("pre-flight") for n in self.nodes),
                "steps": self.state.step, "elapsed_s": round(self.state.elapsed(), 1),
                "total_prompt_tokens": self.state.total_prompt_tokens,
                "completion_tokens": self.state.completion_tokens,
                "nodes": len(self.nodes), "buggy_nodes": sum(n.buggy for n in self.nodes),
                "ops": {o: ops.count(o) for o in ("draft", "improve", "debug")},
                "best_node": best.id if best else None, "best_val": best.score if best else None,
                "best_self_val": best.self_score if best else None,
                "lessons": [vars(l) for l in self.lessons.values() if l.fix or l.count >= self.cfg.lesson_min_count]}
