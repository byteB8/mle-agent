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

import pandas as pd

from .agent import Budget, EpisodeState, accepted_path, env_facts
from .llm import LLM, LLMError
from .sandbox import DockerSandbox, truncate
from .tasks import Task
from .trace import Tracer

SCORE_RE = re.compile(r"VALIDATION_SCORE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

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
- /work/input/train.csv  labelled training data
- /work/input/valid.csv  validation rows, features only: the harness holds the labels and scores your predictions
- /work/input/test.csv   test rows, features only
- /data/sample_submission.csv  the required output format (other task files under /data are read-only)

Requirements for every script:
1. Train only on /work/input/train.csv (use internal cross-validation there if you want to tune).
2. Write `valid_predictions.csv` (for valid.csv) and `submission.csv` (for test.csv) in the current directory, both
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
    children: list[int] = field(default_factory=list)

    def summary(self) -> str:
        res = f"score {self.score:.5f}" if not self.buggy else f"FAILED ({self.error[:80]})"
        return f"- node {self.id} [{self.op}{'' if self.parent is None else f' of {self.parent}'}] {res}: {self.plan[:200]}"


@dataclass
class SearchConfig:
    num_drafts: int = 3
    debug_prob: float = 0.5
    max_debug_depth: int = 3
    node_timeout_s: int = 600
    max_memory_nodes: int = 15      # how many earlier attempts are summarised in a draft/improve prompt
    harness_valid: bool = True      # harness holds out and scores a validation split (when the task allows it)
    valid_frac: float = 0.2


class TreeSearchAgent:
    def __init__(self, llm: LLM, task: Task, sandbox: DockerSandbox, tracer: Tracer, budget: Budget,
                 config: SearchConfig | None = None, temperature: float = 0.7, seed: int | None = None,
                 use_env_facts: bool = False):
        self.llm, self.task, self.sandbox, self.tracer, self.budget = llm, task, sandbox, tracer, budget
        self.cfg = config or SearchConfig()
        self.temperature, self.seed, self.use_env_facts = temperature, seed, use_env_facts
        self.rng = random.Random(seed)
        self.state = EpisodeState()
        self.nodes: list[Node] = []
        self.data_preview = ""
        self.valid_labels = None    # set in harness-valid mode; never written under /work

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
    def select(self) -> tuple[str, Node | None]:
        drafts = [n for n in self.nodes if n.parent is None]
        if len(drafts) < self.cfg.num_drafts:
            return "draft", None
        if self.rng.random() < self.cfg.debug_prob:
            debuggable = [n for n in self.nodes if n.buggy and not n.children
                          and n.debug_depth < self.cfg.max_debug_depth and n.code]
            if debuggable:
                return "debug", self.rng.choice(debuggable)
        best = self.best()
        return ("improve", best) if best else ("draft", None)

    # ---------- prompts ----------
    def _memory(self) -> str:
        done = self.nodes[-self.cfg.max_memory_nodes:]
        return "\n".join(n.summary() for n in done) if done else "(none yet)"

    def _prompt(self, op: str, parent: Node | None) -> str:
        head = f"# Task\n{self.task.description}\n\n# Data preview\n{self.data_preview}\n"
        if op == "draft":
            return (head + f"\n# Earlier attempts\n{self._memory()}\n\n"
                    "Write a NEW solution. Prefer an approach that differs from the earlier attempts; "
                    "a simple, correct, fast solution is better than an ambitious broken one.")
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

    def _run_node(self, node: Node) -> None:
        rel = f"nodes/{node.id}"
        self.sandbox.write_file(f"{rel}/solution.py", node.code)
        timeout = int(max(10, min(self.cfg.node_timeout_s, self._remaining())))
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
        op, parent = self.select()
        prompt = self._prompt(op, parent)
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
        blocks = CODE_RE.findall(text)
        node = Node(id=len(self.nodes), parent=None if parent is None else parent.id, op=op,
                    plan=text.split("```")[0].strip()[:600],
                    code=blocks[-1].strip() if blocks else "",
                    debug_depth=(parent.debug_depth + 1) if op == "debug" else 0)
        if parent is not None:
            parent.children.append(node.id)
        if node.code:
            self._run_node(node)
        else:
            node.error = "no ```python code block in the reply"
            node.output = f"[harness] {node.error}"
        self.nodes.append(node)
        best = self.best()
        self.tracer.log("node", step=self.state.step, id=node.id, parent=node.parent, op=op, plan=node.plan,
                        code=node.code, output=truncate(node.output, 4000), score=node.score,
                        self_score=node.self_score, buggy=node.buggy,
                        error=node.error, exec_s=round(node.exec_s, 1), prompt=prompt, response=text,
                        prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens,
                        llm_latency=round(c.latency, 2), best_id=best.id if best else None,
                        best_score=best.score if best else None)
        return True

    def _final_reserve(self) -> float:
        """Time kept back to re-run the best script on the full training set."""
        best = self.best()
        return 1.5 * best.exec_s + 30 if (best and self.valid_labels is not None) else 0.0

    def _out_of_budget(self) -> str | None:
        if self.state.step >= self.budget.max_steps:
            return "max_steps"
        if self._remaining() <= 30 + self._final_reserve():   # not enough time for a meaningful node
            return "time_limit"
        if self.state.total_prompt_tokens + self.state.completion_tokens >= self.budget.max_total_tokens:
            return "token_limit"
        return None

    def _setup_harness_valid(self) -> None:
        train, valid_x, valid_y = self.task.split_train(seed=self.seed or 0, valid_frac=self.cfg.valid_frac)
        inp = self.sandbox.host_path("input")
        inp.mkdir(parents=True, exist_ok=True)
        train.to_csv(inp / "train.csv", index=False)
        valid_x.to_csv(inp / "valid.csv", index=False)
        shutil.copyfile(self.task.public_dir / "test.csv", inp / "test.csv")
        valid_y.to_csv(self.sandbox.workdir.parent / "valid_labels.csv", index=False)  # audit copy, outside /work
        self.valid_labels = valid_y

    def run(self) -> dict:
        sb = self.sandbox
        harness = self.cfg.harness_valid and self.task.can_split and (self.task.public_dir / "test.csv").exists()
        if harness:
            self._setup_harness_valid()
        template = SYSTEM_HARNESS_VALID if harness else SYSTEM
        self.system = template.format(cpus=sb.cpus, memory=sb.memory,
                                    gpu_note=f", GPU(s) {sb.gpus}" if sb.gpus else ", no GPU",
                                    node_timeout=self.cfg.node_timeout_s, metric=self.task.metric,
                                    direction="higher is better" if self.task.higher_is_better else "lower is better")
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
        """Re-run the chosen script with the validation rows folded back into train.csv.
        Returns True if it produced a valid submission at nodes/final/submission.csv."""
        full = pd.read_csv(self.task.public_dir / "train.csv")
        full.to_csv(self.sandbox.host_path("input/train.csv"), index=False)
        self.sandbox.write_file("nodes/final/solution.py", best.code)
        timeout = int(max(10, min(self.cfg.node_timeout_s, self._remaining())))
        r = self.sandbox.exec("cd nodes/final && python solution.py", timeout=timeout)
        ok = r.exit_code == 0 and not self.task.validate_submission(self.sandbox.host_path("nodes/final/submission.csv"))
        self.tracer.log("final_refit", best_id=best.id, ok=ok, exit=r.exit_code, exec_s=round(r.duration, 1),
                        output=truncate(r.output, 2000))
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
                "steps": self.state.step, "elapsed_s": round(self.state.elapsed(), 1),
                "total_prompt_tokens": self.state.total_prompt_tokens,
                "completion_tokens": self.state.completion_tokens,
                "nodes": len(self.nodes), "buggy_nodes": sum(n.buggy for n in self.nodes),
                "ops": {o: ops.count(o) for o in ("draft", "improve", "debug")},
                "best_node": best.id if best else None, "best_val": best.score if best else None,
                "best_self_val": best.self_score if best else None}
