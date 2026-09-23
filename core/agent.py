"""Baseline agent: a single ReAct-style tool loop inside the sandbox.

The model sees the task, gets bash/file tools in a Docker sandbox, and must call
`submit` with a valid submission file. Budgets (steps, wall-clock, tokens) are
enforced here, not trusted to the model; every LLM call and tool call is traced.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLM, LLMError
from .sandbox import DockerSandbox, truncate
from .tasks import Task
from .tools import Tool, ToolRegistry, ToolResult
from .trace import Tracer

SYSTEM = """You are an expert machine-learning engineer working autonomously on a Kaggle-style task.

Environment:
- You act only through tools. Commands run in a Linux sandbox with Python 3.11 and numpy, pandas,
  scikit-learn, scipy, lightgbm, xgboost, catboost. There is NO internet: do not pip install.
- Task files are read-only under /data. Your working directory is /work (writable, persists across calls).
- Resources: {cpus} CPUs, {memory} RAM{gpu_note}. Each command is killed after its timeout.
- Total budget: {time_limit_min} minutes wall-clock and {max_steps} tool-using steps. Remaining budget is
  shown after every tool result. Plan so that a valid submission exists well before the budget ends.

How to work:
- Inspect the data first, then build a simple, correct baseline and a valid submission early.
- Improve with a proper local validation (hold-out or cross-validation); the test labels are hidden,
  so your own validation score is the only signal you have. Print it.
- Write code to files (write_file) and run them with bash; keep console output short.
- The final answer is a CSV with exactly the columns and ids of /data/sample_submission.csv.
  Call submit(path) when done; if validation fails you will get the errors and can fix them."""


@dataclass
class Budget:
    max_steps: int = 60
    time_limit_s: float = 3600
    max_total_tokens: int = 3_000_000
    context_tokens: int = 60_000        # compact history above this prompt size
    cmd_timeout_s: int = 900
    min_submit_frac: float = 0.0        # reject `submit` before this fraction of wall-clock is used
    loop_warn: int = 5                  # identical consecutive tool calls before an intervention message
    loop_stop: int = 10                 # ... before the episode is stopped


@dataclass
class EpisodeState:
    t0: float = field(default_factory=time.monotonic)
    step: int = 0
    prompt_tokens: int = 0          # size of the latest prompt (= current context)
    total_prompt_tokens: int = 0    # summed over calls (compute cost)
    completion_tokens: int = 0
    submitted: str | None = None
    early_submits: int = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def accepted_path(sandbox: DockerSandbox) -> Path:
    """Where an accepted submission is stored: next to /work, never inside it, so the agent
    can neither collide with it nor plant an unvalidated file there."""
    return sandbox.workdir.parent / "accepted_submission.csv"


def provisional_path(sandbox: DockerSandbox) -> Path:
    """Latest valid submission that was refused only because it came too early (submit gate)."""
    return sandbox.workdir.parent / "provisional_submission.csv"


ENV_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost"]
ENV_PROBE = ("import sys, importlib.metadata as m\n"
             "print('python', sys.version.split()[0])\n"
             f"for p in {ENV_PACKAGES!r}:\n"
             "    try: print(p, m.version(p))\n"
             "    except m.PackageNotFoundError: print(p, 'not installed')\n")


def env_facts(sandbox: DockerSandbox) -> str:
    """Installed versions, read from the sandbox itself, plus a general debugging rule.
    Deliberately task-agnostic: no library-specific API hints."""
    r = sandbox.exec(f"python3 - <<'EOF'\n{ENV_PROBE}EOF", timeout=60)
    versions = r.output.strip() if r.exit_code == 0 else "unavailable"
    return ("\n\nInstalled versions (authoritative; your memory of these libraries' APIs may be outdated):\n"
            f"{versions}\n"
            "If a call fails with an unexpected/unknown-argument error, inspect the installed signature "
            "(e.g. `python -c \"import inspect, lib; print(inspect.signature(lib.fn))\"` or help()) "
            "and adapt, instead of retrying variants from memory.")


def make_tools(sandbox: DockerSandbox, task: Task, state: EpisodeState, budget: Budget) -> ToolRegistry:
    def bash(command: str, timeout: int = 600) -> ToolResult:
        remaining = int(budget.time_limit_s - state.elapsed())
        t = max(10, min(timeout, budget.cmd_timeout_s, remaining))
        r = sandbox.exec(command, timeout=t)
        head = f"[exit {r.exit_code}, {r.duration:.1f}s{', KILLED: timeout ' + str(t) + 's' if r.timed_out else ''}]"
        return ToolResult(f"{head}\n{truncate(r.output)}", error=r.exit_code != 0)

    def write_file(path: str, content: str) -> str:
        p = sandbox.write_file(path, content)
        return f"wrote {len(content)} chars to /work/{p.relative_to(sandbox.workdir)}"

    def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
        if path.startswith("/data/") or path == "/data":
            p = task.public_dir / path[len("/data"):].lstrip("/")
        else:
            p = sandbox.host_path(path)
        lines = p.read_text(errors="replace").splitlines()
        chunk = lines[offset:offset + limit]
        more = f"\n[... {len(lines) - offset - limit} more lines]" if len(lines) > offset + limit else ""
        return truncate("\n".join(f"{i + offset + 1:5d}  {l}" for i, l in enumerate(chunk))) + more

    def submit(path: str = "/work/submission.csv") -> ToolResult:
        p = sandbox.host_path(path)
        errors = task.validate_submission(p)
        if errors:
            return ToolResult("submission rejected:\n- " + "\n- ".join(errors), error=True)
        earliest = budget.min_submit_frac * budget.time_limit_s
        if state.elapsed() < earliest:
            shutil.copyfile(p, provisional_path(sandbox))
            state.early_submits += 1
            return ToolResult(
                f"submission is valid and saved as provisional, but it is too early to finish: "
                f"{state.elapsed() / 60:.1f} of {budget.time_limit_s / 60:.0f} min used, submitting is allowed after "
                f"{earliest / 60:.0f} min. Use the remaining time to improve your validation score (e.g. proper "
                f"cross-validation, feature engineering, tuning, other model families, ensembling), then submit "
                f"your best file. If time runs out, the provisional file is used.", error=True)
        shutil.copyfile(p, accepted_path(sandbox))
        state.submitted = str(p)
        return ToolResult("submission accepted. Episode finished.", done=True)

    return ToolRegistry([
        Tool("bash", "Run a bash command in the sandbox (cwd /work). Returns exit code and combined stdout/stderr.",
             {"command": {"type": "string"},
              "timeout": {"type": "integer", "description": "seconds, default 600"}}, bash, ["command"]),
        Tool("write_file", "Create or overwrite a file under /work.",
             {"path": {"type": "string"}, "content": {"type": "string"}}, write_file, ["path", "content"]),
        Tool("read_file", "Read a text file under /work or /data with line numbers.",
             {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
             read_file, ["path"]),
        Tool("submit", "Validate and submit the final predictions CSV. Ends the episode if valid.",
             {"path": {"type": "string"}}, submit, []),
    ])


def compact(messages: list[dict], keep_last: int = 16, stub_chars: int = 400) -> int:
    """Shrink old tool outputs in place, keeping system/task and the recent window intact.
    Returns chars removed. Tool-call/result pairing is preserved (only content shrinks)."""
    removed = 0
    for m in messages[2:max(2, len(messages) - keep_last)]:
        c = m.get("content") or ""
        if m["role"] == "tool" and len(c) > stub_chars:
            m["content"] = c[:stub_chars] + "\n[older output elided to save context]"
            removed += len(c) - len(m["content"])
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                a = call["function"]["arguments"]
                if len(a) > 2000:  # large write_file payloads: keep the path, drop the body
                    try:
                        path = json.loads(a).get("path", "?")
                    except (json.JSONDecodeError, AttributeError):
                        path = "?"
                    call["function"]["arguments"] = json.dumps(
                        {"path": path, "content": "[elided; read the file if needed]"})
                    removed += len(a) - len(call["function"]["arguments"])
    return removed


class Agent:
    def __init__(self, llm: LLM, task: Task, sandbox: DockerSandbox, tracer: Tracer, budget: Budget,
                 temperature: float = 0.7, seed: int | None = None, use_env_facts: bool = False):
        self.llm, self.task, self.sandbox, self.tracer, self.budget = llm, task, sandbox, tracer, budget
        self.temperature, self.seed, self.use_env_facts = temperature, seed, use_env_facts
        self.state = EpisodeState()
        self.tools = make_tools(sandbox, task, self.state, budget)

    def _status(self) -> str:
        b, s = self.budget, self.state
        gate = ""
        if b.min_submit_frac and s.elapsed() < b.min_submit_frac * b.time_limit_s:
            gate = f", submit allowed after {b.min_submit_frac * b.time_limit_s / 60:.0f} min"
        return (f"[budget: step {s.step}/{b.max_steps}, {s.elapsed() / 60:.1f}/{b.time_limit_s / 60:.0f} min, "
                f"context {s.prompt_tokens} tokens{gate}]")

    def _out_of_budget(self) -> str | None:
        b, s = self.budget, self.state
        if s.step >= b.max_steps:
            return "max_steps"
        if s.elapsed() >= b.time_limit_s:
            return "time_limit"
        if s.total_prompt_tokens + s.completion_tokens >= b.max_total_tokens:
            return "token_limit"
        return None

    def run(self) -> dict:
        sb = self.sandbox
        gpu_note = f", GPU(s) {sb.gpus}" if sb.gpus else ", no GPU"
        system = SYSTEM.format(cpus=sb.cpus, memory=sb.memory, gpu_note=gpu_note,
                               time_limit_min=int(self.budget.time_limit_s / 60), max_steps=self.budget.max_steps)
        if self.budget.min_submit_frac:
            system += (f"\n- submit is only accepted after {self.budget.min_submit_frac:.0%} of the time budget; "
                       "earlier valid submissions are kept as provisional. Use the time to improve.")
        if self.use_env_facts:
            system += env_facts(sb)
        files = sorted(p.name for p in self.task.public_dir.iterdir())
        user = f"{self.task.description}\n\nFiles in /data: {', '.join(files)}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.tracer.log("episode_start", task=self.task.name, model=self.llm.model, budget=vars(self.budget),
                        system=system, user=user)

        stop_reason, idle = None, 0
        last_call, repeats = None, 0
        while not (stop_reason := self._out_of_budget()):
            if self.state.prompt_tokens > self.budget.context_tokens:
                removed = compact(messages)
                self.tracer.log("compact", chars_removed=removed)
            try:
                c = self.llm.chat(messages, tools=self.tools.schemas(), temperature=self.temperature,
                                  seed=self.seed)
            except LLMError as e:
                self.tracer.log("llm_error", error=str(e))
                stop_reason = "llm_error"
                break
            self.state.prompt_tokens = c.prompt_tokens
            self.state.total_prompt_tokens += c.prompt_tokens
            self.state.completion_tokens += c.completion_tokens
            self.tracer.log("llm", step=self.state.step, message=c.message, finish=c.finish_reason,
                            prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens,
                            latency=round(c.latency, 2))
            messages.append(c.message)

            calls = c.message.get("tool_calls") or []
            if not calls:
                idle += 1
                if idle >= 3:
                    stop_reason = "no_tool_calls"
                    break
                note = ("Your reply was cut off (max tokens). Keep tool arguments smaller, e.g. write files in parts."
                        if c.finish_reason == "length" else
                        "Continue by calling a tool. Call submit when your submission is ready.")
                messages.append({"role": "user", "content": f"{note}\n{self._status()}"})
                continue
            idle = 0
            self.state.step += 1
            signature = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
            repeats = repeats + 1 if signature == last_call else 1
            last_call = signature
            if repeats >= self.budget.loop_stop:
                self.tracer.log("loop", step=self.state.step, repeats=repeats, action="stop")
                stop_reason = "loop_detected"
                break

            done = False
            for call in calls:
                name, args = call["function"]["name"], call["function"]["arguments"]
                t0 = time.monotonic()
                res = self.tools.dispatch(name, args)
                self.tracer.log("tool", step=self.state.step, name=name, args=args, output=res.text,
                                error=res.error, done=res.done, duration=round(time.monotonic() - t0, 2))
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": f"{res.text}\n{self._status()}"})
                done = done or res.done
            if repeats >= self.budget.loop_warn and not done:
                self.tracer.log("loop", step=self.state.step, repeats=repeats, action="warn")
                messages.append({"role": "user", "content": (
                    f"You have made the identical tool call {repeats} times in a row and it keeps producing the same "
                    "result. Repeating it will not change anything. Read the last tool result carefully and do "
                    "something different.")})
            if done:
                stop_reason = "submitted"
                break

        result = self._finish(stop_reason)
        self.tracer.log("episode_end", **result)
        return result

    def _finish(self, stop_reason: str) -> dict:
        final = accepted_path(self.sandbox)
        source = "submit" if final.exists() else None
        if source is None and provisional_path(self.sandbox).exists():
            shutil.copyfile(provisional_path(self.sandbox), final)
            source = "provisional"
        if source is None:
            # budget ran out: fall back to what the agent left behind, but only if it validates
            fallback = self.sandbox.workdir / "submission.csv"
            if fallback.exists() and not self.task.validate_submission(fallback):
                shutil.copyfile(fallback, final)
                source = "fallback"
        score = None
        if final.exists():
            try:
                score = self.task.grade(final)
            except Exception as e:
                self.tracer.log("grade_error", error=str(e))
        return {"stop_reason": stop_reason, "score": score, "metric": self.task.metric,
                "higher_is_better": self.task.higher_is_better, "valid_submission": final.exists(),
                "submission_source": source, "early_submits": self.state.early_submits,
                "steps": self.state.step, "elapsed_s": round(self.state.elapsed(), 1),
                "total_prompt_tokens": self.state.total_prompt_tokens,
                "completion_tokens": self.state.completion_tokens}
