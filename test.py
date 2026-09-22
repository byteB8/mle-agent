"""Unit tests (no GPU, no LLM server, no Docker needed):  python test.py"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from core.agent import Agent, Budget, compact
from core.llm import Completion, _normalise
from core.sandbox import DockerSandbox, ExecResult, truncate
from core.tasks import Task
from core.tools import Tool, ToolRegistry
from core.trace import Tracer, read_trace


class LocalSandbox(DockerSandbox):
    """Same interface, runs bash on the host inside the work dir (tests only)."""
    def start(self):
        self.workdir.mkdir(parents=True, exist_ok=True)

    def stop(self):
        pass

    def exec(self, command, timeout=600):
        t0 = time.monotonic()
        r = subprocess.run(["bash", "-c", command], cwd=self.workdir, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return ExecResult(r.returncode, r.stdout.decode(), False, time.monotonic() - t0)


class ScriptedLLM:
    """Replays a fixed list of assistant turns."""
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, **kw):
        self.seen.append(json.loads(json.dumps(messages)))
        msg = self.turns.pop(0)
        return Completion(_normalise(msg), "tool_calls" if msg.get("tool_calls") else "stop", 100, 10, 0.0)


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{time.monotonic_ns()}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def make_task(root: Path) -> Task:
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir()
    pd.DataFrame({"id": [1, 2, 3, 4], "y": [0.5] * 4}).to_csv(root / "public/sample_submission.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(root / "private/answers.csv", index=False)
    (root / "task.json").write_text(json.dumps({"metric": "auc", "id_col": "id", "target_cols": ["y"],
                                                "higher_is_better": True}))
    (root / "description.md").write_text("toy task")
    return Task.load(root)


class TestTools(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry([Tool("add", "", {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                      lambda a, b: a + b, ["a", "b"])])

    def test_ok_and_coercion(self):
        self.assertEqual(self.reg.dispatch("add", '{"a": 2, "b": "3"}').text, "5")

    def test_errors_are_results(self):
        self.assertIn("unknown tool", self.reg.dispatch("nope", "{}").text)
        self.assertIn("not valid JSON", self.reg.dispatch("add", "{a:").text)
        self.assertIn("missing required", self.reg.dispatch("add", '{"a": 1}').text)
        self.assertIn("unknown argument", self.reg.dispatch("add", '{"a": 1, "b": 2, "c": 3}').text)
        self.assertIn("must be integer", self.reg.dispatch("add", '{"a": true, "b": 2}').text)

    def test_tool_exception_contained(self):
        reg = ToolRegistry([Tool("boom", "", {}, lambda: 1 / 0)])
        r = reg.dispatch("boom", "{}")
        self.assertTrue(r.error)
        self.assertIn("ZeroDivisionError", r.text)


class TestSandboxHelpers(unittest.TestCase):
    def test_truncate_keeps_tail(self):
        s = "a" * 50000 + "FINAL_ERROR"
        t = truncate(s, 1000)
        self.assertLess(len(t), 1100)
        self.assertTrue(t.endswith("FINAL_ERROR"))

    def test_host_path_blocks_escape(self):
        with tempfile.TemporaryDirectory() as d:
            sb = DockerSandbox("x", "img", Path(d), Path(d))
            self.assertEqual(sb.host_path("/work/a/b.py"), Path(d).resolve() / "a/b.py")
            self.assertEqual(sb.host_path("a.py"), Path(d).resolve() / "a.py")
            for bad in ("../etc/passwd", "/etc/passwd", "/work/../../x"):
                with self.assertRaises(ValueError):
                    sb.host_path(bad)


class TestTask(unittest.TestCase):
    def test_validate_and_grade(self):
        with tempfile.TemporaryDirectory() as d:
            task = make_task(Path(d) / "t")
            sub = Path(d) / "s.csv"
            pd.DataFrame({"id": [4, 3, 2, 1], "y": [0.9, 0.1, 0.8, 0.2]}).to_csv(sub, index=False)
            self.assertEqual(task.validate_submission(sub), [])
            self.assertAlmostEqual(task.grade(sub), 1.0)
            pd.DataFrame({"id": [1, 2, 3], "pred": [0, 0, 0]}).to_csv(sub, index=False)
            errs = task.validate_submission(sub)
            self.assertTrue(any("columns" in e for e in errs) and any("rows" in e for e in errs))


class TestAgent(unittest.TestCase):
    def test_episode_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            llm = ScriptedLLM([
                {"role": "assistant", "content": "thinking without a tool"},   # nudged, not a step
                call("write_file", path="make.py", content=(
                    "import pandas as pd\n"
                    "pd.DataFrame({'id':[1,2,3,4],'y':[.1,.9,.2,.8]}).to_csv('submission.csv',index=False)\n")),
                call("bash", command="python3 make.py && echo done"),
                call("submit", path="/work/nope.csv"),                          # rejected, agent continues
                call("submit"),
            ])
            sb = LocalSandbox("x", "img", d / "work", task.public_dir)
            tracer = Tracer(d / "trace.jsonl")
            with sb:
                res = Agent(llm, task, sb, tracer, Budget(max_steps=10)).run()
            tracer.close()
            self.assertEqual(res["stop_reason"], "submitted")
            self.assertEqual(res["steps"], 4)
            self.assertAlmostEqual(res["score"], 1.0)
            kinds = [e["kind"] for e in read_trace(d / "trace.jsonl")]
            self.assertEqual(kinds[0], "episode_start")
            self.assertEqual(kinds[-1], "episode_end")
            # every tool result must answer the preceding call id (API requirement)
            last = llm.seen[-1]
            ids = {c["id"] for m in last if m["role"] == "assistant" for c in m.get("tool_calls", [])}
            self.assertTrue(all(m["tool_call_id"] in ids for m in last if m["role"] == "tool"))

    def test_step_budget_and_fallback_submission(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            (d / "work").mkdir()
            pd.DataFrame({"id": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(d / "work/submission.csv", index=False)
            llm = ScriptedLLM([call("bash", command="true")] * 3)
            sb = LocalSandbox("x", "img", d / "work", task.public_dir)
            with sb:
                res = Agent(llm, task, sb, Tracer(None), Budget(max_steps=3)).run()
            self.assertEqual(res["stop_reason"], "max_steps")
            self.assertTrue(res["valid_submission"])

    def test_compact_keeps_pairs_and_recent(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        for i in range(30):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": str(i), "type": "function",
                 "function": {"name": "write_file", "arguments": json.dumps({"path": "f.py", "content": "x" * 5000})}}]})
            msgs.append({"role": "tool", "tool_call_id": str(i), "content": "y" * 5000})
        removed = compact(msgs, keep_last=4)
        self.assertGreater(removed, 0)
        self.assertEqual(len(msgs), 62)
        self.assertEqual(len(msgs[-1]["content"]), 5000)
        self.assertIn('"path": "f.py"', msgs[2]["tool_calls"][0]["function"]["arguments"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
