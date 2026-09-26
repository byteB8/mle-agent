"""Unit tests (no GPU, no LLM server, no Docker needed):  python test.py"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from core.agent import Agent, Budget, compact
from core.gpu import parse_free
from core.llm import Completion, _normalise
from core.sandbox import DockerSandbox, ExecResult, truncate
from core.search import Lesson, SearchConfig, TreeSearchAgent
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
        env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}  # `python` = this one
        r = subprocess.run(["bash", "-c", command], cwd=self.workdir, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout, env=env)
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


class TestGpu(unittest.TestCase):
    def test_highest_free_index_first(self):
        smi = "0, 36955\n1, 74827\n2, 9\n3, 9\n"
        self.assertEqual(parse_free(smi), [3, 2])
        self.assertEqual(parse_free("0, 5\n1, 5\n2, 50000\n3, 70000\n"), [1, 0])


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


def reply(plan, code=None):
    body = plan if code is None else f"{plan}\n```python\n{code}\n```"
    return {"role": "assistant", "content": body}


GOOD = """import pandas as pd
pd.DataFrame({{'id': [1, 2, 3, 4], 'y': {preds}}}).to_csv('submission.csv', index=False)
print('VALIDATION_SCORE: {val}')
"""


class TestTreeSearch(unittest.TestCase):
    def run_tree(self, turns, max_steps, **cfg):
        d = Path(tempfile.mkdtemp())
        task = make_task(d / "t")
        llm = ScriptedLLM(turns)
        tracer = Tracer(d / "trace.jsonl")
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = TreeSearchAgent(llm, task, sb, tracer, Budget(max_steps=max_steps, time_limit_s=300),
                                    SearchConfig(**cfg), seed=0)
            res = agent.run()
        tracer.close()
        return res, agent, llm, read_trace(d / "trace.jsonl")

    def test_drafts_then_improve_best_and_pick_by_validation(self):
        turns = [
            reply("draft A: crashes", "raise SystemExit(3)"),
            reply("draft B: ok", GOOD.format(preds=[.9, .1, .8, .2], val=0.60)),   # val 0.60, test AUC 0.0
            reply("draft C: no code"),
            reply("improve B", GOOD.format(preds=[.1, .9, .2, .8], val=0.95)),      # val 0.95, test AUC 1.0
        ]
        res, agent, llm, trace = self.run_tree(turns, max_steps=4, num_drafts=3, debug_prob=0.0)
        self.assertEqual([n.op for n in agent.nodes], ["draft", "draft", "draft", "improve"])
        self.assertEqual(agent.nodes[3].parent, 1)                 # improved the best valid node
        self.assertIn("exit 3", agent.nodes[0].error)
        self.assertIn("no ```python", agent.nodes[2].error)
        self.assertEqual((res["best_node"], res["best_val"], res["score"]), (3, 0.95, 1.0))
        self.assertEqual((res["nodes"], res["buggy_nodes"]), (4, 2))
        self.assertIn("node 1 [draft] score 0.60000", llm.seen[3][1]["content"])  # memory of attempts
        self.assertEqual(sum(e["kind"] == "node" for e in trace), 4)

    def test_debug_and_validation_failures(self):
        turns = [
            reply("draft: forgets score line", "import pandas as pd\n"
                  "pd.DataFrame({'id':[1,2,3,4],'y':[.1,.9,.2,.8]}).to_csv('submission.csv', index=False)"),
            reply("fix: add score", GOOD.format(preds=[.1, .9, .2, .8], val=0.9)),
        ]
        res, agent, _, _ = self.run_tree(turns, max_steps=2, num_drafts=1, debug_prob=1.0)
        self.assertEqual(agent.nodes[0].error, "no VALIDATION_SCORE line printed")
        self.assertEqual((agent.nodes[1].op, agent.nodes[1].parent, agent.nodes[1].debug_depth), ("debug", 0, 1))
        self.assertEqual(res["score"], 1.0)

    def test_higher_is_better_false_picks_minimum(self):
        turns = [reply("a", GOOD.format(preds=[.1, .9, .2, .8], val=0.3)),
                 reply("b", GOOD.format(preds=[.9, .1, .8, .2], val=0.5))]
        d = Path(tempfile.mkdtemp())
        task = make_task(d / "t")
        task.higher_is_better = False
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = TreeSearchAgent(ScriptedLLM(turns), task, sb, Tracer(None), Budget(max_steps=2, time_limit_s=300),
                                    SearchConfig(num_drafts=2), seed=0)
            res = agent.run()
        self.assertEqual(res["best_node"], 0)


def make_split_task(root: Path, n: int = 60) -> Task:
    """Tabular task whose train.csv carries the label, so the harness can hold out validation rows.
    Label is exactly x > 0, so an honest script is perfect and a 'leaky' one can only lie."""
    import numpy as np
    rng = np.random.default_rng(0)
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir()
    x = rng.normal(size=n)
    train = pd.DataFrame({"id": range(n), "x": x, "y": (x > 0).astype(int)})
    xt = rng.normal(size=20)
    train.to_csv(root / "public/train.csv", index=False)
    pd.DataFrame({"id": range(100, 120), "x": xt}).to_csv(root / "public/test.csv", index=False)
    pd.DataFrame({"id": range(100, 120), "y": 0.5}).to_csv(root / "public/sample_submission.csv", index=False)
    pd.DataFrame({"id": range(100, 120), "y": (xt > 0).astype(int)}).to_csv(root / "private/answers.csv", index=False)
    (root / "task.json").write_text(json.dumps({"metric": "auc", "id_col": "id", "target_cols": ["y"],
                                                "higher_is_better": True}))
    (root / "description.md").write_text("predict y from x")
    return Task.load(root)


SCRIPT = """import pandas as pd
tr = pd.read_csv('/work/input/train.csv'.replace('/work/', '../../'))
for src, dst in [('valid', 'valid_predictions.csv'), ('test', 'submission.csv')]:
    df = pd.read_csv(f'../../input/{{src}}.csv')
    pd.DataFrame({{'id': df['id'], 'y': {pred}}}).to_csv(dst, index=False)
print('VALIDATION_SCORE: {claim}')
print('train rows', len(tr))
"""


class TestLessons(unittest.TestCase):
    def test_fix_found_in_one_branch_reaches_later_prompts(self):
        bad = "raise TypeError(\"train() got an unexpected keyword argument 'early_stopping_rounds'\")"
        turns = [reply("draft", bad),
                 reply("draft 2", GOOD.format(preds=[.1, .9, .2, .8], val=0.8)),
                 reply("Use callbacks=[lgb.early_stopping(50)] instead of early_stopping_rounds.",
                       GOOD.format(preds=[.1, .9, .2, .8], val=0.9)),     # debug fixes node 0
                 reply("improve", GOOD.format(preds=[.1, .9, .2, .8], val=0.95))]
        d = Path(tempfile.mkdtemp())
        task = make_task(d / "t")
        llm = ScriptedLLM(turns)
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = TreeSearchAgent(llm, task, sb, Tracer(None), Budget(max_steps=4, time_limit_s=300),
                                    SearchConfig(num_drafts=2, debug_prob=1.0, lessons=True), seed=0)
            res = agent.run()
        self.assertEqual([n.op for n in agent.nodes], ["draft", "draft", "debug", "improve"])
        self.assertNotIn("Known pitfalls", llm.seen[1][1]["content"])  # one unfixed hit: not a lesson yet
        last_prompt = llm.seen[3][1]["content"]
        self.assertIn("Known pitfalls", last_prompt)
        self.assertIn("unexpected keyword argument 'early_stopping_rounds'", last_prompt)
        self.assertIn("callbacks=[lgb.early_stopping(50)]", last_prompt)
        self.assertEqual(len(res["lessons"]), 1)

    def test_off_by_default(self):
        agent = TreeSearchAgent(None, None, None, Tracer(None), Budget())
        agent.lessons["x"] = Lesson("x", count=5, fix="y")
        self.assertEqual(agent._pitfalls(), "")


class TestTaskFormats(unittest.TestCase):
    def test_label_col_split_is_one_hot_and_test_shaped(self):
        d = Path(tempfile.mkdtemp())
        (d / "public").mkdir()
        (d / "private").mkdir()
        pd.DataFrame({"id": range(20), "text": list("abcdefghijklmnopqrst"), "votes_after": range(20),
                      "author": ["A", "B", "C", "A"] * 5}).to_csv(d / "public/train.csv", index=False)
        pd.DataFrame({"id": [100], "text": ["z"]}).to_csv(d / "public/test.csv", index=False)
        pd.DataFrame({"id": [100], "A": [1 / 3], "B": [1 / 3], "C": [1 / 3]}).to_csv(
            d / "public/sample_submission.csv", index=False)
        (d / "task.json").write_text(json.dumps({"metric": "logloss", "id_col": "id", "target_cols": ["A", "B", "C"],
                                                 "higher_is_better": False, "label_col": "author"}))
        (d / "description.md").write_text("x")
        task = Task.load(d)
        self.assertTrue(task.can_split)
        train, vx, vy = task.split_train(seed=0, valid_frac=0.25)
        self.assertEqual(list(vx.columns), ["id", "text"])              # no label, no after-the-fact field
        self.assertEqual(list(vy.columns), ["id", "A", "B", "C"])
        self.assertTrue((vy[["A", "B", "C"]].sum(axis=1) == 1).all())
        self.assertIn("author", train.columns)
        perfect = vy.copy()
        perfect.to_csv(d / "p.csv", index=False)
        self.assertLess(task.grade(d / "p.csv", answers=vy), 1e-6)

    def test_json_task_and_rmsle(self):
        d = Path(tempfile.mkdtemp())
        (d / "public").mkdir()
        (d / "private").mkdir()
        pd.DataFrame({"rid": ["a", "b", "c", "d", "e"], "f": range(5), "y": [0, 1, 0, 1, 1]}).to_json(
            d / "public/train.json", orient="records")
        pd.DataFrame({"rid": ["z"], "f": [9]}).to_json(d / "public/test.json", orient="records")
        (d / "task.json").write_text(json.dumps({"metric": "auc", "id_col": "rid", "target_cols": ["y"],
                                                 "higher_is_better": True, "train_file": "train.json",
                                                 "test_file": "test.json"}))
        (d / "description.md").write_text("x")
        task = Task.load(d)
        self.assertTrue(task.can_split)
        _, vx, vy = task.split_train(seed=1, valid_frac=0.4)
        self.assertEqual(list(vx.columns), ["rid", "f"])
        from core.tasks import _metric
        t = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        self.assertAlmostEqual(_metric("rmsle_mean", t, t, ["a", "b"]), 0.0)
        self.assertGreater(_metric("rmsle_mean", t, t * 2, ["a", "b"]), 0.1)


# Reads every input through the literal '/work/input/' prefix (as real scripts do), mapped to the LocalSandbox layout.
SCRIPT2 = """import pandas as pd
P = lambda f: ('/work/input/' + f).replace('/work/', '../../')
tr = pd.read_csv(P('train.csv'))
{crash}
for src, dst in [('valid', 'valid_predictions.csv'), ('test', 'submission.csv')]:
    df = pd.read_csv(P(src + '.csv'))
    pd.DataFrame({{'id': df['id'], 'y': df['x']}}).to_csv(dst, index=False)
print('train rows', len(tr))
"""


class TestPreflightAndFidelity(unittest.TestCase):
    def run_one(self, code, n=3000, **cfg):
        d = Path(tempfile.mkdtemp())
        task = make_split_task(d / "t", n=n)
        tracer = Tracer(d / "trace.jsonl")
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = TreeSearchAgent(ScriptedLLM([reply("go", code)]), task, sb, tracer,
                                    Budget(max_steps=1, time_limit_s=300), SearchConfig(num_drafts=1, **cfg), seed=0)
            res = agent.run()
        tracer.close()
        return d, agent, res, read_trace(d / "trace.jsonl")

    def test_preflight_rejects_crash_without_full_run(self):
        code = SCRIPT2.format(crash="raise ValueError('bad column')")
        d, agent, res, _ = self.run_one(code, preflight_rows=50, preflight_min_rows=100)
        node = agent.nodes[0]
        self.assertTrue(node.error.startswith("pre-flight: exit 1: ValueError: bad column"))
        self.assertIn("pre-flight run on a 50-row training sample failed", node.output)
        self.assertFalse((d / "work/nodes/0/submission.csv").exists())   # full run never happened
        self.assertEqual(res["preflight_rejects"], 1)

    def test_preflight_pass_then_full_run(self):
        d, agent, res, _ = self.run_one(SCRIPT2.format(crash=""), preflight_rows=50, preflight_min_rows=100)
        self.assertFalse(agent.nodes[0].buggy)
        self.assertEqual(len(pd.read_csv(d / "work/input_small/train.csv")), 50)
        self.assertEqual(len(pd.read_csv(d / "work/nodes/0_pf/valid_predictions.csv")), 200)
        self.assertIn("train rows 2400", agent.nodes[0].output)             # full run used the real input

    def test_preflight_off_for_small_training_sets(self):
        _, agent, res, _ = self.run_one(SCRIPT2.format(crash=""), n=60, preflight_rows=50, preflight_min_rows=100)
        self.assertFalse(res["preflight"])

    def test_search_on_subsample_refit_on_full(self):
        d, agent, res, trace = self.run_one(SCRIPT2.format(crash=""), search_max_rows=1000)
        self.assertEqual(len(pd.read_csv(d / "work/nodes/0/valid_predictions.csv")), 500)   # valid capped at cap/2
        self.assertIn("train rows 1000", agent.nodes[0].output)
        self.assertEqual(res["full_ratio"], 3.0)                            # 3000 full rows / 1000 search rows
        refit = [e for e in trace if e["kind"] == "final_refit"][0]
        self.assertIn("train rows 3000", refit["output"])
        self.assertEqual((res["submission_source"], res["score"]), ("tree_best_refit", 1.0))
        self.assertIn("subsample of the training data", agent.system)


class TestRefitAndDiversity(unittest.TestCase):
    def test_refit_bounded_by_ratio(self):
        t = TestPreflightAndFidelity()
        d, agent, res, trace = t.run_one(SCRIPT2.format(crash=""), search_max_rows=500, refit_max_ratio=2.0)
        refit = [e for e in trace if e["kind"] == "final_refit"][0]
        self.assertEqual((refit["rows"], refit["ratio"]), (1000, 2.0))      # 2x the 500 search rows, not all 3000
        self.assertIn("train rows 1000", refit["output"])
        self.assertEqual(res["submission_source"], "tree_best_refit")

    def test_refit_skipped_without_time(self):
        class NoTimeAtEnd(TreeSearchAgent):
            at_end = False
            def _remaining(self):
                return 21.0 if self.at_end else 300.0   # (21 - 20) / 1.5 < 1.1x: no time
            def _finish(self, stop_reason):
                self.at_end = True
                return super()._finish(stop_reason)
        d = Path(tempfile.mkdtemp())
        task = make_split_task(d / "t", n=3000)
        tracer = Tracer(d / "trace.jsonl")
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = NoTimeAtEnd(ScriptedLLM([reply("go", SCRIPT2.format(crash=""))]), task, sb, tracer,
                                Budget(max_steps=1, time_limit_s=300), SearchConfig(num_drafts=1, search_max_rows=500),
                                seed=0)
            res = agent.run()
        tracer.close()
        refit = [e for e in read_trace(d / "trace.jsonl") if e["kind"] == "final_refit"][0]
        self.assertTrue(refit["skipped"])
        self.assertEqual((res["submission_source"], res["score"]), ("tree_best", 1.0))  # search model still submitted

    def test_diverse_drafts_get_distinct_families(self):
        turns = [reply(f"d{i}", GOOD.format(preds=[.1, .9, .2, .8], val=0.5 + i / 10)) for i in range(3)]
        d = Path(tempfile.mkdtemp())
        task = make_task(d / "t")
        llm = ScriptedLLM(turns)
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = TreeSearchAgent(llm, task, sb, Tracer(None), Budget(max_steps=3, time_limit_s=300),
                                    SearchConfig(num_drafts=3, diverse_drafts=True), seed=0)
            res = agent.run()
        fams = res["draft_families"]
        self.assertEqual(len(set(fams)), 3)
        for i, fam in enumerate(fams):
            self.assertIn(f"Model family for this draft: {fam}", llm.seen[i][1]["content"])

    def test_unterminated_code_block_is_recovered(self):
        from core.search import extract_code
        self.assertEqual(extract_code("plan\n```python\nprint(1)\n```\nmore\n```python\nprint(2)\n```"), "print(2)")
        self.assertEqual(extract_code("plan\n```python\nimport x\nprint(3)"), "import x\nprint(3)")
        self.assertEqual(extract_code("no code here"), "")


class TestHarnessValidation(unittest.TestCase):
    def test_leaky_self_report_loses_to_honest_script(self):
        d = Path(tempfile.mkdtemp())
        task = make_split_task(d / "t")
        turns = [reply("leaky: claims 0.99, predicts anti-signal", SCRIPT.format(pred="-df['x']", claim=0.99)),
                 reply("honest", SCRIPT.format(pred="df['x']", claim=0.5))]
        tracer = Tracer(d / "trace.jsonl")
        with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
            agent = TreeSearchAgent(ScriptedLLM(turns), task, sb, tracer, Budget(max_steps=2, time_limit_s=300),
                                    SearchConfig(num_drafts=2), seed=0)
            res = agent.run()
        tracer.close()
        self.assertEqual(res["val_mode"], "harness")
        self.assertEqual((agent.nodes[0].self_score, agent.nodes[0].score), (0.99, 0.0))
        self.assertEqual(agent.nodes[1].score, 1.0)
        self.assertEqual((res["best_node"], res["score"], res["submission_source"]), (1, 1.0, "tree_best_refit"))
        # labels never reach the sandbox; the refit saw the full training set
        self.assertNotIn("y", pd.read_csv(d / "work/input/valid.csv").columns)
        self.assertEqual(len(pd.read_csv(d / "work/input/train.csv")), 60)
        refit = [e for e in read_trace(d / "trace.jsonl") if e["kind"] == "final_refit"][0]
        self.assertIn("train rows 60", refit["output"])
        self.assertTrue((d / "valid_labels.csv").exists())


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

    def test_submit_any_name_and_no_unvalidated_grading(self):
        # regression: a submission named like the harness's own output file must still be accepted,
        # and a malformed file planted in /work must never be graded
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            (d / "work").mkdir()
            pd.DataFrame({"id": [1, 2, 3, 4], "y": [.1, .9, .2, .8]}).to_csv(d / "work/final_submission.csv", index=False)
            llm = ScriptedLLM([call("submit", path="/work/final_submission.csv")])
            with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
                res = Agent(llm, task, sb, Tracer(None), Budget(max_steps=5)).run()
            self.assertEqual((res["stop_reason"], res["submission_source"]), ("submitted", "submit"))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            (d / "work").mkdir()
            pd.DataFrame({"id": [1], "junk": [0]}).to_csv(d / "work/final_submission.csv", index=False)
            pd.DataFrame({"id": [1], "junk": [0]}).to_csv(d / "work/submission.csv", index=False)
            llm = ScriptedLLM([call("bash", command="true")] * 2)
            with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
                res = Agent(llm, task, sb, Tracer(None), Budget(max_steps=2)).run()
            self.assertIsNone(res["score"])
            self.assertFalse(res["valid_submission"])

    def test_submit_gate_keeps_provisional(self):
        # early valid submit is refused but kept; a later submit after the gate opens is accepted
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            (d / "work").mkdir()
            pd.DataFrame({"id": [1, 2, 3, 4], "y": [.1, .9, .2, .8]}).to_csv(d / "work/submission.csv", index=False)
            llm = ScriptedLLM([call("submit"), call("bash", command="sleep 0.6"), call("submit")])
            budget = Budget(max_steps=10, time_limit_s=1.0, min_submit_frac=0.5)
            with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
                res = Agent(llm, task, sb, Tracer(None), budget).run()
            self.assertEqual((res["stop_reason"], res["submission_source"], res["early_submits"]),
                             ("submitted", "submit", 1))
            self.assertIn("too early", llm.seen[1][-1]["content"])
        # budget runs out after only early submits: the provisional file is graded
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            (d / "work").mkdir()
            pd.DataFrame({"id": [1, 2, 3, 4], "y": [.1, .9, .2, .8]}).to_csv(d / "work/a.csv", index=False)
            llm = ScriptedLLM([call("submit", path="a.csv"), call("bash", command="rm a.csv")])
            budget = Budget(max_steps=2, time_limit_s=60, min_submit_frac=0.5)
            with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
                res = Agent(llm, task, sb, Tracer(None), budget).run()
            self.assertEqual((res["submission_source"], res["score"]), ("provisional", 1.0))

    def test_loop_detection_warns_then_stops(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            llm = ScriptedLLM([call("bash", command="echo same")] * 12)
            with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
                res = Agent(llm, task, sb, Tracer(None), Budget(max_steps=50, loop_warn=3, loop_stop=6)).run()
            self.assertEqual((res["stop_reason"], res["steps"]), ("loop_detected", 6))
            self.assertIn("identical tool call 3 times", llm.seen[3][-1]["content"])

    def test_env_facts_in_system_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task = make_task(d / "t")
            llm = ScriptedLLM([call("bash", command="true")])
            with LocalSandbox("x", "img", d / "work", task.public_dir) as sb:
                Agent(llm, task, sb, Tracer(None), Budget(max_steps=1), use_env_facts=True).run()
            system = llm.seen[0][0]["content"]
            self.assertIn("Installed versions", system)
            self.assertIn("pandas", system)
            self.assertIn("inspect the installed signature", system)

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
