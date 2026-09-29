"""Build a supervised fine-tuning dataset from tree-search traces (phase 2: distil the agent into a small model).

Each tree node is one self-contained LLM call (system + user prompt -> reply with plan and full script), so each
node that did something useful becomes one chat example:

    useful (default)  draft that produced a valid script; debug that turned a broken parent into a valid one;
                      improve whose harness-measured score beat its parent's
    valid             every node with a valid script

    python export.py --runs runs --tags treecost2,treefix2 --out sft.jsonl
    python export.py --runs runs --tags treecost2 --exclude-tasks leaf-classification --filter valid
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from core.trace import read_trace


def better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


def keep(node: dict, nodes: dict[int, dict], hib: bool, mode: str) -> bool:
    if node["buggy"] or not node.get("response"):
        return False
    if mode == "valid":
        return True
    if node["op"] in ("draft", "debug"):
        return True        # a draft that runs, or a fix: parent was broken (debug), child works
    parent = nodes.get(node["parent"])
    return (node["op"] == "improve" and parent is not None and not parent["buggy"]
            and better(node["score"], parent["score"], hib))


def run_examples(run: Path, mode: str) -> list[dict]:
    result = json.loads((run / "result.json").read_text())
    events = read_trace(run / "trace.jsonl")
    start = next((e for e in events if e["kind"] == "episode_start"), None)
    if start is None or start.get("agent") != "tree":
        return []
    nodes = {e["id"]: e for e in events if e["kind"] == "node"}
    out = []
    for n in nodes.values():
        if keep(n, nodes, result["higher_is_better"], mode):
            out.append({"messages": [{"role": "system", "content": start["system"]},
                                     {"role": "user", "content": n["prompt"]},
                                     {"role": "assistant", "content": n["response"]}],
                        "meta": {"run": run.name, "task": start["task"], "op": n["op"], "node": n["id"],
                                 "score": n["score"], "family": n.get("family", "")}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", required=True, help="comma-separated run tags to include")
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", choices=["useful", "valid"], default="useful")
    ap.add_argument("--exclude-tasks", default="", help="comma-separated task names (held out for evaluation)")
    a = ap.parse_args()
    tags = set(a.tags.split(","))
    excluded = {t for t in a.exclude_tasks.split(",") if t}
    examples, seen, dupes = [], set(), 0
    for run in sorted(Path(a.runs).iterdir()):
        if run.name.split("-")[0] not in tags or not (run / "result.json").exists():
            continue
        for ex in run_examples(run, a.filter):
            if ex["meta"]["task"] in excluded:
                continue
            h = hashlib.sha1(ex["messages"][2]["content"].encode()).hexdigest()
            if h in seen:
                dupes += 1
                continue
            seen.add(h)
            examples.append(ex)
    with open(a.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    chars = sum(len(m["content"]) for ex in examples for m in ex["messages"])
    print(f"{len(examples)} examples ({dupes} duplicate replies dropped), ~{chars / 3.5 / 1e6:.2f}M tokens -> {a.out}")
    for key in ("task", "op"):
        print(f"  by {key}: {dict(Counter(ex['meta'][key] for ex in examples).most_common())}")


if __name__ == "__main__":
    main()
