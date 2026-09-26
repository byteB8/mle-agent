"""Per-run behaviour stats from traces, to tell *why* scores differ, not just by how much.

    python stats.py runs/baseline-*
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from core.trace import read_trace

API_DRIFT = re.compile(r"unexpected keyword argument|got an unexpected|early_stopping_rounds|verbose_eval")
USES_CV = re.compile(r"KFold|cross_val_score|cross_validate|StratifiedKFold")
VAL_SCORE = re.compile(r"(?:auc|AUC|score)[^0-9\n]{0,25}(0\.\d{3,})")


def tree_stats(run: Path, ev: list[dict], res: dict) -> dict:
    nodes = [e for e in ev if e["kind"] == "node"]
    good = [n for n in nodes if not n["buggy"]]
    errors = Counter(re.sub(r"\d+", "N", n["error"])[:70] for n in nodes if n["buggy"])
    return {"run": run.name, "test": res["score"], "best_val": res["best_val"], "nodes": len(nodes),
            "buggy": len(nodes) - len(good), "ops": res["ops"], "best_node": res["best_node"],
            "api_drift_nodes": sum(bool(API_DRIFT.search(n["output"])) for n in nodes if n["buggy"]),
            "median_exec_s": round(st.median(n["exec_s"] for n in nodes), 1) if nodes else None,
            "top_vals": [round(v, 4) for v in sorted((n["score"] for n in good),
                                                    reverse=res["higher_is_better"])[:5]],
            "top_errors": errors.most_common(3), "time_s": res["elapsed_s"],
            "tokens": res["total_prompt_tokens"] + res["completion_tokens"]}


def run_stats(run: Path) -> dict:
    ev = read_trace(run / "trace.jsonl")
    res = json.loads((run / "result.json").read_text())
    if "nodes" in res:
        return tree_stats(run, ev, res)
    tools = [e for e in ev if e["kind"] == "tool"]
    errs = [e for e in tools if e["error"]]
    writes = [json.loads(e["args"]).get("path") for e in tools if e["name"] == "write_file"]
    vals = VAL_SCORE.findall(" ".join(e["output"] for e in tools if e["name"] == "bash" and not e["error"]))
    return {"run": run.name, "test": res["score"], "last_val": float(vals[-1]) if vals else None,
            "steps": res["steps"], "time_s": res["elapsed_s"], "stop": res["stop_reason"],
            "tool_errors": len(errs), "api_drift_errors": sum(bool(API_DRIFT.search(e["output"])) for e in errs),
            "used_cv": any(USES_CV.search(e["args"]) for e in tools),
            "files_written": len(set(writes)), "submit_calls": sum(e["name"] == "submit" for e in tools),
            "early_submits": res.get("early_submits", 0), "source": res.get("submission_source"),
            "tokens": res["total_prompt_tokens"] + res["completion_tokens"]}


def main() -> None:
    rows = [run_stats(Path(p)) for p in sys.argv[1:] if (Path(p) / "result.json").exists()]
    if rows and "nodes" in rows[0]:
        for r in rows:
            fmt = lambda v: "none" if v is None else f"{v:.4f}"
            print(f"{r['run']}: test={fmt(r['test'])} best_val={fmt(r['best_val'])} nodes={r['nodes']} "
                  f"buggy={r['buggy']} ops={r['ops']} best_node={r['best_node']} api_drift_nodes={r['api_drift_nodes']} "
                  f"median_exec={r['median_exec_s']}s tokens={r['tokens']}")
            print(f"    top vals {r['top_vals']}")
            for err, k in r["top_errors"]:
                print(f"    {k} x {err}")
        sc = [r["test"] for r in rows if r["test"] is not None]
        gap = [r["best_val"] - r["test"] for r in rows if r["test"] is not None]
        summary = (f"n={len(sc)} mean={st.mean(sc):.4f} sd={st.pstdev(sc):.4f} mean(best_val - test)={st.mean(gap):+.4f}"
                   if sc else "n=0 valid")
        print(f"\n{summary} buggy_rate="
              f"{sum(r['buggy'] for r in rows) / sum(r['nodes'] for r in rows):.2f}")
        return
    cols = ["run", "test", "last_val", "steps", "time_s", "stop", "tool_errors", "api_drift_errors",
            "used_cv", "files_written", "submit_calls", "early_submits", "source", "tokens"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    sc = [r["test"] for r in rows if r["test"] is not None]
    if sc:
        print(f"\nn={len(sc)} mean={st.mean(sc):.4f} sd={st.pstdev(sc):.4f} "
              f"api_drift_runs={sum(r['api_drift_errors'] > 0 for r in rows)}/{len(rows)} "
              f"cv_runs={sum(r['used_cv'] for r in rows)}/{len(rows)} "
              f"median_time={st.median(r['time_s'] for r in rows):.0f}s")


if __name__ == "__main__":
    main()
