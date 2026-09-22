"""Per-run behaviour stats from traces, to tell *why* scores differ, not just by how much.

    python stats.py runs/baseline-*
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

from core.trace import read_trace

API_DRIFT = re.compile(r"unexpected keyword argument|got an unexpected|early_stopping_rounds|verbose_eval")
USES_CV = re.compile(r"KFold|cross_val_score|cross_validate|StratifiedKFold")
VAL_SCORE = re.compile(r"(?:auc|AUC|score)[^0-9\n]{0,25}(0\.\d{3,})")


def run_stats(run: Path) -> dict:
    ev = read_trace(run / "trace.jsonl")
    tools = [e for e in ev if e["kind"] == "tool"]
    errs = [e for e in tools if e["error"]]
    writes = [json.loads(e["args"]).get("path") for e in tools if e["name"] == "write_file"]
    vals = VAL_SCORE.findall(" ".join(e["output"] for e in tools if e["name"] == "bash" and not e["error"]))
    res = json.loads((run / "result.json").read_text())
    return {"run": run.name, "test": res["score"], "last_val": float(vals[-1]) if vals else None,
            "steps": res["steps"], "time_s": res["elapsed_s"], "stop": res["stop_reason"],
            "tool_errors": len(errs), "api_drift_errors": sum(bool(API_DRIFT.search(e["output"])) for e in errs),
            "used_cv": any(USES_CV.search(e["args"]) for e in tools),
            "files_written": len(set(writes)), "submit_calls": sum(e["name"] == "submit" for e in tools),
            "tokens": res["total_prompt_tokens"] + res["completion_tokens"]}


def main() -> None:
    rows = [run_stats(Path(p)) for p in sys.argv[1:] if (Path(p) / "result.json").exists()]
    cols = ["run", "test", "last_val", "steps", "time_s", "stop", "tool_errors", "api_drift_errors",
            "used_cv", "files_written", "submit_calls", "tokens"]
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
