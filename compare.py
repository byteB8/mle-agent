"""Per-task comparison of experiment arms (run tags), with each arm's mean ± sd over seeds.

    python compare.py react treepf              # every task that has runs for these tags
    python compare.py react treepf --runs runs
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(runs: Path, tags: list[str]) -> dict[tuple[str, str], list[dict]]:
    out = defaultdict(list)
    for f in runs.glob("*/result.json"):
        r = json.loads(f.read_text())
        tag = r["run_id"].split("-")[0]
        if tag in tags:
            task = r["run_id"][len(tag) + 1:].rsplit("-s", 1)[0]
            out[(task, tag)].append(r)
    return out


def cell(rs: list[dict]) -> tuple[str, float | None]:
    sc = [r["score"] for r in rs if r["score"] is not None]
    if not sc:
        return f"none (0/{len(rs)})", None
    return f"{st.mean(sc):.4f} ± {st.pstdev(sc):.4f} ({len(sc)}/{len(rs)})", st.mean(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    data = load(Path(a.runs), a.tags)
    tasks = sorted({t for t, _ in data})
    print(f"| task | metric | {' | '.join(a.tags)} | better |")
    print("|---" * (len(a.tags) + 3) + "|")
    for t in tasks:
        any_run = next(rs[0] for (tt, _), rs in data.items() if tt == t)
        hib = any_run["higher_is_better"]
        cells, means = zip(*(cell(data.get((t, tag), [])) for tag in a.tags))
        scored = [(m, tag) for m, tag in zip(means, a.tags) if m is not None]
        best = (max if hib else min)(scored)[1] if scored else "-"
        print(f"| {t} | {any_run['metric']} {'↑' if hib else '↓'} | {' | '.join(cells)} | {best} |")
    print()
    for t in tasks:
        for tag in a.tags:
            rs = sorted(data.get((t, tag), []), key=lambda r: r["run_id"])
            scores = [None if r["score"] is None else round(r["score"], 4) for r in rs]
            extra = ""
            if rs and "nodes" in rs[0]:
                extra = (f" nodes={[r['nodes'] for r in rs]} valid={[r['nodes'] - r['buggy_nodes'] for r in rs]}"
                         f" preflight_rejects={[r.get('preflight_rejects') for r in rs]}"
                         f" source={[r['submission_source'] for r in rs]}")
            else:
                extra = f" time_s={[round(r['elapsed_s']) for r in rs]} stop={[r['stop_reason'] for r in rs]}"
            print(f"{t[:34]:34s} {tag:8s} {scores}{extra}")


if __name__ == "__main__":
    main()
