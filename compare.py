"""Per-task comparison of experiment arms (run tags), with each arm's mean ± sd over seeds, plus a
normalised score so tasks with different metrics can be averaged:

    normalised = (score - sample_submission_score) / (perfect_score - sample_submission_score)

0 = no better than submitting the competition's sample file, 1 = perfect; negative = worse than the sample file.

    python compare.py react treepf              # every task that has runs for these tags
    python compare.py react treepf --runs runs --data data
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


PERFECT = {"auc": 1.0, "accuracy": 1.0, "logloss": 0.0, "rmse": 0.0, "rmsle_mean": 0.0, "mae": 0.0}


def chance_score(data_dir: Path, task: str) -> float | None:
    """Score of the competition's own sample submission on the held-out test answers."""
    from core.tasks import Task
    root = data_dir / task
    if not (root / "task.json").exists():
        return None
    t = Task.load(root)
    return t.grade(t.public_dir / "sample_submission.csv")


def normalise(score: float, chance: float, metric: str) -> float:
    return (score - chance) / (PERFECT[metric] - chance)


def cell(rs: list[dict]) -> tuple[str, float | None]:
    sc = [r["score"] for r in rs if r["score"] is not None]
    if not sc:
        return f"none (0/{len(rs)})", None
    return f"{st.mean(sc):.4f} ± {st.pstdev(sc):.4f} ({len(sc)}/{len(rs)})", st.mean(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data", help="task dirs, for the sample-submission (chance) score")
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
    # normalised: mean over tasks of the per-task mean, and of the per-task worst seed
    norm = {tag: ([], []) for tag in a.tags}
    for t in tasks:
        any_run = next(rs[0] for (tt, _), rs in data.items() if tt == t)
        chance = chance_score(Path(a.data), t)
        if chance is None or any_run["metric"] not in PERFECT:
            continue
        for tag in a.tags:
            n = [normalise(r["score"], chance, any_run["metric"]) if r["score"] is not None else 0.0
                 for r in data.get((t, tag), [])]   # a run with no valid submission counts as the sample file
            if n:
                norm[tag][0].append(st.mean(n))
                norm[tag][1].append(min(n))
    if all(norm[tag][0] for tag in a.tags):
        k = len(norm[a.tags[0]][0])
        print(f"| **normalised, mean over {k} tasks** | 0 = sample file, 1 = perfect | "
              + " | ".join(f"{st.mean(norm[tag][0]):.3f} (worst seed {st.mean(norm[tag][1]):.3f})" for tag in a.tags)
              + " | |")
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
