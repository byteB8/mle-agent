"""Build task directories.

    python prep.py dev-adult --out data     # smoke-test task from OpenML (not a benchmark task)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def write_task(root: Path, train: pd.DataFrame, test: pd.DataFrame, answers: pd.DataFrame,
               sample: pd.DataFrame, spec: dict, description: str) -> None:
    (root / "public").mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "public" / "train.csv", index=False)
    test.to_csv(root / "public" / "test.csv", index=False)
    sample.to_csv(root / "public" / "sample_submission.csv", index=False)
    answers.to_csv(root / "private" / "answers.csv", index=False)
    (root / "task.json").write_text(json.dumps(spec, indent=2))
    (root / "description.md").write_text(description)


def dev_adult(out: Path) -> None:
    from sklearn.datasets import fetch_openml
    df = fetch_openml("adult", version=2, as_frame=True).frame
    df["income"] = (df.pop("class").astype(str) == ">50K").astype(int)
    df.insert(0, "id", range(len(df)))
    tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df["income"])
    write_task(out / "dev-adult", tr, te.drop(columns="income"), te[["id", "income"]],
               te[["id"]].assign(income=0.5),
               {"metric": "auc", "id_col": "id", "target_cols": ["income"], "higher_is_better": True},
               "# Adult income\n\nPredict the probability that a person's income exceeds $50K/year from census "
               "attributes.\n\n- `train.csv`: features plus binary target `income`.\n- `test.csv`: features only.\n"
               "- Submit `id,income` where `income` is a probability.\n\nMetric: ROC AUC (higher is better).\n")


TASKS = {"dev-adult": dev_adult}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    TASKS[a.task](Path(a.out))
    print("ok", Path(a.out) / a.task)
