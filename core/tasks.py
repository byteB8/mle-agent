"""Task spec, submission validation and grading.

Layout of a task directory:
    task.json            {"metric", "id_col", "target_cols", "higher_is_better"} and optionally
                         "label_col" (train holds one class-label column; answers are one-hot over target_cols),
                         "train_file" / "test_file" (default train.csv / test.csv; .csv or .json records)
    description.md       what the agent is told
    public/              mounted read-only at /data (train, test, sample_submission.csv)
    private/answers.csv  held-out labels; never mounted into the sandbox
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_json(path) if path.suffix == ".json" else pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".json":
        df.to_json(path, orient="records", indent=1)
    else:
        df.to_csv(path, index=False)


def _metric(name: str, y_true: pd.DataFrame, y_pred: pd.DataFrame, target_cols: list[str]) -> float:
    from sklearn import metrics as M
    if name == "auc":
        return float(M.roc_auc_score(y_true[target_cols[0]], y_pred[target_cols[0]]))
    if name == "accuracy":
        return float(M.accuracy_score(y_true[target_cols[0]].astype(str), y_pred[target_cols[0]].astype(str)))
    if name == "rmse":
        return float(np.sqrt(M.mean_squared_error(y_true[target_cols], y_pred[target_cols])))
    if name == "rmsle_mean":  # column-wise RMSLE, averaged over target columns
        return float(np.mean([np.sqrt(M.mean_squared_log_error(y_true[c], np.clip(y_pred[c], 0, None)))
                              for c in target_cols]))
    if name == "mae":
        return float(M.mean_absolute_error(y_true[target_cols], y_pred[target_cols]))
    if name == "logloss":  # multiclass: answers hold one-hot columns, submission holds probabilities
        p = y_pred[target_cols].to_numpy(dtype=float)
        p = np.clip(p / p.sum(axis=1, keepdims=True), 1e-15, 1 - 1e-15)
        return float(M.log_loss(y_true[target_cols].to_numpy().argmax(1), p, labels=range(len(target_cols))))
    raise ValueError(f"unknown metric {name}")


@dataclass
class Task:
    name: str
    root: Path
    metric: str
    id_col: str
    target_cols: list[str]
    higher_is_better: bool
    description: str
    label_col: str | None = None
    train_file: str = "train.csv"
    test_file: str = "test.csv"

    @classmethod
    def load(cls, root: str | Path) -> "Task":
        root = Path(root).resolve()
        spec = json.loads((root / "task.json").read_text())
        return cls(name=root.name, root=root, metric=spec["metric"], id_col=spec["id_col"],
                   target_cols=list(spec["target_cols"]), higher_is_better=bool(spec["higher_is_better"]),
                   description=(root / "description.md").read_text(), label_col=spec.get("label_col"),
                   train_file=spec.get("train_file", "train.csv"), test_file=spec.get("test_file", "test.csv"))

    @property
    def public_dir(self) -> Path:
        return self.root / "public"

    @property
    def answers(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "private" / "answers.csv")

    def validate_submission(self, path: Path, expected_ids: pd.Series | None = None) -> list[str]:
        """Format checks only (the agent sees these messages); never reveals labels.
        `expected_ids` checks against another id set (e.g. a harness validation split) instead of the test ids."""
        if not path.exists():
            return [f"{path.name} does not exist"]
        try:
            sub = pd.read_csv(path)
        except Exception as e:
            return [f"could not parse CSV: {e}"]
        sample = pd.read_csv(self.public_dir / "sample_submission.csv")
        if expected_ids is not None:
            sample = pd.DataFrame({self.id_col: expected_ids.values}).assign(
                **{c: 0 for c in sample.columns if c != self.id_col})[list(sample.columns)]
        errors = []
        if list(sub.columns) != list(sample.columns):
            errors.append(f"columns {list(sub.columns)} != expected {list(sample.columns)}")
        if len(sub) != len(sample):
            errors.append(f"{len(sub)} rows, expected {len(sample)}")
        if self.id_col in sub.columns:
            if sub[self.id_col].duplicated().any():
                errors.append("duplicate ids")
            if set(sub[self.id_col].astype(str)) != set(sample[self.id_col].astype(str)):
                errors.append("ids do not match the expected ids" if expected_ids is not None
                              else "ids do not match sample_submission ids")
        present = [c for c in self.target_cols if c in sub.columns]
        if present and sub[present].isna().any().any():
            errors.append("predictions contain NaN")
        return errors

    @property
    def can_split(self) -> bool:
        """Whether the harness can hold out labelled validation rows from the public train file."""
        train, test = self.public_dir / self.train_file, self.public_dir / self.test_file
        if not (train.exists() and test.exists()):
            return False
        cols = read_table(train).columns
        labels = [self.label_col] if self.label_col else self.target_cols
        return self.id_col in cols and all(c in cols for c in labels)

    def split_train(self, seed: int, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """(train part with labels, valid features, valid labels) from the public train file.
        Valid features keep only the columns the test file has, so validation rows look like test rows
        (e.g. no fields that are only known after the fact)."""
        df = read_table(self.public_dir / self.train_file)
        test_cols = read_table(self.public_dir / self.test_file).columns
        valid = df.sample(frac=valid_frac, random_state=seed)
        train = df.drop(valid.index)
        valid_x = valid[[c for c in test_cols if c in valid.columns]]
        if self.label_col:
            onehot = pd.get_dummies(valid[self.label_col]).reindex(columns=self.target_cols, fill_value=0)
            valid_y = pd.concat([valid[[self.id_col]], onehot.astype(int)], axis=1)
        else:
            valid_y = valid[[self.id_col] + self.target_cols]
        return train, valid_x, valid_y

    def grade(self, path: Path, answers: pd.DataFrame | None = None) -> float:
        """Score a prediction file against the private test answers (default) or any given labels."""
        sub = pd.read_csv(path)
        ans = (self.answers if answers is None else answers).copy()
        sub[self.id_col] = sub[self.id_col].astype(str)
        ans[self.id_col] = ans[self.id_col].astype(str)
        merged = ans.merge(sub, on=self.id_col, suffixes=("", "_pred"), how="left")
        y_pred = merged[[f"{c}_pred" for c in self.target_cols]].set_axis(self.target_cols, axis=1)
        return _metric(self.metric, merged[self.target_cols], y_pred, self.target_cols)
