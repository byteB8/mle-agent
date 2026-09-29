"""Structured data preview for the agent's prompt, computed by the harness from the real input files.

A raw `head` of a file hides its structure: for an indented JSON array it shows `[`, `{` and a couple of
keys, and the model guesses (e.g. parses it as JSON-lines). This summary states the format, row count,
every field with its type and an example, which fields are targets, and which fields exist only in train.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .tasks import read_table


def _kind(s: pd.Series) -> str:
    v = s.dropna()
    if len(v) and isinstance(v.iloc[0], (list, dict)):
        return type(v.iloc[0]).__name__
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if len(v) and v.astype(str).str.len().median() > 40:
        return f"text (median {int(v.astype(str).str.len().median())} chars)"
    return "str"


def _plain(x):
    return x.item() if hasattr(x, "item") and not isinstance(x, (list, dict, str)) else x


def _example(s: pd.Series, width: int = 60) -> str:
    v = s.dropna()
    if not len(v):
        return "all missing"
    r = repr(_plain(v.iloc[0]))
    return r if len(r) <= width else r[:width] + "…"


def _groups(df: pd.DataFrame) -> "OrderedDict[str, list[str]]":
    """Group runs of columns like margin1..margin64 (same prefix, numbered) so wide tables stay readable."""
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for c in df.columns:
        m = re.fullmatch(r"(.*?[A-Za-z_])(\d+)", str(c))
        key = f"{m.group(1)}#" if m else str(c)
        groups.setdefault(key, []).append(c)
    return groups


def describe(path: Path, shown_as: str, targets: set[str], train_only: set[str], max_lines: int = 45) -> str:
    df = read_table(path)
    if path.suffix == ".json":
        head = (f"== {shown_as}: JSON array of {len(df)} records (a single JSON list, NOT JSON-lines; "
                f"load with pd.read_json(path) or json.load)")
    else:
        head = f"== {shown_as}: CSV, {len(df)} rows × {df.shape[1]} columns"
    lines = [head]

    def field(c) -> str:
        tag = "  [TARGET: only in train]" if c in targets else ("  [only in train: not a usable feature]"
                                                              if c in train_only else "")
        kind, s = _kind(df[c]), df[c]
        missing = s.isna().mean()
        miss = f", {missing:.0%} missing" if missing > 0.01 else ""
        uniq = ""
        if kind == "str" or kind.startswith("text"):
            try:
                uniq = f", {s.nunique()} unique"
            except TypeError:
                pass
        return f"   - {c}: {kind}{miss}{uniq}, e.g. {_example(s)}{tag}"

    for key, cols in _groups(df).items():
        if len(cols) >= 4 and key.endswith("#"):
            kinds = [_kind(df[c]) for c in cols]
            major = max(set(kinds), key=kinds.count)
            odd = [c for c, k in zip(cols, kinds) if k != major]
            note = f" ({len(cols) - len(odd)} {major}; the others listed below)" if odd else f", {major}"
            lines.append(f"   - {cols[0]}..{cols[-1]}: {len(cols)} columns{note}")
            lines += [field(c) for c in odd if c not in targets]
            lines += [field(c) for c in cols if c in targets]
            continue
        lines += [field(c) for c in cols]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"   ... {len(lines) - max_lines} more fields"]
    return "\n".join(lines)


def data_preview(files: "OrderedDict[str, Path]", targets: list[str], sample_submission: Path) -> str:
    """files: shown name -> host path, with the training file first."""
    names = list(files)
    frames_cols = {n: set(read_table(files[n]).columns) for n in names}
    train_cols = frames_cols[names[0]]
    others = [frames_cols[n] for n in names[1:]]
    train_only = (train_cols - set.intersection(*others)) if others else set()
    parts = [describe(files[names[0]], names[0], set(targets), train_only - set(targets))]
    for n in names[1:]:
        extra = sorted(frames_cols[n] - train_cols)
        note = f" (also has: {extra})" if extra else ""
        parts.append(f"== {n}: {len(read_table(files[n]))} rows, same fields as train minus the "
                     f"train-only ones{note}")
    if train_only:
        parts.append(f"Fields in train but NOT in the other files (targets or after-the-fact data; do not use as "
                     f"features): {sorted(train_only)}")
    sub = pd.read_csv(sample_submission, nrows=3)
    cols = list(sub.columns)
    shown = cols if len(cols) <= 12 else cols[:6] + [f"... {len(cols) - 6} more"]
    row = {c: _plain(sub[c].iloc[0]) for c in cols[:6]}
    parts.append(f"== sample_submission.csv: {len(cols)} columns {shown}; first row starts {row}")
    return "\n".join(parts)
