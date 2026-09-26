"""Build task directories.

    python prep.py dev-adult --out data                     # smoke-test task from OpenML (not a benchmark task)
    python prep.py leaf-classification --raw ../kaggle_raw  # Kaggle competition, re-split MLE-bench style

Kaggle competitions keep their real test labels private, so (like MLE-bench) each task is rebuilt from the
competition's *train* data: a fixed-seed held-out part becomes the new test set (labels go to private/answers.csv),
the rest is the new train set. The new test file keeps only the columns of the competition's real test file, and
the submission format is the competition's own. Expects the competition files extracted under <raw>/<name>/x/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from core.tasks import read_table, write_table


def write_task(root: Path, train: pd.DataFrame, test: pd.DataFrame, answers: pd.DataFrame,
               sample: pd.DataFrame, spec: dict, description: str) -> None:
    (root / "public").mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(parents=True, exist_ok=True)
    write_table(train, root / "public" / spec.get("train_file", "train.csv"))
    write_table(test, root / "public" / spec.get("test_file", "test.csv"))
    sample.to_csv(root / "public" / "sample_submission.csv", index=False)
    answers.to_csv(root / "private" / "answers.csv", index=False)
    (root / "task.json").write_text(json.dumps(spec, indent=2))
    (root / "description.md").write_text(description)


def dev_adult(out: Path, raw: Path) -> None:
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


# ---------------------------------------------------------------- Kaggle competitions

def resplit(name: str, raw: Path, out: Path, *, id_col: str, metric: str, higher_is_better: bool,
            test_frac: float, description: str, target_cols: list[str] | None = None,
            label_col: str | None = None, stratify: str | None = None, fmt: str = "csv",
            sample_file: str = "sample_submission.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild a competition as a task with a held-out test split of its train data. Returns (new_train, new_test)."""
    src = raw / name / "x"
    orig_train = read_table(src / f"train.{fmt}")
    test_cols = list(read_table(src / f"test.{fmt}").columns)
    sample = pd.read_csv(src / sample_file)
    targets = target_cols or [c for c in sample.columns if c != id_col]

    new_train, held = train_test_split(orig_train, test_size=test_frac, random_state=0,
                                       stratify=orig_train[stratify] if stratify else None)
    new_test = held[[c for c in test_cols if c in held.columns]]
    if label_col:  # one class-label column -> one-hot answers over the submission's class columns
        onehot = pd.get_dummies(held[label_col]).reindex(columns=targets, fill_value=0).astype(int)
        answers = pd.concat([held[[id_col]], onehot], axis=1)
    else:
        answers = held[[id_col] + targets].copy()
        for c in targets:
            if answers[c].dtype == bool:
                answers[c] = answers[c].astype(int)
    # the competition's own sample row (e.g. 0.5, uniform class probabilities) for every new test id
    new_sample = pd.DataFrame({id_col: held[id_col].values})
    for c in targets:
        new_sample[c] = sample[c].iloc[0]

    spec = {"metric": metric, "id_col": id_col, "target_cols": targets, "higher_is_better": higher_is_better,
            "source": f"kaggle:{name}", "resplit": {"test_frac": test_frac, "seed": 0, "stratify": stratify}}
    if label_col:
        spec["label_col"] = label_col
    if fmt != "csv":
        spec["train_file"], spec["test_file"] = f"train.{fmt}", f"test.{fmt}"
    write_task(out / name, new_train, new_test, answers, new_sample[list(sample.columns)], spec, description)
    return new_train, new_test


def tps_may_2022(out: Path, raw: Path) -> None:
    resplit("tabular-playground-series-may-2022", raw, out, id_col="id", metric="auc", higher_is_better=True,
            test_frac=0.1, stratify="target", description="""# Tabular Playground Series, May 2022

Predict the binary `target` of simulated manufacturing-control data. Features `f_00`..`f_30`; `f_27` is a
10-character string, the rest are numeric (some integer-valued). Feature interactions are known to matter.

- `train.csv`: `id`, features, `target` (0/1).
- `test.csv`: `id`, features.
- Submit `id,target` where `target` is a probability.

Metric: ROC AUC (higher is better).
""")


def tps_dec_2021(out: Path, raw: Path) -> None:
    resplit("tabular-playground-series-dec-2021", raw, out, id_col="Id", metric="accuracy", higher_is_better=True,
            test_frac=0.1, description="""# Tabular Playground Series, Dec 2021

Predict the forest cover type (`Cover_Type`, integer classes 1-7) from synthetic cartographic features, generated
from the original Forest Cover Type data. Classes are very imbalanced (class 5 is extremely rare). ~3.6M training rows.

- `train.csv`: `Id`, features, `Cover_Type`.
- `test.csv`: `Id`, features.
- Submit `Id,Cover_Type` with a predicted class for every row.

Metric: classification accuracy (higher is better).
""")


def nomad2018(out: Path, raw: Path) -> None:
    name = "nomad2018-predict-transparent-conductors"
    train, test = resplit(name, raw, out, id_col="id", metric="rmsle_mean", higher_is_better=False, test_frac=0.2,
                          description="""# Nomad2018: predicting transparent conductors

Predict two properties of (Al_x Ga_y In_z)_2N O_3N materials: `formation_energy_ev_natom` (formation energy per
atom) and `bandgap_energy_ev` (band gap), from composition, spacegroup and lattice parameters.

- `train.csv`: `id`, features, both targets. `test.csv`: `id`, features.
- Atomic positions: `train/<id>/geometry.xyz` and `test/<id>/geometry.xyz` (use the folder matching the file the
  row came from).
- Submit `id,formation_energy_ev_natom,bandgap_energy_ev`.

Metric: RMSLE of each target column, averaged over the two columns (lower is better).
""")
    src = raw / name / "x" / "train"
    for part, df in (("train", train), ("test", test)):
        for i in df["id"]:
            dst = out / name / "public" / part / str(i)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / str(i) / "geometry.xyz", dst / "geometry.xyz")


def leaf(out: Path, raw: Path) -> None:
    name = "leaf-classification"
    train, test = resplit(name, raw, out, id_col="id", metric="logloss", higher_is_better=False, test_frac=0.2,
                          label_col="species", stratify="species", description="""# Leaf classification

Identify the plant species (99 classes, ~8 training samples each) of a leaf from pre-extracted features:
64 margin, 64 shape and 64 texture attributes. Binary leaf images are in `images/<id>.jpg`.

- `train.csv`: `id`, `species`, features. `test.csv`: `id`, features.
- Submit `id` plus one probability column per species, in the column order of `sample_submission.csv`.

Metric: multi-class log loss (lower is better).
""")
    img_dir = out / name / "public" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    keep = {f"images/{i}.jpg" for i in pd.concat([train["id"], test["id"]])}
    with zipfile.ZipFile(raw / name / "x" / "images.zip") as z:
        for member in z.namelist():
            if member in keep:
                (img_dir / Path(member).name).write_bytes(z.read(member))


def spooky(out: Path, raw: Path) -> None:
    resplit("spooky-author-identification", raw, out, id_col="id", metric="logloss", higher_is_better=False,
            test_frac=0.2, label_col="author", stratify="author", description="""# Spooky author identification

Identify the author of sentences from horror stories: Edgar Allan Poe (EAP), H.P. Lovecraft (HPL) or
Mary Wollstonecraft Shelley (MWS).

- `train.csv`: `id`, `text`, `author`. `test.csv`: `id`, `text`.
- Submit `id,EAP,HPL,MWS` with a probability for each author.

Metric: multi-class log loss (lower is better).
""")


def pizza(out: Path, raw: Path) -> None:
    resplit("random-acts-of-pizza", raw, out, id_col="request_id", metric="auc", higher_is_better=True,
            test_frac=0.2, target_cols=["requester_received_pizza"], stratify="requester_received_pizza",
            fmt="json", sample_file="sampleSubmission.csv", description="""# Random Acts of Pizza

Predict whether a Reddit request on r/Random_Acts_Of_Pizza resulted in the requester receiving a pizza,
from the request text and requester metadata.

- `train.json`: list of requests with `requester_received_pizza` (bool) and many fields. Fields ending in
  `_at_retrieval` (and some others) were recorded after the outcome; they are not available for test requests.
- `test.json`: requests with only the fields known at request time.
- Submit `request_id,requester_received_pizza` where the latter is a probability.

Metric: ROC AUC (higher is better).
""")


TASKS = {"dev-adult": dev_adult, "tabular-playground-series-may-2022": tps_may_2022,
         "tabular-playground-series-dec-2021": tps_dec_2021, "nomad2018-predict-transparent-conductors": nomad2018,
         "leaf-classification": leaf, "spooky-author-identification": spooky, "random-acts-of-pizza": pizza}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS) + ["all-kaggle"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--raw", default="../kaggle_raw", help="dir with <competition>/x/ extracted files")
    a = ap.parse_args()
    names = [n for n in TASKS if n != "dev-adult"] if a.task == "all-kaggle" else [a.task]
    for n in names:
        TASKS[n](Path(a.out), Path(a.raw))
        print("ok", Path(a.out) / n)
