# byoc_source.py
"""
Bring-Your-Own-Clusters (BYOC) data source.

Lets a user feed their own CSV, feature columns plus exactly one cluster-id
column, into the pipeline as a 4th source, on equal footing with clustbench /
mdcgen / fabricated_data. The user brings *clusters* (a partition they already have),
not raw data to be clustered; the CLM engine then synthesizes labels against that
partition so they can study how their labels relate to their own clusters.

Contract (mirrors the other fetchers, returns the standard frame or None):
    * the CSV path comes from the config (byoc_suite.datasets), never a prompt;
    * exactly one cluster column, named by `cluster_column`; rejected otherwise;
    * every other numeric column is treated as a feature (original names kept);
    * `standardize: true` min-max rescales the features to [0, 1] at import time
      (documented, opt-in), it is applied here, not in centroid detection.
"""

import csv
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Import requirements
#
# BYOC is an IMPORT path, not a generator: the user has already clustered a
# feature subset with their own algorithm and is bringing the result. These
# checks encode what that implies about the file. They are deliberately NOT
# [CLM-###] diagnostics -- those describe the cluster-label matching model,
# while these describe whether a file is a usable clustering at all.
#
# Expected to grow. The manual carries the same list under "BYOC input
# requirements"; keep the two in step.
# --------------------------------------------------------------------------- #

# Names the pipeline itself writes. A user column sharing one is silently
# consumed (Cohort_Class, GroundTruth_*) or produces a duplicate column in the
# output CSV (Cluster_n, Label_n).
RESERVED_EXACT = {"Cohort_Class"}
RESERVED_PREFIX = "GroundTruth_"
RESERVED_PATTERN = re.compile(r"^(Cluster|Label)_\d+$")

# A cluster of one or two points is not a cluster any algorithm meant to
# produce; it is a stray. The engine's maths works on it, which is the problem:
# it would quietly proceed on a partition the user did not intend.
MIN_CLUSTER_SIZE = 3


def _raw_header(path: Path) -> list[str]:
    """The header exactly as written, before pandas renames duplicates.

    `read_csv` mangles a repeated `f1` into `f1.1`, so by the time a frame
    exists the collision is invisible. Read via csv.reader rather than splitting
    on commas so a quoted field containing one does not read as two columns.
    """
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), [])
    except OSError:
        return []


def validate_import(df: pd.DataFrame, header: list[str], cluster_column: str) -> list[str]:
    """Every reason this frame is not a usable BYOC import, or an empty list.

    All checks run, so one pass reports everything wrong with a file rather than
    making the user fix problems one at a time.
    """
    problems = []

    if df.empty or not len(df.columns):
        problems.append(
            f"the file has no data rows ({len(df)} row(s), {len(df.columns)} column(s)); "
            "a header alone is not a clustering")
        return problems                       # nothing below can say anything useful

    duplicates = sorted({h for h in header if header.count(h) > 1})
    if duplicates:
        problems.append(
            f"duplicate column name(s) {duplicates}: pandas renames the second to "
            "'name.1', so the column you meant is ambiguous. Give each column a "
            "distinct name")

    reserved = sorted(c for c in df.columns
                      if c in RESERVED_EXACT or str(c).startswith(RESERVED_PREFIX)
                      or RESERVED_PATTERN.match(str(c)))
    if reserved:
        problems.append(
            f"column name(s) {reserved} are reserved by the pipeline: 'Cohort_Class' "
            "and 'GroundTruth_*' are consumed as ground truth, and 'Cluster_n'/'Label_n' "
            "are written into the output. Rename them")

    if cluster_column in df.columns:
        clusters = df[cluster_column]
        if clusters.isna().any():
            problems.append(
                f"the cluster column '{cluster_column}' has {int(clusters.isna().sum())} "
                "missing value(s): every point must belong to a cluster, and a blank "
                "would become a cluster of its own")
        else:
            sizes = clusters.value_counts()
            if len(sizes) < 2:
                problems.append(
                    f"the cluster column '{cluster_column}' holds {len(sizes)} distinct "
                    "value(s): a single partition has no structure to match a label against")
            undersized = sizes[sizes < MIN_CLUSTER_SIZE]
            if len(undersized):
                shown = ", ".join(f"{k!r}={v}" for k, v in list(undersized.items())[:5])
                problems.append(
                    f"{len(undersized)} cluster(s) hold fewer than {MIN_CLUSTER_SIZE} "
                    f"points ({shown}). Clusters that small are strays rather than "
                    "clusters; merge or drop them before importing")

        features = df.drop(columns=[cluster_column])
        non_numeric = [c for c in features.columns
                       if not pd.api.types.is_numeric_dtype(features[c])]
        if non_numeric:
            problems.append(
                f"non-numeric feature column(s) {non_numeric}: every column other than "
                f"'{cluster_column}' is treated as a feature, and features define the "
                "geometry the clustering was computed in, so they must be numeric")
        elif features.isna().to_numpy().any():
            bad = [c for c in features.columns if features[c].isna().any()]
            problems.append(
                f"missing value(s) in feature column(s) {bad}: the clustering cannot have "
                "been computed from them, and centroid distances would come out NaN")

    return problems


def fetch_byoc_data(
        dataset_group: str = "local",
        dataset_name: str | None = None,
        seed: int = 42,
        cluster_column: str | None = None,
        standardize: bool = False,
        input_dir: str | None = None,
        **kwargs,
) -> pd.DataFrame | None:
    """Loads a user CSV as a dataset: numeric feature columns plus exactly
    one cluster-id column (`cluster_column`), optionally min-max standardized.
    Returns the standard fetcher frame, or None on any rejected input."""
    # --- resolve the CSV path from the config ---
    # `datasets` entries are file STEMS (no extension); '.csv' is appended and the
    # folder comes from `input_dir`. Stems (not full paths) keep the run-folder /
    # output filenames predictable, since main.py builds them from the dataset name.
    if not dataset_name:
        log.error("byoc: no CSV given, list your file stem(s) under byoc_suite.datasets.")
        return None
    name = str(dataset_name)
    path = (Path(input_dir) / f"{name}.csv") if input_dir else Path(f"{name}.csv")
    if not path.is_file():
        log.error(f"byoc: CSV not found: '{path}'. List file stems (without .csv) in "
                  "byoc_suite.datasets and the folder in byoc_suite.input_dir.")
        return None

    # --- exactly one cluster column, named explicitly ---
    if not cluster_column or not isinstance(cluster_column, str):
        log.error("byoc: 'cluster_column' must name exactly one cluster-id column "
                  "(a single string) in byoc_suite.")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(f"byoc: failed to read '{path}': {e}")
        return None

    if cluster_column not in df.columns:
        log.error(f"byoc: cluster_column '{cluster_column}' not found in '{path.name}'. "
                  f"Columns present: {list(df.columns)}.")
        return None

    # Every requirement is checked in one pass, so a file with several problems
    # reports all of them rather than one per attempt.
    problems = validate_import(df, _raw_header(path), cluster_column)
    if problems:
        log.error(f"byoc: '{path.name}' is not a usable import, {len(problems)} problem(s):")
        for problem in problems:
            log.error(f"  - {problem}")
        return None

    # --- features = every OTHER column that is numeric (original names kept) ---
    other_cols = [c for c in df.columns if c != cluster_column]
    numeric = df[other_cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped = [c for c in other_cols if c not in numeric]
    if dropped:
        log.warning(f"byoc: ignoring non-numeric column(s) {dropped} "
                    "(features must be numeric to define geometry).")
    if not numeric:
        log.error(f"byoc: no numeric feature columns found in '{path.name}' "
                  "besides the cluster column.")
        return None

    features = df[numeric].copy()

    # --- optional min-max standardization to [0, 1], applied at import ---
    if standardize:
        lo = features.min()
        span = (features.max() - lo).replace(0, 1.0)   # guard constant columns
        features = (features - lo) / span
        log.info(f"byoc: standardized {len(numeric)} feature(s) to [0, 1].")

    out = features
    # One cluster labeling -> GroundTruth_labels0 (surfaces as Cluster_0 downstream).
    out["GroundTruth_labels0"] = df[cluster_column].to_numpy()
    out["Cohort_Class"] = df[cluster_column].to_numpy()

    log.info(f"byoc: loaded '{path.name}': {len(out)} rows, {len(numeric)} feature(s), "
             f"1 cluster labeling from column '{cluster_column}' "
             f"({df[cluster_column].nunique()} clusters).")
    return out
