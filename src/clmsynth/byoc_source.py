# byoc_source.py
"""
Bring-Your-Own-Clusters (BYOC) data source.

Lets a user feed their own CSV, feature columns plus one cluster-id
column, into the pipeline.

Contract (mirrors the other fetchers, returns the standard frame or None):
    * the CSV path comes from the config (byoc_suite.datasets), never a prompt;
    * exactly one cluster column, named by `cluster_column`; rejected otherwise;
    * every other column is a feature unless named in `tag_columns`, which are
      carried to the output CSV untouched and kept out of the geometry;
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
# checks encode what that implies about the file.
# --------------------------------------------------------------------------- #

# Names the pipeline itself writes. A user column sharing one is silently
# consumed (Cohort_Class, GroundTruth_*) or produces a duplicate column in the
# output CSV (Cluster_n, Label_n).
RESERVED_EXACT = {"Cohort_Class"}
# Transport prefix: tags travel to build_context under it and lose it again
# in the output CSV, the way GroundTruth_* becomes Cluster_n.
TAG_PREFIX = "TagColumn_"
RESERVED_PREFIXES = ("GroundTruth_", TAG_PREFIX)
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


def as_tag_columns(value) -> list[str]:
    """Declared tag columns as a list; one bare name is one column."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def validate_import(df: pd.DataFrame, header: list[str], cluster_column: str,
                    tag_columns: list[str] | None = None) -> list[str]:
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
                      if c in RESERVED_EXACT or str(c).startswith(RESERVED_PREFIXES)
                      or RESERVED_PATTERN.match(str(c)))
    if reserved:
        problems.append(
            f"column name(s) {reserved} are reserved by the pipeline: 'Cohort_Class' "
            "and 'GroundTruth_*' are consumed as ground truth, 'TagColumn_*' carries "
            "declared tags, and 'Cluster_n'/'Label_n' are written into the output. "
            "Rename them")

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

        tags = as_tag_columns(tag_columns)
        absent = [t for t in tags if t not in df.columns]
        if absent:
            problems.append(
                f"byoc_suite.tag_columns names column(s) {absent} that are not in "
                "the file. A tag that is not there is a typo rather than an empty "
                "passenger, so it is refused the way a missing cluster_column is")
        if cluster_column in tags:
            problems.append(
                f"byoc_suite.tag_columns names the cluster column '{cluster_column}'. "
                "A column is either the partition being matched against or a "
                "passenger carried beside it, and it cannot be both")

        carried = [t for t in dict.fromkeys(tags) if t in df.columns]
        features = df.drop(columns=[cluster_column, *carried])
        non_numeric = [c for c in features.columns
                       if not pd.api.types.is_numeric_dtype(features[c])]
        if non_numeric:
            problems.append(
                f"non-numeric feature column(s) {non_numeric}: every column other than "
                f"'{cluster_column}' is a feature unless byoc_suite.tag_columns names "
                "it, and features define the geometry the clustering was computed in, "
                "so they must be numeric. List them under tag_columns to carry them "
                "through untouched instead")
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
        tag_columns=None,
        **kwargs,
) -> pd.DataFrame | None:
    """Loads a user CSV as a dataset: numeric feature columns plus exactly
    one cluster-id column (`cluster_column`), optionally min-max standardized.
    Columns named in `tag_columns` ride along untouched, out of the geometry.
    Returns the standard fetcher frame, or None on any rejected input."""
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
    tags = as_tag_columns(tag_columns)
    problems = validate_import(df, _raw_header(path), cluster_column, tags)
    if problems:
        log.error(f"byoc: '{path.name}' is not a usable import, {len(problems)} problem(s):")
        for problem in problems:
            log.error(f"  - {problem}")
        return None

    # --- features = every OTHER column not declared a tag (names kept) ---
    # validate_import already refused any non-numeric one, so there is no second,
    # disagreeing policy here: what it accepted is what becomes geometry.
    numeric = [c for c in df.columns if c != cluster_column and c not in tags]
    if not numeric:
        log.error(f"byoc: no feature columns found in '{path.name}' besides the "
                  "cluster column and the declared tag column(s).")
        return None

    features = df[numeric].copy()
    # bool passes is_numeric_dtype but is not np.number; make it geometry.
    boolean = [c for c in features.columns if features[c].dtype == bool]
    if boolean:
        features = features.astype(dict.fromkeys(boolean, np.int8))

    # --- optional min-max standardization to [0, 1], applied at import ---
    if standardize:
        lo = features.min()
        span = (features.max() - lo).replace(0, 1.0)   # guard constant columns
        features = (features - lo) / span
        log.info(f"byoc: standardized {len(numeric)} feature(s) to [0, 1].")

    out = features
    for tag in dict.fromkeys(tags):
        out[f"{TAG_PREFIX}{tag}"] = df[tag].to_numpy()
    # One cluster labeling -> GroundTruth_labels0 (surfaces as Cluster_0 downstream).
    out["GroundTruth_labels0"] = df[cluster_column].to_numpy()
    out["Cohort_Class"] = df[cluster_column].to_numpy()

    also = f", {len(set(tags))} tag column(s) {sorted(set(tags))}" if tags else ""
    log.info(f"byoc: loaded '{path.name}': {len(out)} rows, {len(numeric)} feature(s), "
             f"1 cluster labeling from column '{cluster_column}' "
             f"({df[cluster_column].nunique()} clusters){also}.")
    return out
