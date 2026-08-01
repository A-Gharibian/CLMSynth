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

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fetch_byoc_data(
        dataset_group: str = "local",
        dataset_name: Optional[str] = None,
        seed: int = 42,
        cluster_column: Optional[str] = None,
        standardize: bool = False,
        input_dir: Optional[str] = None,
        **kwargs,
) -> Optional[pd.DataFrame]:
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
