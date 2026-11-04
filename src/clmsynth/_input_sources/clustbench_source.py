# SPDX-License-Identifier: MIT
# clustbench_source.py
"""The clustbench source: Gagolewski's clustering benchmark datasets, fetched over
HTTP(S), with their batteries and descriptions. ``dataset_sources`` is its interface
and re-exports these names."""

import gzip
import io
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Final

import numpy as np
import pandas as pd

from . import Source

log: logging.Logger = logging.getLogger(__name__)

# Timeout (seconds) for each clustbench network fetch, so a hung GitHub endpoint
# can't stall the pipeline forever (no circuit breaker otherwise).
CLUSTBENCH_TIMEOUT = 30

CLUSTBENCH_METADATA: Final[Mapping[str, Mapping[str, str]]] = {
    "wut": {
        "description": "Warsaw University of Technology. Covers spatial, density-based, and overlapping clusters.",
        "examples": "smile, x1, x2, twonorm"
    },
    "sipu": {
        "description": "SIPU datasets (Fränti & Sieranoja). Features varying densities, high dimensions, and large cluster numbers.",
        "examples": "a1, s1, unbalance, birch1"
    },
    "fcps": {
        "description": "Fundamental Clustering Problem Suite (A. Ultsch). Classic 2D/3D datasets with distinct geometric challenges.",
        "examples": "atom, chainlink, hepta, lsun"
    },
    "graves": {
        "description": "Features complex shapes, heavy noise, and non-linear boundaries (like zigzags).",
        "examples": "zigzag_noisy, ring_noisy, dense_disk"
    },
    "other": {
        "description": "Assorted community datasets, including those specifically designed to test HDBSCAN.",
        "examples": "hdbscan, chameleon_t8_8k"
    },
    "uci": {
        "description": "UCI Machine Learning Repository. Eight real classification datasets in 7 to 60 "
                       "dimensions, whose classes clustering often finds hard.",
        "examples": "ecoli, wine, yeast"
    },
    "mnist": {
        "description": "MNIST handwritten digits and Fashion-MNIST articles: 28x28 greyscale images "
                       "as 784 features, 10 classes.",
        "examples": "digits, fashion"
    },
    "g2mg": {
        "description": "Two equal-sized Gaussian clusters, 2048 points in 1 to 128 dimensions, named "
                       "g2mg_<dim>_<scale>; the _1_ ones have a single feature.",
        "examples": "g2mg_2_10, g2mg_4_20"
    },
    "h2mg": {
        "description": "Two equal-sized Gaussian-like hubs, each point on a sphere around its "
                       "cluster's center, 2048 points in 1 to 128 dimensions, named "
                       "h2mg_<dim>_<scale>; the _1_ ones have a single feature.",
        "examples": "h2mg_2_10, h2mg_4_20"
    },
}

CLUSTBENCH_DATASETS: Final[Mapping[str, Sequence[str]]] = {
    "wut": [
        "circles", "cross", "graph", "isolation", "labirynth", "mk1", "mk2",
        "mk3", "mk4", "olympic", "smile", "stripes", "trajectories",
        "trapped_lovers", "twosplashes", "windows", "x1", "x2", "x3",
        "z1", "z2", "z3",
    ],
    "sipu": [
        "a1", "a2", "a3", "aggregation", "birch1", "birch2", "compound",
        "d31", "flame", "jain", "pathbased", "r15", "s1", "s2", "s3", "s4",
        "spiral", "unbalance", "worms_2", "worms_64",
    ],
    "fcps": [
        "atom", "chainlink", "engytime", "hepta", "lsun", "target",
        "tetra", "twodiamonds", "wingnut",
    ],
    "graves": [
        "dense", "fuzzyx", "line", "parabolic", "ring", "ring_noisy",
        "ring_outliers", "zigzag", "zigzag_noisy", "zigzag_outliers",
    ],
    "other": [
        "chameleon_t4_8k", "chameleon_t5_8k", "chameleon_t7_10k",
        "chameleon_t8_8k", "hdbscan", "iris", "iris5", "square",
    ],
    "uci": [
        "ecoli", "glass", "ionosphere", "sonar", "statlog", "wdbc",
        "wine", "yeast",
    ],
    "mnist": ["digits", "fashion"],
    "g2mg": [f"g2mg_{d}_{s}" for d in (1, 2, 4, 8, 16, 32, 64, 128)
             for s in range(10, 91, 10)],
    "h2mg": [f"h2mg_{d}_{s}" for d in (1, 2, 4, 8, 16, 32, 64, 128)
             for s in range(10, 91, 10)],
}

CLUSTBENCH_HEAVY: Final[AbstractSet[str]] = {"mnist", "g2mg", "h2mg"}


def _loadtxt_url(url: str, **loadtxt_kwargs) -> np.ndarray:
    """
    Fetches a gzipped text array over HTTP(S) with a timeout, then parses it.

    `base_url` reaches here from the config, and a config is a shareable
    artifact (see SECURITY.md), so a `file://` or `data:` URL would turn a
    dataset fetch into a local read. Only http(s) is opened.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(
            f"clustbench: refusing to fetch {url!r}: only http(s) is fetched, "
            f"not {scheme!r}. Check clustbench_suite.base_url.")
    with urllib.request.urlopen(url, timeout=CLUSTBENCH_TIMEOUT) as resp:  # nosec B310
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        return np.loadtxt(stream, **loadtxt_kwargs)


def fetch_clustbench_labelings(
        dataset_group: str,
        dataset_name: str,
        base_url: str = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0",
        max_labelings: int = 10,
) -> dict[str, np.ndarray]:
    """Downloads every reference labeling (labels0, labels1, ...) of one
    clustbench dataset, up to the first index the server does not have (404).
    A labeling that fails otherwise is skipped and named in one warning."""
    target_path = f"{base_url}/{dataset_group}/{dataset_name}"
    labeling: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for i in range(max_labelings):
        label_name = f"labels{i}"
        try:
            labeling[label_name] = _loadtxt_url(f"{target_path}.{label_name}.gz", dtype="int")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                failed.append(f"{label_name} ({e})")
                continue
            if i == 0:
                log.error(f"No reference labels found for {dataset_group}/{dataset_name}.")
            break
        except Exception as e:
            failed.append(f"{label_name} ({e})")
    if failed:
        log.warning(f"{dataset_group}/{dataset_name}: could not fetch {', '.join(failed)}; "
                    f"continuing with {len(labeling)} labeling(s).")
    return labeling


def fetch_clustbench_data(
        dataset_group: str = "wut",
        dataset_name: str = "smile",
        base_url: str = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"
) -> pd.DataFrame | None:
    """Fetches one Gagolewski benchmark dataset (features + all usable
    labeling) into the standard fetcher frame; None on failure."""
    if dataset_group not in CLUSTBENCH_METADATA:
        log.warning(f"Battery '{dataset_group}' is not in the recommended list, but attempting to fetch anyway.")

    log.info(f"Fetching external clustering dataset: {dataset_group}/{dataset_name}...")
    target_path = f"{base_url}/{dataset_group}/{dataset_name}"

    try:
        data = _loadtxt_url(f"{target_path}.data.gz", ndmin=2)
    except Exception as e:
        log.error(f"Failed to fetch dataset features: {e}")
        return None

    labeling = fetch_clustbench_labelings(dataset_group, dataset_name, base_url)
    labeling = {n: labels for n, labels in labeling.items() if len(labels) == len(data)}
    if "labels0" not in labeling:
        log.error(f"No usable labels0 for {dataset_group}/{dataset_name}.")
        return None

    feature_cols = [f"Feature_{i + 1}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_cols)
    for name, labels in labeling.items():
        df[f"GroundTruth_{name}"] = labels
    df["Cohort_Class"] = labeling["labels0"]

    log.info(f"Loaded '{dataset_group}/{dataset_name}': {df.shape[0]} rows, "
             f"{len(feature_cols)} features, {len(labeling)} labeling(s).")
    return df


SOURCE: Final[Source] = Source(
    key="clustbench",
    display="Clustbench",
    blurb="real benchmark datasets downloaded online (Gagolewski suite)",
    metadata=CLUSTBENCH_METADATA,
    datasets=CLUSTBENCH_DATASETS,
    heavy=CLUSTBENCH_HEAVY,
    seeded=False,
    suite_keys=("base_url",),
    fetch=lambda battery, dataset, seed, **kw: fetch_clustbench_data(
        dataset_group=battery, dataset_name=dataset, **kw),
    licence=("clustering-data-v1 v1.1.0: the repository is AGPL-3.0; its datasets are for "
             "research purposes only unless their description says otherwise. Cite "
             "Gagolewski (2022), SoftwareX 20, 101270, doi:10.1016/j.softx.2022.101270, the "
             "suite version, and the references in each dataset's description file."),
)
