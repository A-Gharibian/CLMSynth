# dataset_sources.py
"""
Consolidated cluster/dataset generation module. Four sources, one shared
registry shape:
    "clustbench"      -> fetch_clustbench_data  (real, fixed geometries, network)
    "mdcgen"          -> fetch_mdcgen_data      (synthetic, needs mdcgenpy)
    "fabricated_data" -> fetch_fabricated_data  (fabricated features with
                                                 perfect-separation labels, offline)
    "byoc"            -> fetch_byoc_data        (bring-your-own-clusters: a user CSV
                                                 with one cluster-id column; lives in
                                                 byoc_source.py)

The registry shape (SOURCE_METADATA / SOURCE_DATASETS / HEAVY_BATTERIES, keyed by
source name) is what makes a source pluggable; more are expected, so add entries to
those dicts rather than special-casing a new fetcher downstream.
"""

import gzip
import io
import logging
import urllib.request
from typing import Optional, List, Dict, Any, Set

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Timeout (seconds) for each clustbench network fetch, so a hung GitHub endpoint
# can't stall the pipeline forever (no circuit breaker otherwise).
CLUSTBENCH_TIMEOUT = 30


# ===========================================================================
# Shared registry, one dict per concern, keyed by source name.
# ===========================================================================

SOURCE_METADATA: Dict[str, Dict[str, Dict[str, str]]] = {
    "clustbench": {
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
    },
    "mdcgen": {
        "basic": {
            "description": "Standard low-dimensional clusters for basic algorithmic testing.",
            "examples": "blobs_2d_5c, blobs_3d_10c"
        },
        "high_dim": {
            "description": "High-dimensional synthetic datasets to test the curse of dimensionality.",
            "examples": "hd_50d_5c, hd_100d_10c"
        },
        "challenging": {
            "description": "Clusters with heavy overlap, variable compactness, or simulated noise/outliers.",
            "examples": "overlapping_2d, noisy_3d"
        },
    },
    # "fabricated_data" is populated once, below, right next to its fetcher and configs.
}

CLUSTBENCH_DATASETS: Dict[str, List[str]] = {
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

MDCGEN_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "basic": {
        "blobs_2d_5c": {"m": 1000, "n": 2, "k": 5},
        "blobs_3d_10c": {"m": 2000, "n": 3, "k": 10},
    },
    "high_dim": {
        "hd_50d_5c": {"m": 1000, "n": 50, "k": 5},
        "hd_100d_10c": {"m": 2000, "n": 100, "k": 10},
    },
    "challenging": {
        "overlapping_2d": {"m": 1000, "n": 2, "k": 4, "cp": 0.1},
        "noisy_3d": {"m": 1500, "n": 3, "k": 5, "out": 0.05},
    },
}

# --- "fabricated_data" source: wraps fabricated_generator.generate_synthetic_data.
# Defined once, here, alongside its own fetcher. Only the label side varies
# between presets, features are always the same 6 engineered columns.
SOURCE_METADATA["fabricated_data"] = {
    "fabricated": {
        "description": "Offline fallback using engineered features + perfect-separation "
                        "labels (fabricated_generator.py). No network, no mdcgenpy.",
        "examples": "baseline_2class, baseline_4class"
    },
}

FABRICATED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_2class": {
        "n_samples": 500,
        "config_dict": {
            "feature_generators": {
                "categorical_features": {"enable": True, "labels": ["Class_0", "Class_1"], "use_faker": False}
            },
            "ground_truths": {
                "class_clustering": {"enforce_perfect_separation": True, "target_label": "Cohort_Class"}
            },
        },
    },
    "baseline_4class": {
        "n_samples": 800,
        "config_dict": {
            "feature_generators": {
                "categorical_features": {"enable": True,
                                          "labels": ["Class_0", "Class_1", "Class_2", "Class_3"],
                                          "use_faker": False}
            },
            "ground_truths": {
                "class_clustering": {"enforce_perfect_separation": True, "target_label": "Cohort_Class"}
            },
        },
    },
}

SOURCE_DATASETS: Dict[str, Dict[str, List[str]]] = {
    "clustbench": CLUSTBENCH_DATASETS,
    "mdcgen": {group: list(cfgs.keys()) for group, cfgs in MDCGEN_CONFIGS.items()},
    "fabricated_data": {"fabricated": list(FABRICATED_CONFIGS.keys())},
    # byoc datasets are user CSV paths, unknown ahead of time: one 'local' battery
    # with a dynamic (empty) list, get_datasets_for_battery trusts the config.
    "byoc": {"local": []},
}

SOURCE_METADATA["byoc"] = {
    "local": {
        "description": "Bring-your-own-clusters: your CSV with feature columns and exactly "
                        "one cluster-id column (named by byoc_suite.cluster_column).",
        "examples": "the CSV path(s) listed in byoc_suite.datasets"
    },
}

HEAVY_BATTERIES: Dict[str, Set[str]] = {
    "clustbench": {"mnist", "g2mg", "h2mg"},
    "mdcgen": {"high_dim"},
    "fabricated_data": set(),
    "byoc": set(),
}


# ===========================================================================
# Shared selection helpers, source-qualified, so no name collides.
# ===========================================================================

def get_available_batteries(source: str) -> List[str]:
    """Battery (dataset group) names registered for a source."""
    return list(SOURCE_METADATA.get(source, {}).keys())


def resolve_selection(source: str, batteries_cfg) -> List[str]:
    """Resolves the config's batteries value ("all" or a list) to real names."""
    all_batteries = SOURCE_DATASETS.get(source, {})
    if batteries_cfg == "all":
        return list(all_batteries.keys())
    return [b for b in batteries_cfg if b in all_batteries]


def get_datasets_for_battery(source: str, battery: str, datasets_cfg) -> List[str]:
    """Resolves the config's datasets value within one battery; byoc (dynamic,
    empty registry) trusts the config's explicit list."""
    all_names = SOURCE_DATASETS.get(source, {}).get(battery, [])
    if not all_names:                       # dynamic source (byoc): trust the config's list
        return [] if datasets_cfg == "all" else list(datasets_cfg)
    if datasets_cfg == "all":
        return all_names
    return [d for d in datasets_cfg if d in all_names]


def is_heavy(source: str, battery: str) -> bool:
    """True when a battery is flagged as slow or large to fetch/generate."""
    return battery in HEAVY_BATTERIES.get(source, set())


def print_battery_info(source: str) -> None:
    """Logs the registered batteries of a source with their descriptions."""
    metadata = SOURCE_METADATA.get(source, {})
    log.info(f"Available '{source}' dataset batteries:")
    for battery, info in metadata.items():
        log.info(f"  - '{battery}': {info['description']} (Examples: {info['examples']})")


# ===========================================================================
# Source 1: clustbench (real, fixed geometries, network fetch)
# ===========================================================================

def _loadtxt_url(url: str, **loadtxt_kwargs) -> np.ndarray:
    """Fetches a gzipped text array over HTTP(S) with a timeout, then parses it.

    np.loadtxt can read a URL directly but offers no timeout hook, so a hung
    server would block the pipeline forever. We fetch the bytes with an explicit
    CLUSTBENCH_TIMEOUT and hand the decompressed stream to np.loadtxt, preserving
    its parsing kwargs (dtype, ndmin, ...)."""
    with urllib.request.urlopen(url, timeout=CLUSTBENCH_TIMEOUT) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        return np.loadtxt(stream, **loadtxt_kwargs)


def fetch_clustbench_labelings(
        dataset_group: str,
        dataset_name: str,
        base_url: str = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0",
        max_labelings: int = 10,
) -> Dict[str, np.ndarray]:
    """Downloads every reference labeling (labels0, labels1, ...) of one
    clustbench dataset, stopping at the first missing index."""
    target_path = f"{base_url}/{dataset_group}/{dataset_name}"
    labelings: Dict[str, np.ndarray] = {}
    for i in range(max_labelings):
        label_name = f"labels{i}"
        try:
            labelings[label_name] = _loadtxt_url(f"{target_path}.{label_name}.gz", dtype="int")
        except Exception:
            if i == 0:
                log.error(f"No reference labels found for {dataset_group}/{dataset_name}.")
            break
    return labelings


def fetch_clustbench_data(
        dataset_group: str = "wut",
        dataset_name: str = "smile",
        base_url: str = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"
) -> Optional[pd.DataFrame]:
    """Fetches one Gagolewski benchmark dataset (features + all usable
    labelings) into the standard fetcher frame; None on failure."""
    if dataset_group not in SOURCE_METADATA["clustbench"]:
        log.warning(f"Battery '{dataset_group}' is not in the recommended list, but attempting to fetch anyway.")

    log.info(f"Fetching external clustering dataset: {dataset_group}/{dataset_name}...")
    target_path = f"{base_url}/{dataset_group}/{dataset_name}"

    try:
        data = _loadtxt_url(f"{target_path}.data.gz", ndmin=2)
    except Exception as e:
        log.error(f"Failed to fetch dataset features: {e}")
        return None

    labelings = fetch_clustbench_labelings(dataset_group, dataset_name, base_url)
    labelings = {n: l for n, l in labelings.items() if len(l) == len(data)}
    if not labelings:
        log.error(f"No usable labelings for {dataset_group}/{dataset_name}.")
        return None

    feature_cols = [f"Feature_{i + 1}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_cols)
    for name, labels in labelings.items():
        df[f"GroundTruth_{name}"] = labels
    df["Cohort_Class"] = labelings["labels0"]

    log.info(f"Loaded '{dataset_group}/{dataset_name}': {df.shape[0]} rows, "
             f"{len(feature_cols)} features, {len(labelings)} labeling(s).")
    return df


# ===========================================================================
# Source 2: mdcgen (synthetic, needs mdcgenpy installed)
# ===========================================================================

def fetch_mdcgen_data(
        dataset_group: str = "basic",
        dataset_name: str = "blobs_2d_5c",
        seed: int = 1,
        **kwargs
) -> Optional[pd.DataFrame]:
    """Generates one synthetic dataset via mdcgenpy (optional dependency)
    from the registered preset, seeded; None if unavailable or failing."""
    try:
        import mdcgenpy
    except ImportError:
        log.error("mdcgenpy is not installed. Please install it via: "
                   "pip install git+https://github.com/CN-TU/mdcgenpy")
        return None

    if dataset_group not in SOURCE_METADATA["mdcgen"]:
        log.warning(f"Battery '{dataset_group}' is not in the recommended list, but attempting to generate anyway.")

    log.info(f"Generating synthetic clustering dataset: {dataset_group}/{dataset_name}...")

    config = MDCGEN_CONFIGS.get(dataset_group, {}).get(dataset_name, {})
    if not config:
        log.error(f"No configuration mapping found for {dataset_group}/{dataset_name}.")
        return None

    config = config.copy()
    if kwargs:
        config.update(kwargs)
    config["seed"] = seed  # confirmed kwarg name: ClusterGenerator.__init__(self, seed=1, ...)

    try:
        cluster_gen = mdcgenpy.clusters.ClusterGenerator(**config)
        data, labels = cluster_gen.generate_data()
    except Exception as e:
        log.error(f"Failed to generate dataset using mdcgenpy: {e}")
        return None

    feature_cols = [f"Feature_{i + 1}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_cols)
    df["GroundTruth_labels0"] = labels
    df["Cohort_Class"] = labels

    log.info(f"Generated '{dataset_group}/{dataset_name}' (seed={seed}): {df.shape[0]} rows, "
             f"{len(feature_cols)} features, 1 labeling(s).")
    return df


# ===========================================================================
# Source 3: fabricated_data, offline fallback, wraps fabricated_generator.py
# ===========================================================================

def fetch_fabricated_data(
        dataset_group: str = "fabricated",
        dataset_name: str = "baseline_2class",
        seed: int = 42,
        **kwargs
) -> Optional[pd.DataFrame]:
    """Generates one offline dataset via fabricated_generator; cluster ids
    are strings ("Class_0", ...), unlike the integer ids of other sources."""
    from . import fabricated_generator  # deferred, mirrors mdcgenpy's lazy import

    preset = FABRICATED_CONFIGS.get(dataset_name)
    if preset is None:
        log.error(f"No fabricated_data preset '{dataset_name}'. Available: {list(FABRICATED_CONFIGS.keys())}")
        return None

    n_samples = int(kwargs.get("n_samples", preset["n_samples"]))
    log.info(f"Generating offline fabricated_data dataset '{dataset_name}' (n={n_samples}, seed={seed})...")

    df = fabricated_generator.generate_synthetic_data(
        n_samples=n_samples,
        output_file=None,
        seed=seed,
        config_dict=preset["config_dict"],
    )

    if "Cohort_Class" not in df.columns:
        log.error(f"fabricated_data preset '{dataset_name}' produced no ground truth (categorical_features.enable "
                  "must be True). Skipping, label_generation has nothing to key off.")
        return None

    df["GroundTruth_labels0"] = df["Cohort_Class"]

    log.info(f"Generated fabricated_data '{dataset_name}': {df.shape[0]} rows, "
             f"{df.shape[1] - 2} feature(s), 1 labeling(s) (values are strings, e.g. 'Class_0').")
    return df