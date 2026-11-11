# SPDX-License-Identifier: MIT
# dataset_sources.py
"""
Consolidated cluster/dataset generation module. Four sources:
    "clustbench"      real, fixed geometries, network; _input_sources/clustbench_source.py
    "mdcgen"          synthetic, needs mdcgenpy; _input_sources/mdcgen_source.py
    "fabricated_data" fabricated features with perfect-separation labels, offline; here
    "byoc"            bring-your-own-clusters: a user CSV with one cluster-id column;
                      byoc_source.py

Each source declares one ``Source``; ``SOURCES`` collects them, the tables below
are built from it, and ``fetch`` fetches one dataset of any of them.
"""

import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from numbers import Integral
from typing import Any, Final, Literal

import pandas as pd

from . import byoc_source
from ._input_sources import Source, clustbench_source, mdcgen_source
from ._input_sources.clustbench_source import (
    CLUSTBENCH_DATASETS as CLUSTBENCH_DATASETS,
)
from ._input_sources.mdcgen_source import (
    MDCGEN_CONFIGS as MDCGEN_CONFIGS,
)
from ._utils.write_files import resolved_for_report as resolved_for_report

log: logging.Logger = logging.getLogger(__name__)

# ===========================================================================
# fabricated_data, offline fallback, wraps _input_sources/fabricated_generator.py
# ===========================================================================

FABRICATED_CONFIGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "baseline_2class": {
        "n_samples": 500,
        "config_dict": {
            "feature_generators": {
                "categorical_features": {"enable": True, "labels": ["Class_0", "Class_1"]}
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
                                          "labels": ["Class_0", "Class_1", "Class_2", "Class_3"]}
            },
            "ground_truths": {
                "class_clustering": {"enforce_perfect_separation": True, "target_label": "Cohort_Class"}
            },
        },
    },
    # Cluster ids and no feature space at all. Spatial placement is the one thing
    # that needs geometry, so combining this preset with centroid_dependence raises
    # [CLM-125] in every mode but random.
    "labels_only_4class": {
        "n_samples": 800,
        "config_dict": {
            "feature_generators": {
                "categorical_features": {"enable": True,
                                          "labels": ["Class_0", "Class_1", "Class_2", "Class_3"],
                                          "labels_only": True}
            },
            "ground_truths": {"class_clustering": {"target_label": "Cohort_Class"}},
        },
    },
}


def _fetch_fabricated_data(
        dataset_group: str = "fabricated",
        dataset_name: str = "baseline_2class",
        seed: int = 42,
        **kwargs: Any
) -> pd.DataFrame | None:
    """Generates one offline dataset via fabricated_generator; cluster ids are
    integers 0...K-1, matching clustbench/mdcgen, so one clm_label config ports
    across sources without retyping cluster references."""
    from ._input_sources import fabricated_generator

    preset = FABRICATED_CONFIGS.get(dataset_name)
    if preset is None:
        log.error(f"No fabricated_data preset '{dataset_name}'. Available: {list(FABRICATED_CONFIGS.keys())}")
        return None

    # Written with no value, as absent: the preset's size.
    n_samples = kwargs.get("n_samples")
    if n_samples is None:
        n_samples = preset["n_samples"]
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        log.error(f"fabricated_data_suite.n_samples must be a whole number of points, 1 or "
                  f"more, got {n_samples!r}.")
        return None
    n_samples = int(n_samples)
    log.info(f"Generating offline fabricated_data dataset '{dataset_name}' (n={n_samples}, seed={seed})...")

    try:
        df = fabricated_generator.generate_synthetic_data(
            n_samples=n_samples,
            output_file=None,
            seed=seed,
            config_dict=preset["config_dict"],
        )
    except ValueError as e:
        log.error(f"fabricated_data_suite.n_samples = {n_samples} is too few points for the "
                  f"preset '{dataset_name}': {e}")
        return None

    if "Cohort_Class" not in df.columns:
        log.error(f"fabricated_data preset '{dataset_name}' produced no ground truth (categorical_features.enable "
                  "must be True). Skipping, label_generation has nothing to key off.")
        return None

    gt = df["Cohort_Class"]
    codes = (gt.cat.codes if isinstance(gt.dtype, pd.CategoricalDtype)
             # qcut's category order is the percentile order, so cat.codes is
             # meaningful (0 = lowest bin); factorize covers the non-categorical
             # path (the random, non-perfect-separation branch of the generator).
             else pd.Series(pd.factorize(gt, sort=True)[0], index=gt.index))
    df["GroundTruth_labels0"] = codes.astype(int)

    log.info(f"Generated fabricated_data '{dataset_name}': {df.shape[0]} rows, "
             f"{df.shape[1] - 2} feature(s), 1 labeling(s) "
             f"(cluster ids 0..{df['GroundTruth_labels0'].nunique() - 1}).")
    return df


_FABRICATED: Final[Source] = Source(
    key="fabricated_data",
    display="Fabricated",
    blurb="tiny offline example data (no internet, no extra packages)",
    metadata={
        "fabricated": {
            "description": "Offline fallback using engineered features + perfect-separation "
                            "labels. No network, no mdcgenpy.",
            "examples": "baseline_2class, baseline_4class"
        },
    },
    datasets={"fabricated": list(FABRICATED_CONFIGS.keys())},
    heavy=frozenset(),
    seeded=True,
    suite_keys=("n_samples",),
    fetch=lambda battery, dataset, seed, **kw: _fetch_fabricated_data(
        dataset_group=battery, dataset_name=dataset, seed=seed, **kw),
    licence="MIT: generated by CLMSynth's own _input_sources/fabricated_generator.py.",
)

# ===========================================================================
# The registry, and the per-concern tables built from it, keyed by source name.
# ===========================================================================

SOURCES: Final[Mapping[str, Source]] = {
    source.key: source
    for source in (clustbench_source.SOURCE, mdcgen_source.SOURCE, _FABRICATED,
                   byoc_source.SOURCE)
}

SOURCE_METADATA: Final[Mapping[str, Mapping[str, Mapping[str, str]]]] = {
    key: source.metadata for key, source in SOURCES.items()}

SOURCE_DATASETS: Final[Mapping[str, Mapping[str, Sequence[str]]]] = {
    key: source.datasets for key, source in SOURCES.items()}

HEAVY_BATTERIES: Final[Mapping[str, AbstractSet[str]]] = {
    key: source.heavy for key, source in SOURCES.items()}


# ===========================================================================
# Shared selection helpers
# ===========================================================================

def get_available_batteries(source: str) -> list[str]:
    """Battery (dataset group) names registered for a source."""
    return list(SOURCE_METADATA.get(source, {}).keys())


def resolve_selection(source: str, batteries_cfg: Literal["all"] | Iterable[str]) -> list[str]:
    """Resolves the config's batteries value ("all" or a list) to real names."""
    all_batteries = SOURCE_DATASETS.get(source, {})
    if batteries_cfg == "all":
        return list(all_batteries.keys())
    return [b for b in batteries_cfg if b in all_batteries]


def get_datasets_for_battery(source: str, battery: str,
                             datasets_cfg: Literal["all"] | Iterable[str]) -> list[str]:
    """Resolves the config's datasets value within one battery; byoc (dynamic,
    empty registry) trusts the config's explicit list."""
    all_names = SOURCE_DATASETS.get(source, {}).get(battery, [])
    if not all_names:                       # dynamic source (byoc): trust the config's list
        return [] if datasets_cfg == "all" else list(datasets_cfg)
    if datasets_cfg == "all":
        return list(all_names)
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


def fetch(source: str, battery: str, dataset: str, *, seed: int = 42,
          **kwargs: Any) -> pd.DataFrame | None:
    """One dataset of a registered source, in the frame the pipeline labels:
    feature columns, ``GroundTruth_labels0``... and ``Cohort_Class``; None when
    it cannot be fetched or generated, with the reason logged.

    ``seed`` reaches only the sources whose data depend on it (mdcgen,
    fabricated_data). ``kwargs`` are the source's own ``<source>_suite`` keys,
    such as ``base_url`` or ``input_dir``; a key the source does not take is
    logged and gives None."""
    if source not in SOURCES:
        raise ValueError(f"Unknown data_source '{source}'. Valid options: {list(SOURCES)}")
    takes = SOURCES[source].suite_keys
    unknown = [] if takes is None else [repr(key) for key in kwargs if key not in takes]
    if unknown:
        log.error(f"'{source}' does not take {', '.join(unknown)}. It takes: "
                  f"{', '.join(takes or ()) or 'no key of its own'}.")
        return None
    return SOURCES[source].fetch(battery, dataset, seed, **kwargs)


# The per-source names this module exported before fetch().
_DEPRECATED: Final[Mapping[str, Callable[[], Any]]] = {
    "fetch_clustbench_data": lambda: clustbench_source.fetch_clustbench_data,
    "fetch_clustbench_labelings": lambda: clustbench_source.fetch_clustbench_labelings,
    "CLUSTBENCH_TIMEOUT": lambda: clustbench_source.CLUSTBENCH_TIMEOUT,
    "_loadtxt_url": lambda: clustbench_source._loadtxt_url,
    "fetch_mdcgen_data": lambda: mdcgen_source.fetch_mdcgen_data,
    "fetch_fabricated_data": lambda: _fetch_fabricated_data,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED:
        warnings.warn(f"clmsynth.dataset_sources.{name} is deprecated and is removed in "
                      "0.7.8; use dataset_sources.fetch(source, battery, dataset, seed=...)",
                      DeprecationWarning, stacklevel=2)
        return _DEPRECATED[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
