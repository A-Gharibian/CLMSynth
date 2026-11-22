# SPDX-License-Identifier: MIT
# clmsynth/_input_sources/__init__.py
"""Data-source modules behind ``dataset_sources``, which is their interface,
``Source``, the declaration every source makes, and the columns of the one flat
frame every source returns."""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from ..config_tree import Kind

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "COHORT_COLUMN",
    "ENV_VAR",
    "FEATURE_RECORD",
    "GROUND_TRUTH_PREFIX",
    "INCOMPLETE",
    "NOT_CACHED",
    "SUITE_SEED_MIN",
    "TAG_PREFIX",
    "Fetch",
    "Source",
    "column_roles",
    "engine_features",
]

# A fetched frame's column roles, by name: each GroundTruth_<name> is a ground-truth
# labeling, each TagColumn_<name> a passenger column, Cohort_Class the source's own
# class column, which is dropped; every other column is a feature.
GROUND_TRUTH_PREFIX: Final = "GroundTruth_"
TAG_PREFIX: Final = "TagColumn_"
COHORT_COLUMN: Final = "Cohort_Class"

# The dataset cache's folder, when set. Here rather than in dataset_cache.py, so the
# command's --help imports no numpy or pandas.
ENV_VAR: Final = "CLMSYNTH_CACHE_DIR"
# byoc reads the user's own file, and fabricated_data is generated in milliseconds.
NOT_CACHED: Final[frozenset[str]] = frozenset({"byoc", "fabricated_data"})

# The lowest `<source>_suite.seed` a seeded source takes.
SUITE_SEED_MIN = 0

# The key in a fetched frame's `attrs` under which a source records which features
# carry cluster structure, as lines for the dataset's summary txt.
FEATURE_RECORD = "feature_record"

# The key in a fetched frame's `attrs`, True when the source delivers less than the
# dataset has. The dataset cache does not keep such a frame, whatever the logging
# settings.
INCOMPLETE = "incomplete"


def column_roles(columns: Iterable[Hashable]) -> tuple[list[str], list[str], list[Hashable]]:
    """A fetched frame's ground-truth, tag and feature columns, each in frame order.
    A name that is not text, as pd.DataFrame(X) gives its columns, carries no
    prefix: it is a feature."""
    columns = list(columns)
    truths = [c for c in columns if isinstance(c, str) and c.startswith(GROUND_TRUTH_PREFIX)]
    tags = [c for c in columns if isinstance(c, str) and c.startswith(TAG_PREFIX)]
    features = [c for c in columns if c not in truths and c not in tags and c != COHORT_COLUMN]
    return truths, tags, features


def engine_features(frame: "pd.DataFrame") -> "pd.DataFrame":
    """The feature columns as a dataset's `DatasetContext` keeps them, whose
    to_numpy() the engine reads its coordinates from."""
    return frame[column_roles(frame.columns)[2]].reset_index(drop=True)


class Fetch(Protocol):
    """A source's fetch: (battery, dataset, seed, **suite keys) -> the dataset's
    frame, or None with the reason logged."""

    def __call__(self, battery: str, dataset: str, seed: int,
                 **suite_keys: Any) -> "pd.DataFrame | None": ...


@dataclass(frozen=True)
class Source:
    """What one data source declares about itself. ``dataset_sources.SOURCES``
    collects the declarations, and every per-source table is built from them."""

    key: str                                    # the data_source value
    display: str                                # run folder, plot titles, summaries
    blurb: str                                  # its line in the wizard's source question
    metadata: Mapping[str, Mapping[str, str]]   # battery -> description, examples
    datasets: Mapping[str, Sequence[str]]       # battery -> dataset names
    heavy: AbstractSet[str]                     # batteries slow or large to fetch
    seeded: bool                                # the suite seed reaches the data
    # Its own <source>_suite keys and what each takes; None: unchecked, passed on
    suite_keys: Mapping[str, Kind] | None
    fetch: Fetch
    needs: str | None = None                    # optional package it imports
    # (**suite keys) -> one extra log line about the input
    describe: Callable[..., str] | None = None
    # (battery, dataset, **suite keys) -> the dataset's cluster ids, read before any fetch
    # so the run refuses a [CLM-105] mismatch early; None from it: not read, judged later
    cluster_ids: Callable[..., list[int] | None] | None = None
    # (battery, dataset, **suite keys) -> whether a dataset is there, asked before any
    # fetch; for a source that reads what the config names. None: `datasets` lists all
    has_dataset: Callable[..., bool] | None = None
    # (the <source>_suite block) -> raises a coded error for a key the source cannot
    # pass on; run with the run's key checks, before any fetch. None: nothing more
    suite_check: Callable[[Mapping[Any, Any]], None] | None = None
