# SPDX-License-Identifier: MIT
# clmsynth/_input_sources/__init__.py
"""Data-source modules behind ``dataset_sources``, which is their interface, and
``Source``, the declaration every source makes."""

from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import pandas as pd

__all__ = ["FEATURE_RECORD", "Source"]

# The key in a fetched frame's `attrs` under which a source records which features
# carry cluster structure, as lines for the dataset's summary txt.
FEATURE_RECORD = "feature_record"


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
    suite_keys: tuple[str, ...] | None          # its own <source>_suite keys; None: unchecked
    fetch: Callable[..., pd.DataFrame | None]   # (battery, dataset, seed, **suite keys)
    licence: str | None                         # upstream licence and what to cite; None: no upstream
    needs: str | None = None                    # optional package it imports
    # (**suite keys) -> one extra log line about the input
    describe: Callable[..., str] | None = None
    # (battery, dataset, **suite keys) -> the dataset's cluster ids, read before any fetch
    # so the run refuses a [CLM-105] mismatch early; None from it: not read, judged later
    cluster_ids: Callable[..., list[int] | None] | None = None
    # (battery, dataset, **suite keys) -> whether a dataset is there, asked before any
    # fetch; for a source that reads what the config names. None: `datasets` lists all
    has_dataset: Callable[..., bool] | None = None
