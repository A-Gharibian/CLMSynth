# SPDX-License-Identifier: MIT
# labeling.py
"""One dataset and its generated labels: `DatasetContext` holds the dataset in
flight, `LabelGenerator` the run's label_generation block, which calls the CLM
engine once per label."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from .byoc_source import TAG_PREFIX
from .clm_errors import clm_missing
from .clm_label_engine import _check_config, generate_clm_labels
from .config_tree import default
from .metrics import clustering_ari, clustering_mcc

__all__ = ["DatasetContext", "LabelGenerator", "build_context", "generate_additional_labels"]

log: logging.Logger = logging.getLogger(__name__)

@dataclass
class DatasetContext:
    """One dataset in flight: features, ground truths, tags and generated labels.

    Every column added is length-checked against ``features``, so a misaligned
    series is refused at attach time rather than silently reindexed.

    Attributes
    ----------
    battery : str
        Group the dataset belongs to within its source.
    dataset : str
        Dataset name, used for the output file stem and in log messages.
    features : pandas.DataFrame
        Numeric feature columns. Defines ``n_rows`` and the geometry the engine
        derives centroids from.
    source : str, default "unknown"
        The ``data_source`` value of the source that produced this.
    ground_truths : dict of str to pandas.Series
        Reference clusterings, keyed by their source name (``labels0``, ...).
        Surfaced in the output frame as ``Cluster_0``, ``Cluster_1``, ...
    generated_labels : dict of str to pandas.Series
        Labels produced by the engine, keyed ``Label_0``, ``Label_1``, ...
    label_reports : dict of str to dict
        What the engine reported for each label (the ``report`` of
        ``generate_clm_labels``), keyed like ``generated_labels``. A label whose
        report is empty has no entry.
    tags : pandas.DataFrame
        Passenger columns carried to the output untouched and kept out of the
        feature geometry. Need not be numeric.
    """

    battery: str
    dataset: str
    features: pd.DataFrame
    source: str = "unknown"  # the data_source value of the source that produced this
    ground_truths: dict[str, pd.Series] = field(default_factory=dict)
    generated_labels: dict[str, pd.Series] = field(default_factory=dict)
    label_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Passenger columns: carried to the CSV, kept out of the geometry.
    tags: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.features = self.features.reset_index(drop=True)
        self.n_rows: int = len(self.features)
        self.ground_truths = {name: self._align(s, name) for name, s in self.ground_truths.items()}
        self.tags = self.tags.reset_index(drop=True)
        if len(self.tags.columns) and len(self.tags) != self.n_rows:
            raise ValueError(
                f"[{self._tag}] tag columns have {len(self.tags)} rows, "
                f"features have {self.n_rows}. Refusing to attach a misaligned column."
            )

    @property
    def _tag(self) -> str:
        return f"{self.source}/{self.battery}/{self.dataset}"

    def _align(self, series: pd.Series, name: str) -> pd.Series:
        series = pd.Series(series).reset_index(drop=True)
        if len(series) != self.n_rows:
            raise ValueError(
                f"[{self._tag}] '{name}' has {len(series)} rows, "
                f"features have {self.n_rows}. Refusing to attach a misaligned column."
            )
        return series

    def add_generated_label(self, name: str, series: pd.Series,
                            report: dict[str, Any] | None = None) -> None:
        """Attaches a generated label column, rejecting misaligned lengths, and
        the engine's report for it when there is one."""
        self.generated_labels[name] = self._align(series, name)
        if report:
            self.label_reports[name] = dict(report)
        log.info(f"[{self._tag}] Attached generated label '{name}'.")

    def next_generated_label_name(self) -> str:
        """Next output column name: Label_0, Label_1, ..."""
        return f"Label_{len(self.generated_labels)}"

    def gt_column_name(self, name: str) -> str:
        """Output/display name for a ground-truth labeling, by position: the
        first labeling -> 'Cluster_0', the second -> 'Cluster_1', ... (so the
        config's source_labeling='labels0' surfaces in the CSV as 'Cluster_0')."""
        return f"Cluster_{list(self.ground_truths).index(name)}"

    def gt_column(self, source_labeling: str) -> str | None:
        """The output name of `source_labeling`, or None when the dataset has no
        such labeling."""
        if source_labeling not in self.ground_truths:
            return None
        return self.gt_column_name(source_labeling)

    def scores(self, source_labeling: str) -> list[dict[str, Any]]:
        """Per generated label: MCC and ARI against `source_labeling` (None
        without it), the label counts, and the engine's report."""
        truth = self.ground_truths.get(source_labeling)
        results: list[dict[str, Any]] = []
        for name, labels in self.generated_labels.items():
            mcc = ari = None
            if truth is not None:
                mcc = float(clustering_mcc(truth, labels))
                ari = float(clustering_ari(truth, labels))
            counts = labels.value_counts().sort_index().to_dict()
            results.append({"name": name, "mcc": mcc, "ari": ari, "counts": counts,
                            "report": self.label_reports.get(name)})
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Assembles the output frame: features, tags, Cluster_n, Label_n."""
        # Annotated because the first element is a DataFrame and the rest are
        # Series; without it the list type is inferred from `features` alone.
        parts: list[pd.DataFrame | pd.Series] = [self.features]
        if len(self.tags.columns):
            parts.append(self.tags)
        parts += [s.rename(self.gt_column_name(n)) for n, s in self.ground_truths.items()]
        parts += [s.rename(n) for n, s in self.generated_labels.items()]
        return pd.concat(parts, axis=1)


def build_context(source: str, battery: str, dataset: str, df: pd.DataFrame) -> DatasetContext:
    """Splits a fetched DataFrame from any of the fetchers into a DatasetContext.

    Column roles are carried in the frame by prefix, so every fetcher can return
    a single flat DataFrame and have it unpacked identically.

    Parameters
    ----------
    source : str
        The ``data_source`` value of the source that produced ``df``.
    battery : str
        Group the dataset belongs to within that source.
    dataset : str
        Dataset name.
    df : pandas.DataFrame
        Flat frame. ``GroundTruth_*`` columns become ``ground_truths`` with the
        prefix stripped, ``TagColumn_*`` become ``tags`` likewise, ``Cohort_Class``
        is dropped, and every remaining column is treated as a feature.

    Returns
    -------
    DatasetContext
        With ``generated_labels`` still empty.
    """
    gt_cols = [c for c in df.columns if c.startswith("GroundTruth_")]
    tag_cols = [c for c in df.columns if c.startswith(TAG_PREFIX)]
    feature_cols = [c for c in df.columns
                    if c not in gt_cols and c not in tag_cols and c != "Cohort_Class"]
    ground_truths = {c.replace("GroundTruth_", ""): df[c] for c in gt_cols}
    # The transport prefix comes off here, so the CSV shows the user's own name.
    tags = df[tag_cols].rename(columns=lambda c: c[len(TAG_PREFIX):])
    return DatasetContext(battery, dataset, df[feature_cols], source=source,
                          ground_truths=ground_truths, tags=tags)


class LabelGenerator:
    """The run's label_generation block: how many labels, matched against which
    ground-truth labeling, from which seed, under which clm_label block. Built
    once per run and applied to each dataset. Values are kept as given; the
    checks run in `check` and `apply`."""

    def __init__(self, clm_config: dict[str, Any] | None, n_labels: int = 1,
                 source_labeling: str = "labels0", seed: int = 42) -> None:
        self.clm_config = clm_config
        self.n_labels = n_labels
        self.source_labeling = source_labeling
        self.seed = seed

    @classmethod
    def from_block(cls, block: Mapping[str, Any], seed: int) -> Self:
        """From a config's label_generation block, with the run's default for
        each setting it leaves out. `seed` comes resolved: an absent or valueless
        seed is decided by the run."""
        return cls(block.get("clm_label"),
                   n_labels=block.get("n_labels", default("label_generation.n_labels")),
                   source_labeling=block.get("source_labeling",
                                             default("label_generation.source_labeling")),
                   seed=seed)

    def check(self) -> None:
        """The checks on clm_label that need no dataset: [CLM-207] when it is
        missing or empty, then the engine's, without their warnings."""
        clm_config = _require_clm_config(self.clm_config)
        _check_config(clm_config, warn=False)

    def seed_for(self, i: int) -> int:
        """Label ``i``'s seed: the columns differ from one another while the run
        as a whole stays reproducible."""
        return self.seed + i

    def apply(self, context: DatasetContext) -> None:
        """Attaches ``n_labels`` generated label columns to ``context``, one
        engine call each, as ``Label_0``, ``Label_1``, ..."""
        if self.source_labeling not in context.ground_truths:
            raise KeyError(f"'{self.source_labeling}' not found for "
                           f"{context.battery}/{context.dataset}; "
                           f"available: {list(context.ground_truths)}")

        cluster_labels = context.ground_truths[self.source_labeling].to_numpy()
        coords = context.features.to_numpy()
        clm_config = _require_clm_config(self.clm_config)

        for i in range(self.n_labels):
            report: dict[str, Any] = {}
            series = generate_clm_labels(cluster_labels, coords, clm_config,
                                         seed=self.seed_for(i), report=report)
            context.add_generated_label(context.next_generated_label_name(), series,
                                        report=report)


def generate_additional_labels(
        context: DatasetContext,
        n_labels: int = 1,
        source_labeling: str = "labels0",
        clm_config: dict[str, Any] | None = None,
        noise: float = 0.1,
        seed: int = 42,
) -> None:
    """Attaches ``n_labels`` generated label columns to ``context``.

    Object-level entry point, reads one ground-truth labeling out
    of ``context``, calls the engine once per requested label, and attaches each
    result back as ``Label_0``, ``Label_1``, ... No file is read or written.
    The same as ``LabelGenerator(clm_config, n_labels, source_labeling,
    seed).apply(context)``.

    Parameters
    ----------
    context : DatasetContext
        Container holding the features and at least one ground-truth labeling.
        Mutated in place.
    n_labels : int, default 1
        How many label columns to generate.
    source_labeling : str, default "labels0"
        Key into ``context.ground_truths`` naming the clustering to match against.
    clm_config : dict, optional
        The ``clm_label`` block: ``num_classes``, ``matching_mode`` and whatever
        that mode requires. Required despite the default; ``None`` or an empty
        block raises ``[CLM-207]``.
    noise : float, default 0.1
        Accepted for call-signature compatibility and unused; the noise fallback
        was retired in favor of the engine.
    seed : int, default 42
        Label ``i`` uses ``seed + i``, so the columns differ from one another
        while the run as a whole stays reproducible.

    Returns
    -------
    None
        ``context`` is modified in place; read the result from
        ``context.generated_labels`` or ``context.to_dataframe()``, and the
        engine's report from ``context.label_reports``.

    Raises
    ------
    KeyError
        If ``source_labeling`` is not in ``context.ground_truths``.
    MissingConfigKey
        A ``KeyError`` subclass, carrying ``[CLM-207]`` when ``clm_config`` is
        missing or empty.
    InfeasibleAllocationError
        If the configuration is valid but the requested counts cannot fit the
        cluster capacities, or ``target_metric.strict`` refuses a missed target.
    """
    LabelGenerator(clm_config, n_labels=n_labels, source_labeling=source_labeling,
                   seed=seed).apply(context)


def _require_clm_config(clm_config: dict[str, Any] | None) -> dict[str, Any]:
    """clm_config, or [CLM-207] when it is missing or empty."""
    # A missing key, as for any other required key of the block.
    if not clm_config:
        raise clm_missing(207, where="label_generation", key="clm_label")
    return clm_config


def _legacy_noise_flip(base, noise, seed):
    """Superseded by generate_clm_labels. M.G. also has a similar idea in clustbench.
    Excludes the point's own class from the reassignment pool, without this,
    `noise` doesn't equal the true corruption rate."""
    rng = np.random.default_rng(seed)
    classes = np.unique(base)
    noisy = base.copy()
    for idx in np.where(rng.random(len(base)) < noise)[0]:
        noisy[idx] = rng.choice(classes[classes != base[idx]])
    return pd.Series(noisy)
