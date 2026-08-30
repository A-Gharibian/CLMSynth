# label_context.py

import logging
from dataclasses import dataclass, field

import pandas as pd

from .byoc_source import TAG_PREFIX

log = logging.getLogger(__name__)

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
        Which fetcher produced this: ``clustbench``, ``mdcgen``,
        ``fabricated_data`` or ``byoc``.
    ground_truths : dict of str to pandas.Series
        Reference clusterings, keyed by their source name (``labels0``, ...).
        Surfaced in the output frame as ``Cluster_0``, ``Cluster_1``, ...
    generated_labels : dict of str to pandas.Series
        Labels produced by the engine, keyed ``Label_0``, ``Label_1``, ...
    tags : pandas.DataFrame
        Passenger columns carried to the output untouched and kept out of the
        feature geometry. Need not be numeric.
    """

    battery: str
    dataset: str
    features: pd.DataFrame
    source: str = "unknown"  # clustbench | mdcgen | fabricated_data, which fetcher produced this
    ground_truths: dict[str, pd.Series] = field(default_factory=dict)
    generated_labels: dict[str, pd.Series] = field(default_factory=dict)
    # Passenger columns: carried to the CSV, kept out of the geometry.
    tags: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.features = self.features.reset_index(drop=True)
        self.n_rows = len(self.features)
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

    def add_generated_label(self, name: str, series: pd.Series) -> None:
        """Attaches a generated label column, rejecting misaligned lengths."""
        self.generated_labels[name] = self._align(series, name)
        log.info(f"[{self._tag}] Attached generated label '{name}'.")

    def next_generated_label_name(self) -> str:
        """Next output column name: Label_0, Label_1, ..."""
        return f"Label_{len(self.generated_labels)}"

    def gt_column_name(self, name: str) -> str:
        """Output/display name for a ground-truth labeling, by position: the
        first labeling -> 'Cluster_0', the second -> 'Cluster_1', ... (so the
        config's source_labeling='labels0' surfaces in the CSV as 'Cluster_0')."""
        return f"Cluster_{list(self.ground_truths).index(name)}"

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
    a single flat DataFrame and have it unpacked identically here.

    Parameters
    ----------
    source : str
        Fetcher that produced ``df``: ``clustbench``, ``mdcgen``,
        ``fabricated_data`` or ``byoc``.
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

    Examples
    --------
    >>> import pandas as pd
    >>> from clmsynth import build_context
    >>> df = pd.DataFrame({"f1": [0.0, 1.0], "GroundTruth_labels0": [0, 1]})
    >>> ctx = build_context("byoc", "local", "demo", df)
    >>> list(ctx.features.columns), list(ctx.ground_truths)
    (['f1'], ['labels0'])
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
