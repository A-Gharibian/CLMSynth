# label_context.py

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

@dataclass
class DatasetContext:
    battery: str
    dataset: str
    features: pd.DataFrame
    source: str = "unknown"  # clustbench | mdcgen | fabricated_data, which fetcher produced this
    ground_truths: dict[str, pd.Series] = field(default_factory=dict)
    generated_labels: dict[str, pd.Series] = field(default_factory=dict)

    def __post_init__(self):
        self.features = self.features.reset_index(drop=True)
        self.n_rows = len(self.features)
        self.ground_truths = {name: self._align(s, name) for name, s in self.ground_truths.items()}

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
        """Assembles the output frame: features, Cluster_n columns, Label_n columns."""
        # Annotated because the first element is a DataFrame and the rest are
        # Series; without it the list type is inferred from `features` alone.
        parts: list[pd.DataFrame | pd.Series] = [self.features]
        parts += [s.rename(self.gt_column_name(n)) for n, s in self.ground_truths.items()]
        parts += [s.rename(n) for n, s in self.generated_labels.items()]
        return pd.concat(parts, axis=1)


def build_context(source: str, battery: str, dataset: str, df: pd.DataFrame) -> DatasetContext:
    """Splits a fetched DataFrame, from any of the three fetchers in
    test_cluster_generators.py, back into a DatasetContext."""
    gt_cols = [c for c in df.columns if c.startswith("GroundTruth_")]
    feature_cols = [c for c in df.columns if c not in gt_cols and c != "Cohort_Class"]
    ground_truths = {c.replace("GroundTruth_", ""): df[c] for c in gt_cols}
    return DatasetContext(battery, dataset, df[feature_cols], source=source, ground_truths=ground_truths)