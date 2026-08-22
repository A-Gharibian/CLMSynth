# label_generator.py
"""Dispatch layer between the pipeline and the CLM engine: generates
`n_labels` synthetic label columns for one DatasetContext."""

from typing import Any

import numpy as np
import pandas as pd

from .clm_label_engine import generate_clm_labels
from .label_context import DatasetContext


def generate_additional_labels(
        context: DatasetContext,
        n_labels: int = 1,
        source_labeling: str = "labels0",
        clm_config: dict[str, Any] | None = None,
        noise: float = 0.1,
        seed: int = 42,
) -> None:
    """Attaches `n_labels` generated label columns to `context`.

    Each label "i" uses "seed + i", so labels differ but the run is reproducible.
    Requires `clm_config`; the noise fallback is retired.
    """
    if source_labeling not in context.ground_truths:
        raise KeyError(f"'{source_labeling}' not found for {context.battery}/{context.dataset}; "
                       f"available: {list(context.ground_truths)}")

    cluster_labels = context.ground_truths[source_labeling].to_numpy()
    coords = context.features.to_numpy()

    if not clm_config:
        raise KeyError(f"'clm_label' is required for {context.battery}/{context.dataset}; "
                       "the noise fallback is retired.")

    for i in range(n_labels):
        series = generate_clm_labels(cluster_labels, coords, clm_config, seed=seed + i)
        context.add_generated_label(context.next_generated_label_name(), series)


def _legacy_noise_flip(base, noise, seed):
    """Superseded by generate_clm_labels; no longer called.
    Excludes the point's own class from the reassignment pool, without this,
    `noise` doesn't equal the true corruption rate."""
    rng = np.random.default_rng(seed)
    classes = np.unique(base)
    noisy = base.copy()
    for idx in np.where(rng.random(len(base)) < noise)[0]:
        noisy[idx] = rng.choice(classes[classes != base[idx]])
    return pd.Series(noisy)
