# label_generator.py
"""Dispatch layer between the pipeline and the CLM engine: generates
`n_labels` synthetic label columns for one DatasetContext."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .clm_label_engine import generate_clm_labels
from .label_context import DatasetContext


def generate_additional_labels(
        context: DatasetContext,
        n_labels: int = 1,
        source_labeling: str = "labels0",
        clm_config: Optional[Dict[str, Any]] = None,
        noise: float = 0.1,
        seed: int = 42,
) -> None:
    """Attaches `n_labels` generated label columns to `context`.

    Each label i uses seed + i, so labels differ but the run is reproducible.
    With a `clm_config` the CLM engine is used; without one the legacy
    noise-flip fallback corrupts the ground truth at rate `noise`.
    """
    if source_labeling not in context.ground_truths:
        raise KeyError(f"'{source_labeling}' not found for {context.battery}/{context.dataset}; "
                       f"available: {list(context.ground_truths)}")

    cluster_labels = context.ground_truths[source_labeling].to_numpy()
    coords = context.features.to_numpy()

    for i in range(n_labels):
        if clm_config:
            series = generate_clm_labels(cluster_labels, coords, clm_config, seed=seed + i)
        else:
            series = _legacy_noise_flip(cluster_labels, noise=noise, seed=seed + i)
        context.add_generated_label(context.next_generated_label_name(), series)


def _legacy_noise_flip(base, noise, seed):
    """Superseded by generate_clm_labels; kept only for configs without clm_label.
    Excludes the point's own class from the reassignment pool, without this,
    `noise` doesn't equal the true corruption rate."""
    rng = np.random.default_rng(seed)
    classes = np.unique(base)
    noisy = base.copy()
    for idx in np.where(rng.random(len(base)) < noise)[0]:
        noisy[idx] = rng.choice(classes[classes != base[idx]])
    return pd.Series(noisy)