# SPDX-License-Identifier: MIT
# label_generator.py
"""Dispatch between the pipeline and the CLM engine: generates
`n_labels` synthetic label columns for one DatasetContext."""

from typing import Any

import numpy as np
import pandas as pd

from .clm_errors import clm_missing
from .clm_label_engine import generate_clm_labels
from .label_context import DatasetContext

__all__ = ["generate_additional_labels"]


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

    Examples
    --------
    >>> import pandas as pd
    >>> from clmsynth import build_context, generate_additional_labels
    >>> df = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0],
    ...                    "f2": [0.0, 1.0, 2.0, 3.0],
    ...                    "GroundTruth_labels0": [0, 0, 1, 1]})
    >>> ctx = build_context("byoc", "local", "demo", df)
    >>> cfg = {"num_classes": 2, "matching_mode": "perfect"}
    >>> generate_additional_labels(ctx, n_labels=1, clm_config=cfg, seed=0)
    >>> list(ctx.generated_labels)
    ['Label_0']
    """
    if source_labeling not in context.ground_truths:
        raise KeyError(f"'{source_labeling}' not found for {context.battery}/{context.dataset}; "
                       f"available: {list(context.ground_truths)}")

    cluster_labels = context.ground_truths[source_labeling].to_numpy()
    coords = context.features.to_numpy()
    clm_config = _require_clm_config(clm_config)

    for i in range(n_labels):
        report: dict[str, Any] = {}
        series = generate_clm_labels(cluster_labels, coords, clm_config, seed=seed + i,
                                     report=report)
        context.add_generated_label(context.next_generated_label_name(), series, report=report)


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
