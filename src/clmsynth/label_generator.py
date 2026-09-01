# label_generator.py
"""Dispatch between the pipeline and the CLM engine: generates
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
    """Attaches ``n_labels`` generated label columns to ``context``.

    Object-level entry point, reads one ground-truth labeling out
    of ``context``, calls the engine once per requested label, and attaches each
    result back as ``Label_0``, ``Label_1``, ... No file is read or written.

    Parameters
    ----------
    context : DatasetContext
        Container holding the features and at least one ground-truth labelling.
        Mutated in place.
    n_labels : int, default 1
        How many label columns to generate.
    source_labeling : str, default "labels0"
        Key into ``context.ground_truths`` naming the clustering to match against.
    clm_config : dict, optional
        The ``clm_label`` block: ``num_classes``, ``matching_mode`` and whatever
        that mode requires. Required despite the default; passing ``None`` raises.
    noise : float, default 0.1
        Accepted for call-signature compatibility and unused; the noise fallback
        was retired in favour of the engine.
    seed : int, default 42
        Label ``i`` uses ``seed + i``, so the columns differ from one another
        while the run as a whole stays reproducible.

    Returns
    -------
    None
        ``context`` is modified in place; read the result from
        ``context.generated_labels`` or ``context.to_dataframe()``.

    Raises
    ------
    KeyError
        If ``source_labeling`` is not in ``context.ground_truths``, or if
        ``clm_config`` is missing or empty.
    InfeasibleAllocationError
        If the configuration is valid but the requested counts cannot fit the
        cluster capacities.

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

    if not clm_config:
        raise KeyError(f"'clm_label' is required for {context.battery}/{context.dataset}; "
                       "the noise fallback is retired.")

    for i in range(n_labels):
        series = generate_clm_labels(cluster_labels, coords, clm_config, seed=seed + i)
        context.add_generated_label(context.next_generated_label_name(), series)


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
