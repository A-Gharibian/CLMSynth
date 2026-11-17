# SPDX-License-Identifier: MIT
# spatial_placement.py
"""
Which points of a cluster carry each label, once clm_label_engine has counted
them: whether the coordinates can be used, the score a feature_dependence entry
gives each point, and the choice of the points, by their distance to the
centroid, by a feature or at random.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .clm_errors import clm_error, clm_warn
from .config_tree import _PLACED_BY_DISTANCE, default
from .config_validation import _feature_entry, _placement

__all__ = ["assign_points_in_cluster"]

log: logging.Logger = logging.getLogger(__name__)

# The private clm_label key under which LabelGenerator.apply hands over the names of
# the coords columns: the engine takes an array, and a feature_dependence entry names
# its feature.
_FEATURE_COLUMNS = "_feature_columns"


def _ensure_coords(cfg: dict[str, Any], coords: np.ndarray | None, N: int) -> np.ndarray:
    """[CLM-125] guard: spatial placement needs per-point geometry with at least
    two features; one feature is out of scope. [CLM-133] when a coordinate it
    would read is NaN or infinite."""
    placement = _placement(cfg)
    if coords is None or np.size(coords) == 0:
        if placement:
            raise clm_error(125, placement=placement,
                            got="missing" if coords is None else "empty")
        return np.zeros((N, 1))
    coords = np.asarray(coords)
    if placement and (coords.ndim != 2 or coords.shape[1] < 2):
        raise clm_error(125, placement=placement, got=f"of shape {coords.shape}")
    if placement and coords.dtype.kind not in "fiub":
        raise clm_error(133, problem=f"coords holds {coords.dtype} values, not numbers, and "
                                     f"{placement} places points by them")
    bad = _count_non_finite(coords) if placement else 0
    if bad:
        raise clm_error(133, problem=f"{bad} value(s) in coords are NaN or infinite, and "
                                     f"{placement} places points by them")
    return coords


def _count_non_finite(coords: np.ndarray) -> int:
    """How many values of a float array are NaN or infinite; 0 for any other dtype."""
    return int((~np.isfinite(coords)).sum()) if coords.dtype.kind == "f" else 0


@dataclass
class _Follow:
    """One feature_dependence entry against a dataset: each point's score, higher
    is favored."""
    entry: Mapping[str, Any]
    label: int
    score: np.ndarray
    profile: str


def _followed_features(cfg: dict[str, Any], coords: np.ndarray) -> list[_Follow]:
    """The feature_dependence entries against this dataset's columns, in the order
    written. [CLM-125] when the dataset lacks an entry's feature, or the call has
    no column names; [CLM-133] when a point's distance to `value` is not finite."""
    entries = cfg.get("feature_dependence") or []
    if not entries:
        return []
    names = cfg.get(_FEATURE_COLUMNS)
    if names is None:
        raise clm_error(125, placement="feature_dependence",
                        got="an array without column names, and an entry names its feature "
                            "by column: label through a DatasetContext "
                            "(generate_additional_labels, or a run)")
    names = list(names)
    follows = []
    for i, entry in enumerate(entries):
        label, feature, favors, value, profile = _feature_entry(i, entry, cfg["num_classes"])
        if feature not in names:
            shown = ", ".join(str(n) for n in names[:12]) + (", ..." if len(names) > 12 else "")
            raise clm_error(125, placement=f"feature_dependence entry {i}",
                            got=f"without its feature {feature!r} (the dataset's features: "
                                f"{shown})")
        column = coords[:, names.index(feature)].astype(float)
        if favors == "high":
            score = column
        elif favors == "low":
            score = -column
        else:
            # An overflow is judged on the result, not announced by numpy first.
            with np.errstate(over="ignore", invalid="ignore"):
                score = -np.abs(column - value)
            if not np.isfinite(score).all():
                bad = int((~np.isfinite(score)).sum())
                raise clm_error(133, problem=f"the distance of {bad} point(s) to the value "
                                             f"{value!r} of feature_dependence entry {i} is "
                                             f"not finite, feature {feature!r} is too large "
                                             "for a distance to be computed. Standardize or "
                                             "rescale the features (for byoc, "
                                             "byoc_suite.standardize: true)")
        follows.append(_Follow(entry, label, score, profile))
    return follows


def _clusters_shown(clusters: Sequence[Any]) -> str:
    """'cluster 1', or 'clusters 70, 72'."""
    if len(clusters) == 1:
        return f"cluster {clusters[0]}"
    return "clusters " + ", ".join(str(k) for k in clusters)


def _warn_entries_ignored(follows: Sequence[_Follow],
                          rule_demand: Mapping[Any, Mapping[int, int]],
                          noise_labels: Mapping[Any, Mapping[Any, str]],
                          cluster_sizes: Mapping[Any, int]) -> None:
    """[CLM-305] for each feature_dependence entry that moves nothing, or nothing in
    some clusters: a `value` its favors does not read, a label the matching places
    nowhere, a cluster where the label is competing noise, and a cluster the label
    fills. `rule_demand` is the matching's demand, before competing_noise adds its
    own."""
    for follow in follows:
        entry, label = follow.entry, follow.label

        def warn(effect: str, cause: str, entry: Mapping[str, Any] = entry) -> None:
            clm_warn(log, 305, where="feature_dependence", entry=entry, effect=effect,
                     cause=cause)

        favors = entry.get("favors", default("clm_label.feature_dependence.favors"))
        if favors != "value" and entry.get("value") is not None:
            warn("reads no value",
                 f"'value' is read only under favors 'value', and its favors is {favors!r}")
        placed_in = [k for k, per_label in rule_demand.items() if per_label.get(label, 0) > 0]
        if not placed_in:
            warn("has no effect",
                 f"label {label} has no placed points: only an assignment_matrix row or "
                 "single_match places a label, and its other points are spillover or "
                 "competing noise, which an entry does not move")
            continue
        noise_in = [k for k in rule_demand if label in noise_labels.get(k, {})]
        if noise_in:
            warn(f"has no effect in {_clusters_shown(noise_in)}",
                 f"label {label} is a competing_noise label there, and those points stay "
                 "where competing_noise put them")
        filled = [k for k in placed_in if rule_demand[k][label] >= cluster_sizes[k]]
        if filled:
            warn(f"has no effect in {_clusters_shown(filled)}",
                 f"label {label} takes every point there, so there is nothing to choose "
                 "between")


# ---------------------------------------------------------------------------
# Spatial placement (centroid_dependence)
# ---------------------------------------------------------------------------

def _weighted_pick(idx_avail: np.ndarray, count: int, coreness_avail: np.ndarray,
                    profile: str, steepness: float, rng: np.random.Generator) -> np.ndarray:
    """Selects `count` of the available point indices by the configured
    profile, given each candidate's coreness score."""
    if _off_scale(coreness_avail):
        coreness_avail = _unit_span(coreness_avail)
    if profile == "step":
        jitter = rng.random(len(idx_avail)) * 1e-9
        order = np.argsort(-(coreness_avail + jitter))
        return idx_avail[order[:count]]
    if profile in ("linear", "exponential"):
        span = coreness_avail.max() - coreness_avail.min()
        scaled = (coreness_avail - coreness_avail.min()) / (span + 1e-12)
        w = scaled if profile == "linear" else np.exp(steepness * (scaled - 1))
        w += 1e-9
        return rng.choice(idx_avail, size=count, replace=False, p=w / w.sum())
    raise clm_error(110, profile=profile)


def _off_scale(score: np.ndarray) -> bool:
    """Whether the 1e-9 and 1e-12 of `_weighted_pick`, which are absolute, cannot
    serve this score: a span below 1e-6 or above 1e150, or one to which the score's
    dtype cannot add 1e-12. A span of 0 is not off scale: every point then ties."""
    if score.size == 0:
        return False
    span = score.max() - score.min()
    width = float(span)
    return bool(0 < width < 1e-6 or width > 1e150 or span + 1e-12 == 0)


def _unit_span(score: np.ndarray) -> np.ndarray:
    """`score` rescaled to 0..1 over its own span, so that the 1e-9 jitter of a
    'step' pick, which only breaks ties, stays below the differences of a feature
    measured in small units. Halved first: the span of two finite values can
    overflow."""
    if score.size == 0:
        return score
    half = score / 2
    span = half.max() - half.min()
    return (half - half.min()) / span if span > 0 else np.zeros(len(score))


def _distances_overflowed(distances: np.ndarray) -> ValueError:
    """[CLM-133] for finite coordinates too large to square: above 1e154, or 1e19
    in float32 and 255 in float16."""
    bad = int((~np.isfinite(distances)).sum())
    return clm_error(133, problem=f"the distance of {bad} point(s) to their cluster's centroid "
                                  "is not finite, the coordinates are too large for a distance "
                                  "to be computed. Standardize or rescale the features (for "
                                  "byoc, byoc_suite.standardize: true)")


def assign_points_in_cluster(
        coords: np.ndarray,
        label_targets: dict[int, int],
        spillover_label_draws: list[int],
        centroid_cfg: dict[str, Any],
        rng: np.random.Generator,
        favors_overrides: dict[int, str] | None = None,
        feature_scores: Mapping[int, tuple[np.ndarray, str]] | None = None,
) -> np.ndarray:
    """Picks the concrete rows of ONE cluster for each demanded label.
    `feature_scores` maps a label to each row's score (higher is favored) and a
    profile, 'linear' or 'step': those labels are placed first, in its order, each
    by its own score.
    """
    n = len(coords)
    out = np.full(n, -1, dtype=int)
    available = np.ones(n, dtype=bool)

    enabled = centroid_cfg.get("enabled", default("clm_label.centroid_dependence.enabled"))
    profile = centroid_cfg.get("profile", default("clm_label.centroid_dependence.profile"))
    favors = centroid_cfg.get("favors", default("clm_label.centroid_dependence.favors"))
    steepness = centroid_cfg.get("steepness", default("clm_label.centroid_dependence.steepness"))

    # Distances are needed by the global centroid_dependence AND by any core or
    # boundary competing_noise override, which places its label even when the
    # global setting is off.
    if enabled or any(v in _PLACED_BY_DISTANCE for v in (favors_overrides or {}).values()):
        # An overflow is judged on the result, not announced by numpy first.
        with np.errstate(over="ignore", invalid="ignore"):
            centroid = coords.mean(axis=0)
            distances = np.linalg.norm(coords - centroid, axis=1)
        if not np.isfinite(distances).all():
            raise _distances_overflowed(distances)
    else:
        distances = None

    if enabled:
        # `enabled` implies the branch above ran, so distances is real here.
        assert distances is not None
        coreness = -distances if favors == "core" else distances
    else:
        coreness = np.zeros(n)

    # The target is set by the matching and comes first; noise takes its core,
    # boundary or random points from what the targets leave. The counts, and so
    # MCC and ARI, are the same in either order: only which points carry a label.
    # A label that follows a feature goes before the other targets, or a bigger
    # label placed at random would take the points it favors.
    follows = feature_scores or {}
    ordered = [(label, label_targets[label]) for label in follows if label in label_targets]
    ordered += sorted((kv for kv in label_targets.items() if kv[0] not in follows),
                      key=lambda kv: (((favors_overrides or {}).get(kv[0]) is not None), -kv[1]))
    for label, count in ordered:
        if count == 0:
            continue
        idx_avail = np.where(available)[0]
        count = min(count, len(idx_avail))

        override = (favors_overrides or {}).get(label)
        if label in follows:
            score, follow_profile = follows[label]
            chosen = _weighted_pick(idx_avail, count, _unit_span(score[idx_avail]),
                                    follow_profile, steepness, rng)
        elif override in _PLACED_BY_DISTANCE:
            # competing_noise placement: per-label coreness sign, same profile
            # machinery as the global setting (defaults to 'linear' when the
            # global centroid_dependence is off).
            # A favors_override for this label means `favors_overrides` was
            # true, so distances was computed above.
            assert distances is not None
            sc = -distances if override == "core" else distances
            chosen = _weighted_pick(idx_avail, count, sc[idx_avail], profile, steepness, rng)
        elif override == "random" or not enabled:
            chosen = rng.choice(idx_avail, size=count, replace=False)
        else:
            chosen = _weighted_pick(idx_avail, count, coreness[idx_avail], profile, steepness, rng)

        out[chosen] = label
        available[chosen] = False

    leftover_idx = np.where(available)[0]
    rng.shuffle(leftover_idx)
    out[leftover_idx] = spillover_label_draws[: len(leftover_idx)]
    return out
