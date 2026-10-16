# clm_label_engine.py
"""
Assigns synthetic labels L(x) to each point x given its existing
ground-truth cluster c(x). Clusters and cluster membership are fixed
inputs from upstream, this module only decides which label each point
gets, per the clm_label config schema.
"""

import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from .clm_errors import (
    InfeasibleAllocationError,
    clm_error,
    clm_infeasible,
    clm_missing,
    clm_warn,
)
from .metrics import (
    MetricCeilings,
    clustering_ari,
    clustering_ari_pair,
    clustering_mcc,
    clustering_mcc_pair,
)

log = logging.getLogger(__name__)

MAX_CARDINALITY = 64
MATCHING_MODES = ("perfect", "single", "random", "custom")
BALANCE_VALUES = ("balanced", "unbalanced")


# ---------------------------------------------------------------------------
# Label totals: proportions / balance / skew_rule -> exact m_c per label
# ---------------------------------------------------------------------------

def _largest_remainder_counts(proportions: Sequence[float], total: int) -> list[int]:
    """Converts float proportions into integer counts summing exactly to `total`."""
    raw = [p * total for p in proportions]
    floors = [int(np.floor(x)) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda j: raw[j] - floors[j], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def _skewed_proportions(M: int, skew_rule: str, params: dict,
                         rng: np.random.Generator | None = None) -> list[float]:
    if skew_rule == "geometric":
        r = params.get("ratio", 0.5)
        raw = [r ** i for i in range(M)]
    elif skew_rule == "dominant_minority":
        dom_idx = params.get("dominant_index", 0)
        dom_share = params.get("dominant_share", 0.5)
        raw = [(1 - dom_share) / (M - 1)] * M
        raw[dom_idx] = dom_share
    elif skew_rule == "dirichlet":
        # Stochastic-but-reproducible skew: one Dirichlet(alpha,...,alpha) draw.
        # Small alpha -> near-degenerate imbalance, large alpha -> near-uniform.
        # Allows batch benchmarks to sample many imbalance scenarios by seed
        # instead of hand-picking proportion vectors.
        alpha = params.get("alpha", 1.0)
        rng = rng if rng is not None else np.random.default_rng(0)
        raw = rng.dirichlet([alpha] * M).tolist()
    else:
        raise clm_error(107, skew_rule=skew_rule)
    s = sum(raw)
    return [x / s for x in raw]


def resolve_label_counts(cfg: dict, N: int,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """balance='balanced' -> uniform 1/M, always (explicit proportions ignored;
    a warning is logged). Anything else -> explicit `proportions` take
    precedence, and `skew_rule` is the fallback when no proportions are given.
    generate_clm_labels refuses a balance outside BALANCE_VALUES before calling
    this ([CLM-132]), and does not call it under 'perfect'."""
    M = cfg["num_classes"]
    balance = cfg.get("balance", "balanced")

    if balance == "balanced":
        if cfg.get("proportions"):
            clm_warn(log, 301)
        proportions = [1.0 / M] * M
    elif cfg.get("proportions"):
        proportions = cfg["proportions"]
        # [CLM-141] A mapping ({0: 0.8, 1: 0.2}) is iterated by its KEYS: 0 + 1
        # passed [CLM-106] and every point got label 1. A NaN entry passed it too,
        # and a negative one sums to 1 beside a share above it (0.5, 0.6, -0.1, 0.0)
        # and gives its label a negative count.
        if (not isinstance(proportions, (list, tuple))
                or not all(_is_finite_real(p) and p >= 0 for p in proportions)):
            raise clm_error(141, key="proportions",
                            expected="a list of finite numbers >= 0, one per label",
                            value=proportions)
        # [CLM-121] Length must match M exactly. A longer list used to enlarge the
        # label space silently
        if len(proportions) != M:
            detail = (
                f"The surplus entries become labels {list(range(M, len(proportions)))}, "
                "which num_classes never declared; uniform/concentrated spillover then "
                "writes them into the dataset."
                if len(proportions) > M else
                f"Labels {list(range(len(proportions), M))} would be left with no share "
                "of the data."
            )
            raise clm_error(121, n=len(proportions), M=M, detail=detail)
        if abs(sum(proportions) - 1.0) > 1e-6:
            raise clm_error(106, total=sum(proportions))
    else:
        # `or {}`, not a .get default: a bare `skew_params:` key in YAML parses to
        # None, which .get would hand straight to _skewed_proportions as the params
        # mapping and crash on .get() there.
        if "skew_rule" not in cfg:
            raise clm_missing(203)
        proportions = _skewed_proportions(M, cfg["skew_rule"], cfg.get("skew_params") or {}, rng)

    return np.array(_largest_remainder_counts(proportions, N))


# ---------------------------------------------------------------------------
# matching_mode -> a uniform rule list: (label, clusters, recall_target)
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """One matching rule: route `recall_target` of label's budget into `clusters`."""
    label: int
    clusters: list[int]
    recall_target: float


def _check_pair(label: int, clusters: list, M: int, cluster_ids: list, where: str) -> None:
    """Fail fast on out-of-range label / unknown cluster ids in a matching
    rule, before allocate() would otherwise surface them as an uncoded numpy
    IndexError (label >= M), a KeyError (unknown cluster)."""
    is_int = isinstance(label, (int, np.integer)) and not isinstance(label, bool)
    if not is_int:
        raise clm_error(104, where=where, label=label, hi=M - 1,
                        problem="is not an integer; labels are indices in")
    if not (0 <= label < M):
        raise clm_error(104, where=where, label=label, hi=M - 1, problem="out of range")
    # [CLM-141] Configuration errors, so not [CLM-105], which skips one dataset. A
    # string was iterated by character.
    if not isinstance(clusters, (list, tuple)):
        raise clm_error(141, key=f"{where}: clusters", expected="a list of cluster ids",
                        value=clusters)
    for k in clusters:
        _check_cluster_id(k, f"{where}: cluster")
    unknown = [k for k in clusters if k not in cluster_ids]
    if unknown:
        raise clm_error(105, where=where, unknown=unknown,
                        available=sorted(cluster_ids, key=str))


def _check_named_once(named: dict[tuple[Any, Any], int], label, clusters: list, i: int) -> None:
    """[CLM-142] A cluster is named once per label across assignment_matrix. Two
    rows naming it added their points, so no entry stated the value; one row
    listing it twice lost points, since the split gave it two parts and kept one.
    `named` maps (label, cluster) to the row that named it, across calls."""
    in_row = set()
    for k in clusters:
        if k in in_row:
            raise clm_error(142, where=f"assignment_matrix row {i}", cluster=k, label=label,
                            fix="list each cluster once in a row.")
        in_row.add(k)
        first = named.setdefault((label, k), i)
        if first != i:
            raise clm_error(142, where=f"assignment_matrix rows {first} and {i}", cluster=k,
                            label=label, fix="keep it in one of the two rows.")


def _single_match(cfg: dict) -> dict[str, Any]:
    """single_match with both required keys."""
    sm: dict[str, Any] | None = cfg.get("single_match")
    if sm is None:
        raise clm_missing(204)
    _check_mapping(sm, "single_match", "{cluster: 0, label: 0}")
    for key in ("cluster", "label"):
        if key not in sm:
            raise clm_missing(205, key=key)
    return sm


def build_rules(cfg: dict, cluster_ids: list[int],
                 recall_target_override: float | None = None) -> list[Rule]:
    """Normalizes the matching mode into a validated list of `Rule` records.

    `recall_target_override` replaces every rule's recall with one global
    value; the target-metric solver passes its solved alpha through it.
    """
    mode = cfg["matching_mode"]
    M, K = cfg["num_classes"], len(cluster_ids)

    if mode == "perfect":
        if recall_target_override is not None:
            raise clm_error(111, mode="perfect")
        if M != K:
            raise clm_error(102, M=M, K=K)
        # Only warn about keys the caller actually set.
        if any(k in cfg for k in ("proportions", "balance", "skew_rule", "skew_params")):
            clm_warn(log, 302)
        return [Rule(label=i, clusters=[cluster_ids[i]], recall_target=1.0) for i in range(K)]

    if mode == "single":
        if M < 2 or K < 2:
            raise clm_error(103, M=M, K=K)
        sm = _single_match(cfg)
        _check_pair(sm["label"], [sm["cluster"]], M, cluster_ids, "single_match")
        rt = recall_target_override if recall_target_override is not None else 1.0
        return [Rule(label=sm["label"], clusters=[sm["cluster"]], recall_target=rt)]

    if mode == "custom":
        # `is None`, not falsiness: [] is zero rules, which is legal.
        matrix = cfg.get("assignment_matrix")
        if matrix is None:
            raise clm_missing(206)
        _check_list_of_mappings(matrix, "assignment_matrix", "row",
                                "{clusters: [0], label: 0, recall_target: 0.8}")
        rules = []
        named: dict[tuple[Any, Any], int] = {}   # (label, cluster) -> the row naming it
        for i, row in enumerate(matrix):
            where = f"assignment_matrix row {i}"
            for key in ("clusters", "label"):
                if key not in row:
                    raise clm_missing(207, where=where, key=key)
            _check_pair(row["label"], row["clusters"], M, cluster_ids, where)
            _check_named_once(named, row["label"], row["clusters"], i)
            if recall_target_override is not None:
                rt = recall_target_override
            elif "recall_target" not in row:
                raise clm_missing(208, where=where)
            else:
                rt = row["recall_target"]
                if not _is_finite_real(rt) or rt < 0:
                    raise clm_error(141, key=f"{where}: recall_target",
                                    expected="a finite number >= 0", value=rt)
            rules.append(Rule(label=row["label"], clusters=row["clusters"], recall_target=rt))
        return rules

    raise clm_error(101, mode=mode)


def validate_matching_ids(cfg: dict, cluster_ids: list) -> None:
    """Run only the [CLM-104]/[CLM-105] id checks against ONE dataset's cluster ids.

    Deliberately tolerant of a malformed config. [CLM-105] is a statement about a
    *dataset*, unlike every other [CLM-1xx] code. [CLM-104] bounds labels by
    num_classes, so it is configuration-scoped; main.py evaluates it once.
    """
    M = cfg.get("num_classes")
    if not isinstance(M, int) or isinstance(M, bool):
        return

    mode = cfg.get("matching_mode")
    if mode == "single":
        sm = cfg.get("single_match") or {}
        if isinstance(sm, Mapping) and "label" in sm and "cluster" in sm:
            _check_pair(sm["label"], [sm["cluster"]], M, cluster_ids, "single_match")
    elif mode == "custom":
        matrix = cfg.get("assignment_matrix")
        for i, row in enumerate(matrix if isinstance(matrix, (list, tuple)) else []):
            if isinstance(row, dict) and "label" in row and "clusters" in row:
                _check_pair(row["label"], row["clusters"], M, cluster_ids,
                            f"assignment_matrix row {i}")


def _validate_dataset_input(cluster_labels, coords, N: int) -> None:
    """[CLM-133] guard: one usable cluster id and, when given, one feature row
    per point. A missing id (NaN/None) is merged by np.unique into one "cluster"
    that, since NaN != NaN, counts no points, so its points were never labeled;
    ids mixing strings and numbers cannot be sorted; extra coords rows were
    silently dropped and too few raised IndexError. Pipeline runs meet only the
    first: byoc reports missing ids as [CLM-133] itself (byoc_source), reads ids of
    one column as one type, and builds coords from the same rows.
    """
    ids = np.asarray(cluster_labels)
    missing = int(np.sum(pd.isna(ids)))
    if missing:
        raise clm_error(133, problem=f"{missing} of the {N} cluster ids are missing "
                                     "(NaN/None); they would never be labelled")
    # Only an object array can hold both; a numeric or str array cannot.
    if ids.dtype == object and len({isinstance(v, str) for v in ids}) > 1:
        raise clm_error(133, problem="the cluster ids mix strings and numbers, which "
                                     "cannot be ordered")
    if coords is not None and np.size(coords) > 0 and len(coords) != N:
        raise clm_error(133, problem=f"coords has {len(coords)} rows for {N} cluster ids")


def _ensure_coords(cfg: dict, coords, N: int) -> np.ndarray:
    """[CLM-125] guard: spatial placement needs real per-point geometry.
    Placement is requested by `centroid_dependence.enabled` or by any
    competing_noise entry whose favors is 'core'/'boundary' (the default is
    'boundary'). Both blocks are first read here, in every mode, so their shape
    ([CLM-141]) is checked here too.
    """
    cd = cfg.get("centroid_dependence") or {}
    _check_mapping(cd, "centroid_dependence", "{enabled: true, favors: core}")
    # [CLM-141] Checked here, where it is first read. A quoted "false" or "no" is a
    # non-empty string, so it used to switch placement on.
    if "enabled" in cd and not isinstance(cd["enabled"], (bool, np.bool_)):
        raise clm_error(141, key="centroid_dependence.enabled", expected="true or false",
                        value=cd["enabled"])
    noise = cfg.get("competing_noise") or []
    _check_list_of_mappings(noise, "competing_noise", "entry",
                            "{cluster: 1, label: 2, share: 0.5}")
    spatial_noise = [e for e in noise if e.get("favors", "boundary") != "random"]
    placement = None
    if cd.get("enabled"):
        placement = "centroid_dependence.enabled"
    elif spatial_noise:
        placement = "competing_noise with favors 'core'/'boundary'"

    if coords is None or np.size(coords) == 0:
        if placement:
            raise clm_error(125, placement=placement,
                            got="missing" if coords is None else "empty")
        return np.zeros((N, 1))
    return np.asarray(coords)


def _validate_target_metric_cfg(cfg: dict, cluster_ids: list) -> None:
    # An absent OR empty/null target_metric (e.g. `target_metric:` in YAML -> None)
    # is a no-op
    tm = cfg.get("target_metric")
    if not tm:
        return
    mode = cfg["matching_mode"]
    if mode not in ("single", "custom"):
        raise clm_error(111, mode=mode)
    _check_mapping(tm, "target_metric", "{type: mcc, value: 0.5}")
    if tm.get("type") not in ("mcc", "ari"):
        raise clm_error(112, type=tm.get("type"))
    if "value" not in tm:
        raise clm_missing(209)
    value = tm["value"]
    # True passed as 1; a bare `value:` (None) or a string crashed the comparison.
    if not _is_real(value) or not (-1.0 <= value <= 1.0):
        raise clm_error(113, value=value if _is_real(value) else repr(value))
    # [CLM-141] A NaN, infinite or boolean tolerance meant [CLM-309]/[CLM-310]
    # could never fire.
    tol = tm.get("tolerance", 0.01)
    if not _is_finite_real(tol) or tol < 0:
        raise clm_error(141, key="target_metric.tolerance", expected="a finite number >= 0",
                        value=tol)
    scope = tm.get("scope", "global")
    if scope not in ("pair", "global"):
        raise clm_error(122, scope=scope)
    # max_iter is read only by the global search. True ran a single iteration; a
    # negative value skipped the refinement without a word.
    max_iter = tm.get("max_iter", 40)
    if scope == "global" and (isinstance(max_iter, (bool, np.bool_))
                              or not isinstance(max_iter, (int, np.integer)) or max_iter < 0):
        raise clm_error(141, key="target_metric.max_iter", expected="an integer >= 0",
                        value=max_iter)
    # probe_seed too is read only by the global search; empty is the run seed, as
    # absent. True seeded the probes as 1, and a negative, a float or a string
    # crashed uncoded in numpy at the first probe.
    probe_seed = tm.get("probe_seed")
    if scope == "global" and probe_seed is not None and (
            isinstance(probe_seed, (bool, np.bool_))
            or not isinstance(probe_seed, (int, np.integer)) or probe_seed < 0):
        raise clm_error(141, key="target_metric.probe_seed", expected="an integer >= 0",
                        value=probe_seed)
    if scope == "pair":
        # The single-pair MCC inverts in closed form (see _pair_label_counts);
        # the pair is taken from single_match.
        # Pair ARI has no closed-form inverse, so it is refused here. The search
        # over sizes in _pair_ari_size stays in the package, unused.
        if tm["type"] != "mcc":
            raise clm_error(123)
        if mode != "single":
            raise clm_error(124)
        # Validate the (cluster, label) pair
        sm = _single_match(cfg)
        _check_pair(sm["label"], [sm["cluster"]], cfg["num_classes"], cluster_ids, "single_match")
        # [CLM-130] The closed form is only exact while EVERY point of l* stays
        # inside k*
        lstar = sm["label"]
        spill = cfg.get("spillover_rule", "proportional_to_marginal")
        bad = None
        if spill == "uniform":
            # Draws uniformly over all M labels, so l* lands outside k* by design.
            bad = "spillover_rule 'uniform'"
        elif spill == "concentrated":
            targets = cfg.get("concentrated_labels")
            if targets is None:
                # Defaults to the largest label, resolved AFTER l* is resized, so
                # whether it picks l* cannot be known here.
                bad = ("spillover_rule 'concentrated' without an explicit "
                       "concentrated_labels (it defaults to the largest label, which "
                       "may be the target label)")
            elif lstar in targets:
                bad = f"spillover_rule 'concentrated' targeting label {lstar} itself"
        if bad is None:
            for entry in (cfg.get("competing_noise") or []):
                if entry.get("label") == lstar:
                    bad = f"a competing_noise entry emitting label {lstar}"
                    break
        if bad is not None:
            raise clm_error(130, what=bad, label=lstar, cluster=sm["cluster"])


def _validate_spillover_cfg(cfg: dict) -> None:
    """
    [CLM-128] guard: concentrated_labels must name labels that actually exist.
    """
    if cfg.get("spillover_rule") != "concentrated":
        return
    given = cfg.get("concentrated_labels")
    if given is None:
        return                      # documented default: the single largest label

    M = cfg["num_classes"]
    valid = isinstance(given, (list, tuple)) and len(given) > 0
    if valid:
        for v in given:
            try:                    # accept 2 and 2.0, reject 1.5, True, "2", None, inf
                is_int = not isinstance(v, bool) and v == int(v)
            except (TypeError, ValueError, OverflowError):
                is_int = False
            if not is_int or not (0 <= v < M):
                valid = False
                break
    if not valid:
        raise clm_error(128, hi=M - 1, given=given)


def _is_real(value) -> bool:
    """True for a real number, excluding bool (which is an int subclass and would
    otherwise sail through every range check as 0/1)."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_finite_real(value) -> bool:
    """_is_real, excluding NaN and inf. Every comparison with NaN is False, so a NaN
    passes any range check that refuses only the out-of-range side."""
    return _is_real(value) and math.isfinite(value)


def _check_mapping(value, key: str, example: str) -> None:
    """[CLM-141] A block read key by key must be a mapping. Anything else crashed
    uncoded on its first .get(), or passed `in` as a list or a string and was then
    reported as a missing key."""
    if not isinstance(value, Mapping):
        raise clm_error(141, key=key, expected=f"a mapping such as {example}", value=value)


def _check_list_of_mappings(value, key: str, item: str, example: str) -> None:
    """[CLM-141] A list block whose entries are mappings. A single mapping written
    without the YAML dash was iterated by its keys, and a string by its characters."""
    if not isinstance(value, (list, tuple)):
        raise clm_error(141, key=key, expected=f"a list of mappings such as [{example}]",
                        value=value)
    for i, entry in enumerate(value):
        _check_mapping(entry, f"{key} {item} {i}", example)


def _check_cluster_id(k, key: str) -> None:
    """[CLM-141] One cluster id: a number or a string. True == 1 matched cluster 1,
    and a list or a mapping either crashed as unhashable or was reported as an id
    the dataset lacks ([CLM-105], which skips the dataset as if the data were at
    fault). A null or NaN id is no id at all: it too was read as a cluster the
    dataset lacks ([CLM-105]/[CLM-119])."""
    missing = k is None or (isinstance(k, (float, np.floating)) and np.isnan(k))
    if missing or isinstance(k, (bool, np.bool_, list, tuple, dict, set)):
        raise clm_error(141, key=key, expected="one cluster id (a number or a string)", value=k)


def _validate_balance_cfg(cfg: dict) -> None:
    """[CLM-132] guard: balance must be exactly 'balanced' or 'unbalanced'.
    """
    if "balance" in cfg and cfg["balance"] not in BALANCE_VALUES:
        raise clm_error(132, balance=cfg["balance"])


def _validate_skew_cfg(cfg: dict) -> None:
    """[CLM-131] guard: skew parameters must be in range for the chosen skew_rule.

    Called from generate_clm_labels *before* resolve_label_counts,
    Unknown skew_rule values stay [CLM-107], raised by _skewed_proportions itself.
    """
    if cfg.get("balance", "balanced") == "balanced" or cfg.get("proportions"):
        return

    M = cfg["num_classes"]
    params = cfg.get("skew_params")
    if params is None:
        params = {}                      # bare `skew_params:` key: take the defaults
    if not isinstance(params, dict):
        raise clm_error(131, hi=M - 1,
                        problem=f"expected a mapping of parameters, got {params!r}")

    rule = cfg.get("skew_rule")
    problem = None

    if rule == "geometric":
        ratio = params.get("ratio", 0.5)
        if not _is_finite_real(ratio) or ratio < 0:
            problem = (f"geometric 'ratio' must be a finite number >= 0, got {ratio!r}; a "
                       "negative ratio alternates sign across labels")

    elif rule == "dominant_minority":
        if M < 2:
            problem = ("dominant_minority needs num_classes >= 2, got 1; the rule "
                       "divides the remaining share by (num_classes - 1)")
        else:
            share = params.get("dominant_share", 0.5)
            index = params.get("dominant_index", 0)
            if not _is_real(share) or not (0.0 <= share <= 1.0):
                problem = (f"dominant_minority 'dominant_share' must be a number in "
                           f"[0, 1], got {share!r}")
            elif isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                problem = (f"dominant_minority 'dominant_index' must be an integer, got "
                           f"{index!r}; it indexes the proportions list directly")
            elif not (0 <= index < M):
                problem = (f"dominant_minority 'dominant_index' must be in 0..{M - 1}, "
                           f"got {index!r}")

    elif rule == "dirichlet":
        alpha = params.get("alpha", 1.0)
        if not _is_finite_real(alpha) or alpha <= 0:
            problem = (f"dirichlet 'alpha' must be a finite number > 0, got {alpha!r}; at "
                       "exactly 0 every draw is 0 and normalising them divides by zero")

    if problem is not None:
        raise clm_error(131, problem=problem, hi=M - 1)


def _validate_centroid_cfg(cfg: dict) -> None:
    """[CLM-129] favors, [CLM-110] profile and [CLM-141] steepness, each checked
    wherever placement reads it.
    """
    cd = cfg.get("centroid_dependence") or {}
    enabled = cd.get("enabled", False)
    if enabled:
        favors = cd.get("favors", "core")
        if favors not in ("core", "boundary"):
            raise clm_error(129, favors=favors)
    noise_placed = any(e.get("favors", "boundary") in ("core", "boundary")
                       for e in cfg.get("competing_noise") or [])
    if not (enabled or noise_placed):
        return
    profile = cd.get("profile", "linear")
    if profile not in ("linear", "exponential", "step"):
        raise clm_error(110, profile=profile)
    steepness = cd.get("steepness", 3.0)
    if profile == "exponential" and (not _is_finite_real(steepness) or steepness < 0):
        raise clm_error(141, key="centroid_dependence.steepness",
                        expected="a finite number >= 0", value=steepness)


def _validate_split_cfg(cfg: dict) -> None:
    """[CLM-108] guard: split_rule must name a known rule wherever 'custom' reads it.
    """
    if cfg.get("matching_mode") != "custom":
        return
    split_rule = cfg.get("split_rule", "proportional_to_size")
    if split_rule not in ("proportional_to_size", "equal"):
        raise clm_error(108, split_rule=split_rule)


def _noise_entry(i: int, entry: Mapping[str, Any], M: int) -> tuple[Any, int, Any, str]:
    """One competing_noise entry as (cluster, label, share, favors), defaults filled
    in, after the checks that hold for every dataset."""
    # A bare KeyError used to log only 'cluster'.
    for key in ("cluster", "label"):
        if key not in entry:
            raise clm_missing(207, where=f"competing_noise entry {i}", key=key)
    k, label = entry["cluster"], entry["label"]
    share = entry.get("share", 1.0)
    favors = entry.get("favors", "boundary")

    # [CLM-141], not [CLM-119]: a config error, not a dataset's.
    _check_cluster_id(k, f"competing_noise entry {i}: cluster")
    is_int = isinstance(label, (int, np.integer)) and not isinstance(label, bool)
    if not is_int:
        raise clm_error(118, label=label, hi=M - 1,
                        problem="is not an integer; labels are indices in")
    if not (0 <= label < M):
        raise clm_error(118, label=label, hi=M - 1, problem="out of range")
    # True passed as 1.0; a string crashed the comparison.
    if not _is_real(share) or not (0.0 <= share <= 1.0):
        raise clm_error(117, share=share if _is_real(share) else repr(share))
    if favors not in ("core", "boundary", "random"):
        raise clm_error(116, favors=favors)
    return k, label, share, favors


def _plain_pair(label, k) -> bool:
    """An integer label and a number-or-string cluster id: what build_rules accepts.
    Anything else is left for it to report ([CLM-104], [CLM-141])."""
    return (isinstance(label, (int, np.integer)) and not isinstance(label, bool)
            and isinstance(k, (int, float, str, np.integer, np.floating))
            and not isinstance(k, bool))


def _named_targets(cfg: dict) -> dict[tuple[Any, Any], str]:
    """(label, cluster) -> where the config makes the label a target of that cluster,
    for [CLM-144]. Read from the config, so it holds for every dataset. 'perfect'
    pairs labels with clusters by the dataset's ids, and leaves no unclaimed point
    for noise anyway ([CLM-305]), so it is left out. Malformed parts are skipped:
    build_rules reports them."""
    targets: dict[tuple[Any, Any], str] = {}
    if cfg["matching_mode"] == "single":
        sm = cfg.get("single_match")
        if isinstance(sm, Mapping) and _plain_pair(sm.get("label"), sm.get("cluster")):
            targets[(sm["label"], sm["cluster"])] = "single_match"
    elif cfg["matching_mode"] == "custom":
        matrix = cfg.get("assignment_matrix")
        for j, row in enumerate(matrix if isinstance(matrix, (list, tuple)) else []):
            if not (isinstance(row, Mapping) and isinstance(row.get("clusters"), (list, tuple))):
                continue
            for k in row["clusters"]:
                if _plain_pair(row.get("label"), k):
                    targets.setdefault((row["label"], k), f"assignment_matrix row {j}")
    return targets


def _validate_competing_noise_cfg(cfg: dict) -> None:
    """Each competing_noise entry, checked once before allocation and the
    target_metric search: [CLM-207], [CLM-141], [CLM-118], [CLM-117] and
    [CLM-116], then [CLM-144] and [CLM-142]. All of them hold for every dataset.
    Raised during allocation, they came after any [CLM-15x], and under
    target_metric they could end as [CLM-120]. [CLM-119] is about a dataset's ids
    and stays with the allocation.

    A label gets one entry per cluster, either/or: two entries placing it core and
    boundary kept only the last favors and summed the points, so the core half
    went to the boundary.
    """
    targets = _named_targets(cfg)
    placed: dict[tuple[Any, Any], tuple[int, str, Any]] = {}
    for i, entry in enumerate(cfg.get("competing_noise") or []):
        k, label, share, favors = _noise_entry(i, entry, cfg["num_classes"])
        if (label, k) in targets:
            raise clm_error(144, i=i, label=label, cluster=k, source=targets[(label, k)])
        if (label, k) in placed:
            j, favors_j, share_j = placed[(label, k)]
            total = share_j + share
            if favors != favors_j:
                fix = f"entry {j} places it at '{favors_j}' and entry {i} at '{favors}'; choose one."
            elif total <= 1.0:
                fix = f"write one entry with share {total:g}."
            else:
                fix = (f"together they ask for share {total:g}, and an entry's share is at "
                       "most 1.0; write one entry.")
            raise clm_error(142, where=f"competing_noise entries {j} and {i}", cluster=k,
                            label=label, fix=fix)
        placed[(label, k)] = (i, favors, share)


def _pair_ari_curve(N: int, n_k: int) -> np.ndarray:
    """The pair ARI the single-dominant construction delivers at m_c = 1..n_k.

    All m_c points of l* inside k* make the 2x2 table [[m_c, n_k - m_c],
    [0, N - n_k]]. Counted in pairs of points: A share k* or its rest, B share l*
    or its rest, S share both, out of P; each is a quadratic in m_c, and the
    pair ARI is sklearn's 2(tp tn - fn fp) / ((tp + fn)(fn + tn) + (tp + fp)(fp + tn))
    with tp = S, fn = A - S, fp = B - S and tn = P - A - B + S. It rises with
    m_c while k* holds at most half the points; past that it first dips below 0,
    then rises to 1 at m_c = n_k. Evaluated at every m_c: a search over this one
    construction, not an inverse.

    Unused by the engine: [CLM-123] refuses scope 'pair' with type 'ari', since
    setting ARI to a target has no closed-form solution.
    """
    m = np.arange(1, n_k + 1, dtype=np.float64)
    rest = N - n_k
    pairs = N * (N - 1) / 2
    a = n_k * (n_k - 1) / 2 + rest * (rest - 1) / 2
    b = m * (m - 1) / 2 + (N - m) * (N - m - 1) / 2
    s = m * (m - 1) / 2 + (n_k - m) * (n_k - m - 1) / 2 + rest * (rest - 1) / 2
    fn, fp = a - s, b - s
    tn = pairs - a - b + s
    with np.errstate(divide="ignore", invalid="ignore"):
        ari = 2 * (s * tn - fn * fp) / (a * (pairs - b) + b * (pairs - a))
    # sklearn's special case: identical partitions score 1.
    return np.where((fn == 0) & (fp == 0), 1.0, ari)


def _pair_ari_size(N: int, n_k: int, target: float) -> int:
    """The size m_c of l* whose pair ARI is closest to `target`.

    Taken on the rising side of the curve: past its dip two sizes can score the
    same, and the larger is the side every target above the dip lands on.
    Unused by the engine, see `_pair_ari_curve`.
    """
    curve = _pair_ari_curve(N, n_k)
    low = int(np.argmin(curve))
    return low + 1 + int(np.argmin(np.abs(curve[low:] - target)))


def pair_reachable_range(N: int, n_k: int, kind: str = "mcc") -> tuple[float, float]:
    """The pair MCC that scope 'pair' can deliver for a cluster of n_k of N points.

    The single-dominant construction places m_c in [1, n_k] points of label l*
    entirely inside k*. The pair MCC rises with m_c: phi_min at m_c = 1,
    phi_max = 1 at m_c = n_k (an exact 1:1 pair). Shared by the engine and the
    pipeline's summary txt, so the range is computed one way only.

    kind='ari' gives the range of the pair ARI over the same sizes: it also ends
    at 1, but its floor is the bottom of its curve, below 0 when k* holds more
    than half the points. The engine never asks for it ([CLM-123]).
    """
    if kind == "ari":
        return float(_pair_ari_curve(N, n_k).min()), 1.0
    phi_min = float(np.sqrt((N - n_k) / (n_k * (N - 1)))) if (n_k >= 1 and N > 1) else 0.0
    return phi_min, 1.0


def _pair_label_counts(cfg: dict, cluster_sizes: dict[int, int],
                       m_counts: np.ndarray, N: int) -> np.ndarray:
    """Exact single-pair MCC target via the single-dominant construction
    (target_metric.scope='pair', single mode). The 'ari' branch is unused:
    [CLM-123] refuses a pair ARI target before this runs.
    """
    tm = cfg["target_metric"]
    sm = cfg["single_match"]
    kind = tm["type"]
    lstar, n_k = sm["label"], cluster_sizes[sm["cluster"]]
    phi = tm["value"]

    # A request outside the reachable range is clamped to the nearest reachable
    # value with [CLM-307]; the subset construction can not produce a negative phi.
    # The range is logged either way, so a run inside it reports it too.
    phi_min, phi_max = pair_reachable_range(N, n_k, kind)
    log.info(f"target_metric scope 'pair': reachable pair {kind} for cluster "
             f"{sm['cluster']}, label {lstar} is [{phi_min:.4f}, {phi_max:.4f}] "
             f"(N={N}, cluster size {n_k}).")
    if not (phi_min <= phi <= phi_max):
        clm_warn(log, 307, target=phi, phi_min=phi_min, phi_max=phi_max)
        phi = min(max(phi, phi_min), phi_max)

    if kind == "ari":
        m_star = _pair_ari_size(N, n_k, phi)
    else:
        phi2 = phi * phi
        denom = N - n_k * (1.0 - phi2)
        m_star = n_k if (phi >= 1.0 or denom <= 0) else int(round(phi2 * n_k * N / denom))
    m_star = min(max(m_star, 1), n_k)                # a present label, capped at the cluster
    clm_warn(log, 308, label=lstar, m=m_star, metric=kind.upper())

    out = np.asarray(m_counts, dtype=int).copy()
    others = [j for j in range(len(out)) if j != lstar]
    if others:
        # Rescale the other labels to fill N - m_star so the counts still sum to N.
        # When they were all zero (o_sum == 0, reachable via a proportion of exactly
        # 0 or an extreme skew), fall back to a uniform split instead of leaving them
        # at zero, which would make sum(out) == m_star != N and later starve the
        # proportional_to_marginal spillover pool (truncated fill -> -1 labels).
        o_sum = int(sum(int(out[j]) for j in others))
        props = ([int(out[j]) / o_sum for j in others] if o_sum > 0
                 else [1.0 / len(others)] * len(others))
        # strict: `props` is built with one entry per label in `others`, so a
        # length mismatch would mean a label silently lost its share.
        for idx, count in zip(others, _largest_remainder_counts(props, N - m_star),
                              strict=True):
            out[idx] = count
    out[lstar] = m_star
    return out


# ---------------------------------------------------------------------------
# Feasibility + allocation
# ---------------------------------------------------------------------------

def _split_row_allocation(tp_row: int, clusters: list[int], sizes: dict[int, int], split_rule: str) -> dict[int, int]:
    if len(clusters) == 1:
        return {clusters[0]: tp_row}
    if split_rule == "equal":
        weights = [1.0] * len(clusters)
    elif split_rule == "proportional_to_size":
        weights = [sizes[k] for k in clusters]
    else:
        raise clm_error(108, split_rule=split_rule)
    total_w = sum(weights)
    counts = _largest_remainder_counts([w / total_w for w in weights], tp_row)
    # strict: `counts` comes from one weight per cluster, so an unequal zip here
    # would drop a cluster's allocation without a word.
    return dict(zip(clusters, counts, strict=True))


def _size_source(cfg: dict, label: int) -> str:
    """Names the setting a label's point budget came from, for [CLM-150]. Mirrors
    resolve_label_counts' precedence; 'perfect' never reaches [CLM-150]. It must
    not raise: it runs inside the target_metric solver's probes, which catch only
    InfeasibleAllocationError."""
    if cfg.get("balance", "balanced") == "balanced":
        return f"balance 'balanced' (1/{cfg['num_classes']} of the points per label)"
    props = cfg.get("proportions")
    if props:
        # Indexed only when it is a list of M entries, so this cannot raise.
        if isinstance(props, (list, tuple)) and len(props) == cfg["num_classes"]:
            return f"proportions[{label}] = {props[label]}"
        return "proportions"
    return f"skew_rule {cfg.get('skew_rule')!r}"


def _infeasible_remedy(cfg: dict, rule: Rule, m_label: int, capacity: int) -> str:
    """The [CLM-150] advice. 'single' has no recall_target key to lower: it places
    the whole label in its cluster, so only the label's size can change. Under
    target_metric there is none either: the recall is the level the search is
    probing, the same for every rule, so the advice names that level."""
    source = _size_source(cfg, rule.label)
    if cfg["matching_mode"] == "single" and not cfg.get("target_metric"):
        return (f"matching_mode 'single' places all of label {rule.label} in that cluster, "
                f"so the label can have at most {capacity} points; its {m_label} come from "
                f"{source}.")
    if cfg.get("target_metric"):
        # Only a probe gets here: the solved level was feasible, and scope 'pair'
        # sizes the label to fit. A level is at most 1, so capacity < m_label.
        max_rt = math.floor(capacity / m_label * 1000) / 1000
        more = "" if cfg["matching_mode"] == "single" else ", or give the rule more clusters"
        return (f"Under target_metric this is the search's recall level "
                f"alpha={rule.recall_target:.3f}, not a recall_target to lower; levels up "
                f"to {max_rt:.3f} fit here. For a higher level, make label {rule.label} "
                f"smaller (its {m_label} points come from {source}){more}.")
    if capacity >= m_label:
        # The clusters could hold the whole label, so only a recall_target above 1
        # gets here (e.g. 80 meant as 80 %).
        return (f"The rule's recall_target={rule.recall_target} is a fraction of the "
                f"label's points and must be at most 1.0.")
    # Rounded down: an advised value rounded up can itself be infeasible.
    max_rt = math.floor(capacity / m_label * 1000) / 1000
    return (f"The rule's recall_target={rule.recall_target} is feasible up to "
            f"{max_rt:.3f} here. Lower it, or make label {rule.label} smaller; its "
            f"{m_label} points come from {source}.")


def allocate(cfg: dict, rules: list[Rule], m_counts: np.ndarray, cluster_sizes: dict[int, int]):
    """Turns rules into integer per-(cluster, label) point demands.
    Checks each rule's budget against its clusters' capacity ([CLM-150]), each
    label against the joint claims on it ([CLM-153]), and each cluster against
    the joint claims on it ([CLM-151]). Returns the demand mapping and the
    per-cluster capacity left for spillover.
    """
    split_rule = cfg.get("split_rule", "proportional_to_size")
    demand: dict[int, dict[int, int]] = {k: {} for k in cluster_sizes}
    claimed_per_label: dict[int, int] = {}
    rules_per_label: dict[int, list[str]] = {}

    for rule in rules:
        m_label = m_counts[rule.label]
        tp_row = int(round(rule.recall_target * m_label))
        capacity = sum(cluster_sizes[k] for k in rule.clusters)

        if tp_row > capacity:
            raise clm_infeasible(150, label=rule.label, tp=tp_row, m=m_label,
                                 clusters=rule.clusters, capacity=capacity,
                                 remedy=_infeasible_remedy(cfg, rule, m_label, capacity))

        claimed_per_label[rule.label] = claimed_per_label.get(rule.label, 0) + tp_row
        rules_per_label.setdefault(rule.label, []).append(
            f"clusters {rule.clusters} at recall {rule.recall_target} = {tp_row}")

        for k, count in _split_row_allocation(tp_row, rule.clusters, cluster_sizes, split_rule).items():
            demand[k][rule.label] = demand[k].get(rule.label, 0) + count

    for label, claimed in claimed_per_label.items():
        budget = int(m_counts[label])
        if claimed > budget:
            raise clm_infeasible(153, label=label, claimed=claimed, budget=budget,
                                 n_rules=len(rules_per_label[label]),
                                 breakdown="; ".join(rules_per_label[label]))

    remaining_capacity = {}
    for k, size in cluster_sizes.items():
        claimed = sum(demand[k].values())
        if claimed > size:
            raise clm_infeasible(151, k=k, size=size, claimed=claimed, per_label=demand[k])
        remaining_capacity[k] = size - claimed

    return demand, remaining_capacity


def _spillover_draws(cfg: dict, m_counts: np.ndarray, used_per_label: np.ndarray,
                      n_spillover: int, rng: np.random.Generator) -> list[int]:
    rule_name = cfg.get("spillover_rule", "proportional_to_marginal")
    # Read M from the CONFIG, not from len(m_counts). Deriving it from the counts
    # array is what let a too-long `proportions` widen the label space here; the
    # two are now equal by construction ([CLM-121]), so this only makes the
    # invariant explicit and keeps a future mismatch from silently reappearing.
    M = cfg["num_classes"]

    if rule_name == "proportional_to_marginal":
        # Clipped at 0: competing_noise can push a label past its target
        # marginal, and negative repeats would crash np.repeat. The clipped
        # pool is always >= the spillover need (clipping only enlarges the
        # sum), and the per-cluster slicing in the pipeline truncates any
        # excess. A no-op whenever competing_noise is absent.
        remaining = np.clip(m_counts - used_per_label, 0, None)
        draws = np.repeat(np.arange(M), remaining)
        rng.shuffle(draws)
        return draws.tolist()
    if rule_name == "uniform":
        uniform: list[int] = rng.integers(0, M, size=n_spillover).tolist()
        return uniform
    if rule_name == "concentrated":
        targets = cfg.get("concentrated_labels") or [int(np.argmax(m_counts))]
        concentrated: list[int] = rng.choice(targets, size=n_spillover).tolist()
        return concentrated
    raise clm_error(109, rule_name=rule_name)


# ---------------------------------------------------------------------------
# Spatial placement (centroid_dependence)
# ---------------------------------------------------------------------------

def _weighted_pick(idx_avail: np.ndarray, count: int, coreness_avail: np.ndarray,
                    profile: str, steepness: float, rng: np.random.Generator) -> np.ndarray:
    """Selects `count` of the available point indices by the configured
    profile, given each candidate's coreness score."""
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


def assign_points_in_cluster(
        coords: np.ndarray,
        label_targets: dict[int, int],
        spillover_label_draws: list[int],
        centroid_cfg: dict,
        rng: np.random.Generator,
        favors_overrides: dict[int, str] | None = None,
) -> np.ndarray:
    """Picks the concrete rows of ONE cluster for each demanded label.

    The target labels are placed first (centroid-weighted when
    `centroid_dependence` is enabled), then the competing_noise labels, each by
    its own `favors_overrides` entry, then the leftover rows consume the spillover
    draws. A label is never both here ([CLM-144]). Returns the cluster's label
    column.
    """
    n = len(coords)
    out = np.full(n, -1, dtype=int)
    available = np.ones(n, dtype=bool)

    enabled = centroid_cfg.get("enabled", False)
    profile = centroid_cfg.get("profile", "linear")
    favors = centroid_cfg.get("favors", "core")
    steepness = centroid_cfg.get("steepness", 3.0)

    # Distances are needed by the global centroid_dependence AND by any
    # competing_noise override, which places its label core/boundary even
    # when the global setting is off.
    if enabled or favors_overrides:
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
    else:
        distances = None

    if enabled:
        # `enabled` implies the branch above ran, so distances is real here. The
        # assert states that for a reader and for a type checker, which cannot
        # correlate this condition with the one four lines up. Deliberately not
        # a zeros array in the else-branch: [CLM-125] exists for the same reason.
        assert distances is not None
        coreness = -distances if favors == "core" else distances
    else:
        coreness = np.zeros(n)

    # The target is set by the matching and comes first; noise takes its core,
    # boundary or random points from what the targets leave. The counts, and so
    # MCC and ARI, are the same in either order: only which points carry a label.
    ordered = sorted(label_targets.items(),
                     key=lambda kv: (((favors_overrides or {}).get(kv[0]) is not None), -kv[1]))
    for label, count in ordered:
        if count == 0:
            continue
        idx_avail = np.where(available)[0]
        count = min(count, len(idx_avail))

        override = (favors_overrides or {}).get(label)
        if override in ("core", "boundary"):
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


# ---------------------------------------------------------------------------
# Optional structured noise (competing_noise)
# ---------------------------------------------------------------------------

def _competing_demand(cfg: dict, remaining_capacity: dict[int, int], M: int,
                      warn: bool = True):
    """
    OPTIONAL feature, active only when the config carries a 'competing_noise'
    list;
    Each entry converts `share` of ONE cluster's UNCLAIMED points (leftover
    after all rules were allocated) into one specific competing label, placed
    core/boundary/random within that cluster:

        competing_noise:
          - {cluster: 1, label: 2, share: 1.0, favors: boundary}

    `warn=False` silences [CLM-305] only; every error is still raised.
    """
    extra: dict[int, dict[int, int]] = {}
    overrides: dict[int, dict[int, str]] = {}

    for i, entry in enumerate(cfg["competing_noise"]):
        # The checks that hold for every dataset; generate_clm_labels has run them
        # already, with [CLM-142]/[CLM-144] (_validate_competing_noise_cfg).
        k, label, share, favors = _noise_entry(i, entry, M)
        if k not in remaining_capacity:
            raise clm_error(119, cluster=k, available=sorted(remaining_capacity, key=str))

        count = int(round(share * remaining_capacity[k]))
        if count == 0:
            if warn:
                clm_warn(log, 305, k=k, entry=entry)
            continue
        extra.setdefault(k, {})
        extra[k][label] = extra[k].get(label, 0) + count
        overrides.setdefault(k, {})[label] = favors

    for k, per_label in extra.items():
        claimed = sum(per_label.values())
        if claimed > remaining_capacity[k]:
            raise clm_infeasible(152, k=k, claimed=claimed, remaining=remaining_capacity[k])

    # No [CLM-304] here: whether the noise moves the label counts is known only
    # after the spillover fill, so generate_clm_labels checks the delivered counts.
    return extra, overrides


# ---------------------------------------------------------------------------
# Shared allocation pipeline (used both by the probe search and the final commit)
# ---------------------------------------------------------------------------

def _run_allocation_pipeline(cluster_labels, coords, cfg, rules, cluster_ids,
                              cluster_sizes, m_counts, rng, warn: bool = True) -> np.ndarray:
    # `warn` is False for target_metric probes: a probe's labeling is not the one
    # delivered, so its [CLM-305] could report an entry that the delivered alpha
    # does apply.
    demand, remaining_capacity = allocate(cfg, rules, m_counts, cluster_sizes)

    # Optional structured competing-label noise (see _competing_demand).
    # With no 'competing_noise' key this block is skipped and the pipeline
    # is identical to the base engine.
    favors_overrides: dict[int, dict[int, str]] = {}
    if cfg.get("competing_noise"):
        extra, favors_overrides = _competing_demand(cfg, remaining_capacity,
                                                     cfg["num_classes"], warn=warn)
        for k, per_label in extra.items():
            for label, count in per_label.items():
                demand[k][label] = demand[k].get(label, 0) + count
                remaining_capacity[k] -= count

    used_per_label = np.zeros(cfg["num_classes"], dtype=int)
    for k in demand:
        for label, count in demand[k].items():
            used_per_label[label] += count

    n_spillover_total = sum(remaining_capacity.values())
    spillover_pool = _spillover_draws(cfg, m_counts, used_per_label, n_spillover_total, rng)

    out = np.full(len(cluster_labels), -1, dtype=int)
    cursor = 0
    for k in cluster_ids:
        idx = np.where(cluster_labels == k)[0]
        n_here = remaining_capacity[k]
        this_spill = spillover_pool[cursor: cursor + n_here]
        cursor += n_here
        out[idx] = assign_points_in_cluster(
            coords[idx], demand[k], this_spill,
            # `or`, not a .get default: a bare `centroid_dependence:` key parses to
            # None, and assign_points_in_cluster calls .get() on whatever it is given.
            cfg.get("centroid_dependence") or {"enabled": False}, rng,
            favors_overrides=favors_overrides.get(k),
        )
    return out


# ---------------------------------------------------------------------------
# Global target-metric solving
# ---------------------------------------------------------------------------

_METRIC_FUNCS = {"mcc": clustering_mcc, "ari": clustering_ari}
# The highest value each target metric can take given only the cluster and label
# counts. For mcc some labeling with those counts reaches it, though the rules of
# a config may not; for ari it is a bound that no labeling may reach.
_SIZE_CEILINGS = {"mcc": MetricCeilings.rk_ceiling, "ari": MetricCeilings.ari_bound}


def _count_movers(cfg: dict) -> list[str]:
    """The settings that can deliver label counts other than their targets.

    Empty means every labeling of this config has exactly the target counts,
    whatever the recall level: proportional_to_marginal refills each label to its
    target and nothing else claims points. [CLM-304] and the size ceiling both
    depend on it.
    """
    spill = cfg.get("spillover_rule", "proportional_to_marginal")
    causes = [f"spillover_rule {spill!r}"] if spill in ("uniform", "concentrated") else []
    if cfg.get("competing_noise"):
        causes.append("competing_noise")
    return causes


def _ceiling_note(metric: str, ceiling: float | None, target: float, tol: float) -> str:
    """The size ceiling as [CLM-306] words it, and for mcc the floor of 0: the
    matching maximises R_K, and R_K averages 0 over all matchings, so no labeling
    of any counts scores below 0. Empty when neither applies."""
    note = ""
    if ceiling is not None:
        name = "ceiling" if metric == "mcc" else "bound"
        note = f" Size {name} for these label counts: {metric}={ceiling:.4f}"
        if target - tol > ceiling:
            note += "; the target is above it, so no labeling with these counts reaches it"
        note += "."
    if metric == "mcc" and target + tol < 0:
        note += " mcc is never below 0, so no labeling reaches the target."
    return note


class _ProbeStop(Exception):
    """Ends the root-find early; metric None means infeasible."""

    def __init__(self, alpha: float, metric: float | None):
        super().__init__(alpha, metric)
        self.alpha = alpha
        self.metric = metric


def _generate_for_alpha(cluster_labels, coords, cfg, alpha, rng, cluster_ids,
                         cluster_sizes, m_counts) -> np.ndarray | InfeasibleAllocationError:
    """One probe evaluation. Returns, rather than raises, the error of an
    infeasible alpha, so the search can report which [CLM-15x] blocked it."""
    try:
        rules = build_rules(cfg, cluster_ids, recall_target_override=alpha)
        return _run_allocation_pipeline(cluster_labels, coords, cfg, rules,
                                         cluster_ids, cluster_sizes, m_counts, rng,
                                         warn=False)
    except InfeasibleAllocationError as exc:
        return exc


def _log_blocked_levels(blocked: dict[float, InfeasibleAllocationError]) -> None:
    """One warning per distinct [CLM-15x] code among the infeasible probes,
    ordered by the lowest alpha it blocked, quoting the error at that alpha."""
    by_code: dict[int | None, list[float]] = {}
    for alpha in sorted(blocked):
        by_code.setdefault(getattr(blocked[alpha], "code", None), []).append(alpha)
    for alphas in by_code.values():
        log.warning(f"target_metric: {len(alphas)} probed recall level(s) in "
                    f"[{alphas[0]:.3f}, {alphas[-1]:.3f}] are infeasible; "
                    f"at alpha={alphas[0]:.3f}: {blocked[alphas[0]]}")


def solve_alpha_for_target_metric(cluster_labels, coords, cfg, cluster_ids,
                                   cluster_sizes, m_counts, seed: int = 0,
                                   report: dict[str, float] | None = None,
                                   ceiling: float | None = None) -> float:
    """
    Finds alpha in [0, 1] (substituted as every rule's recall_target) such
    that the achieved global metric (clustering_mcc/clustering_ari between
    `cluster_labels` and the generated label array) is as close as possible
    to cfg['target_metric']['value'].

    No closed form: the achieved global metric depends on every rule's
    outcome jointly (unlike the single-pair solve of scope='pair', which
    inverts exactly in _pair_label_counts). The solution runs a
    coarse grid scan first, then a Brent root-find within the bracket it
    finds, all probe evaluations share a fixed seed (common random numbers) so
    differences across candidates come from alpha, not randomization noise.

    The feasible alphas are not assumed to form one interval. alpha = 0
    claims no point, so without competing_noise it is always feasible; the
    [CLM-150] and [CLM-153] claims only grow with alpha; [CLM-151] splits
    budgets by largest remainder, which is not shown to be monotone; and
    competing_noise converts a share of the points the rules leave
    unclaimed, which shrink as alpha grows, so [CLM-152] can block low
    alphas while high alphas fill the cluster. The achieved metric is not
    monotone either, hence the grid scan rather than a blind bisection.

    Every infeasible probe keeps its error, so [CLM-120] (no grid alpha is
    feasible) and [CLM-306] (the target was missed) are preceded by one
    warning per [CLM-15x] code, naming the levels it blocked.

    `report`, when given, receives under "metric" the value the search
    scored at the returned alpha; generate_clm_labels compares it with the
    delivered labeling for [CLM-309]. Under "highest" and "highest_alpha" it
    receives the highest value any probe scored and the lowest alpha that scored
    it, read off the probes already run. A grid can step over a higher value, so
    it is what the search found, not the most the configuration can reach.

    `ceiling` is the size ceiling of these label counts, or None when the counts
    can move. [CLM-306] names it, and for mcc a target below 0; the search acts
    on neither.
    """
    tm = cfg["target_metric"]
    metric_fn = _METRIC_FUNCS[tm["type"]]
    target = tm["value"]
    tol = tm.get("tolerance", 0.01)
    max_iter = tm.get("max_iter", 40)
    # Defaults to the RUN's seed, which is the stream the delivered labeling is
    # generated on. Every probe still shares one stream, so candidates remain
    # comparable (common random numbers); the difference is that the winning
    # probe is now reproduced exactly by the final generation instead of being
    # re-rolled on an unrelated stream. That divergence is the only thing
    # [CLM-309] reports, and this removes it rather than measuring it. Setting an
    # explicit probe_seed different from the run seed re-introduces it, deliberately.
    # Empty (`probe_seed:`) is the run seed too: as None it seeded every probe from
    # fresh OS entropy, so the probes were not comparable and no run repeated.
    probe_seed = tm.get("probe_seed")
    if probe_seed is None:
        probe_seed = seed

    # A probe is a full allocation run, so never pay for one alpha twice.
    probed: dict[float, float | None] = {}
    blocked: dict[float, InfeasibleAllocationError] = {}

    def achieved(alpha: float) -> float | None:
        alpha = float(alpha)
        if alpha not in probed:
            rng = np.random.default_rng(probe_seed)
            labels = _generate_for_alpha(cluster_labels, coords, cfg, alpha, rng,
                                          cluster_ids, cluster_sizes, m_counts)
            if isinstance(labels, InfeasibleAllocationError):
                blocked[alpha] = labels
                probed[alpha] = None
            else:
                probed[alpha] = metric_fn(cluster_labels, labels)
        return probed[alpha]

    def returned(alpha: float, metric: float) -> float:
        if report is not None:
            report["metric"] = metric
            scored = {a: m for a, m in probed.items() if m is not None}
            top = min(scored, key=lambda a: (-scored[a], a))
            report["highest"], report["highest_alpha"] = scored[top], top
        return alpha

    grid = np.linspace(0.0, 1.0, 11)
    scored = [(float(a), achieved(a)) for a in grid]
    feasible = sorted((a, m) for a, m in scored if m is not None)

    if not feasible:
        _log_blocked_levels(blocked)
        raise clm_error(120)

    best_alpha, best_metric = min(feasible, key=lambda am: abs(am[1] - target))
    if abs(best_metric - target) <= tol:
        log.info("target_metric: grid search hit tolerance directly "
                 f"(alpha={best_alpha:.3f}, achieved={best_metric:.3f}, target={target:.3f}).")
        return returned(best_alpha, best_metric)

    # brentq needs the target straddled; the grid is what finds that.
    bracket = None
    for (a1, m1), (a2, m2) in itertools.pairwise(feasible):
        if (m1 - target) * (m2 - target) <= 0:
            bracket = (a1, a2)
            break

    if max_iter > 0:
        best = [best_alpha, best_metric]

        def residual(alpha: float) -> float:
            """Signed distance to the target, stopping once inside tolerance."""
            m = achieved(alpha)
            if m is None:
                # Feasibility is not monotone in alpha, so the bracket or the
                # bounded span can hold an infeasible level; the best feasible
                # probe so far then stands as a miss.
                raise _ProbeStop(float(alpha), None)
            if abs(m - target) < abs(best[1] - target):
                best[0], best[1] = float(alpha), m
            if abs(m - target) <= tol:
                raise _ProbeStop(float(alpha), m)
            return float(m - target)

        try:
            if bracket is not None:
                brentq(residual, bracket[0], bracket[1], xtol=1e-12, maxiter=max_iter)
            else:
                # No sign change, so no root: minimize the distance instead.
                minimize_scalar(lambda a: abs(residual(a)), method="bounded",
                                bounds=(feasible[0][0], feasible[-1][0]),
                                options={"xatol": 1e-6, "maxiter": max_iter})
        except _ProbeStop as stop:
            if stop.metric is not None:
                log.info(f"target_metric: converged in {len(probed) - len(grid)} probe(s) "
                         f"(alpha={stop.alpha:.4f}, achieved={stop.metric:.4f}, "
                         f"target={target:.3f}).")
                return returned(stop.alpha, stop.metric)
        except (RuntimeError, ValueError):
            # Budget spent, or a bracket the solver would not accept. Either way
            # the grid's best still stands and [CLM-306] below reports it.
            pass
        best_alpha, best_metric = best

    _log_blocked_levels(blocked)
    clm_warn(log, 306, max_iter=max_iter, best_alpha=best_alpha,
             best_metric=best_metric, target=target, tol=tol,
             ceiling=_ceiling_note(tm["type"], ceiling, target, tol))
    return returned(best_alpha, best_metric)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_clm_labels(cluster_labels: np.ndarray, coords: np.ndarray | None, cfg: dict,
                        seed: int = 42, report: dict[str, Any] | None = None) -> pd.Series:
    """Generates one synthetic label column for an existing clustering.

    Parameters
    ----------
    cluster_labels : numpy.ndarray of shape (n_samples,)
        The existing clustering to match against. Ids may be integers or strings
        and need not be contiguous; ``K`` is their number of distinct values.
    coords : numpy.ndarray of shape (n_samples, n_features)
        Feature vectors, used to derive cluster centroids for spatial placement.
        May be ``None`` or empty only for labels-only configs; any spatial
        placement (``centroid_dependence``, or ``competing_noise`` favoring
        core/boundary) requires real vectors and raises ``[CLM-125]`` without them.
    cfg : dict
        The ``clm_label`` block, a mapping (anything else is ``[CLM-141]``).
        ``num_classes`` and ``matching_mode`` are always required; each mode then
        requires its own keys, and ``target_metric`` turns the requested recall
        into a solved-for quantity.
    seed : int, default 42
        Seeds one generator used by both the target-metric search and the final
        generation, so a solved value is the value delivered.
    report : dict, optional
        Filled under ``target_metric`` and left untouched otherwise. Keys
        ``scope``, ``type``, ``requested`` and ``achieved``, the value of the
        delivered labeling. ``scope: global`` adds ``alpha``, the solved recall
        level; ``highest`` and ``highest_alpha``, the highest value a search probe
        scored and where; and ``ceiling``, the size ceiling: no labeling with
        these cluster and label counts scores higher. For ``ari`` it is a bound
        that is not always reached. It is ``None`` when ``uniform`` or
        ``concentrated`` spillover or ``competing_noise`` can move the label
        counts. Reported only: a target above it is still searched.

    Returns
    -------
    pandas.Series
        Named ``clm_label``, one entry per input point, values in
        ``0..num_classes - 1``.

    Raises
    ------
    ValueError
        Carrying a ``[CLM-1xx]`` code for an invalid or incompatible configuration.
    MissingConfigKey
        Carrying a ``[CLM-2xx]`` code when a required key is absent.
    InfeasibleAllocationError
        Carrying a ``[CLM-15x]`` code when the configuration is valid but the
        requested counts do not fit the cluster capacities.

    Notes
    -----
    ``num_classes`` and the number of clusters are both capped at
    ``MAX_CARDINALITY``. Diagnostic codes are a public contract: they are never
    renumbered or reused, and the full catalogue is in the troubleshooting manual.

    Examples
    --------
    >>> import numpy as np
    >>> from clmsynth import generate_clm_labels
    >>> clusters = np.repeat([0, 1], 50)
    >>> coords = np.zeros((100, 2))
    >>> cfg = {"num_classes": 2, "matching_mode": "perfect"}
    >>> generate_clm_labels(clusters, coords, cfg, seed=0).value_counts().to_dict()
    {0: 50, 1: 50}
    """
    # None, a number or a bool crashed on `in`; a list or a string passed it and
    # was reported as missing num_classes ([CLM-201]).
    _check_mapping(cfg, "clm_label", "{num_classes: 4, matching_mode: perfect}")
    rng = np.random.default_rng(seed)
    N = len(cluster_labels)
    if "num_classes" not in cfg:
        raise clm_missing(201)
    M = cfg["num_classes"]
    # bool is an int subclass: `num_classes: yes` would otherwise pass as 1.
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)):
        raise clm_error(126, M=repr(M), max_val=MAX_CARDINALITY)
    if not (1 <= M <= MAX_CARDINALITY):
        raise clm_error(126, M=M, max_val=MAX_CARDINALITY)

    # The mode decides which of the other keys are read at all, so it is checked
    # before any of them: an error in a key the mode may never read must not stand
    # in for a missing or misspelled mode.
    # Written with no value (`matching_mode:`) is left out too, not an unknown mode.
    if cfg.get("matching_mode") is None:
        raise clm_missing(202)
    if cfg["matching_mode"] not in MATCHING_MODES:
        raise clm_error(101, mode=cfg["matching_mode"])

    _validate_dataset_input(cluster_labels, coords, N)   # [CLM-133]
    coords = _ensure_coords(cfg, coords, N)   # [CLM-125] placement needs geometry
    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    cluster_sizes = {k: int(np.sum(cluster_labels == k)) for k in cluster_ids}
    K = len(cluster_ids)
    if K > MAX_CARDINALITY:
        raise clm_error(127, K=K, max_val=MAX_CARDINALITY)
    if cfg["matching_mode"] == "perfect" and M != K:
        raise clm_error(102, M=M, K=K)

    if cfg["matching_mode"] == "perfect":
        # Label i is cluster i, so the counts are the cluster sizes. balance,
        # proportions, skew_rule and skew_params are never read ([CLM-302]), so they
        # are neither resolved nor validated.
        m_counts = np.array([cluster_sizes[cluster_ids[i]] for i in range(M)])
    else:
        # Both guards must precede resolve_label_counts, which consumes these keys
        # without objecting: it reads any balance but 'balanced' as 'unbalanced'
        # ([CLM-132]), and turns an out-of-range skew_params into negative label
        # counts that still sum to N ([CLM-131]).
        _validate_balance_cfg(cfg)
        _validate_skew_cfg(cfg)
        m_counts = resolve_label_counts(cfg, N, rng)

    if cfg["matching_mode"] == "random":
        # Validity (not membership): a null/empty target_metric is "not set"
        if cfg.get("target_metric"):
            raise clm_error(114)
        if cfg.get("competing_noise"):
            raise clm_error(115)
        draws = np.repeat(np.arange(cfg["num_classes"]), m_counts)
        rng.shuffle(draws)
        return pd.Series(draws, name="clm_label")

    # These run before the target-metric solver: a config error must surface as
    # itself, not as a solved score computed over an invalid labeling.
    _validate_spillover_cfg(cfg)
    _validate_centroid_cfg(cfg)
    _validate_target_metric_cfg(cfg, cluster_ids)
    _validate_split_cfg(cfg)
    _validate_competing_noise_cfg(cfg)
    movers = _count_movers(cfg)

    # Resolve once and gate every branch on validity. `"target_metric" in cfg`
    # would be True for a null/empty value (`target_metric:` in YAML), then crash
    # at `.get(...)` on None; validation above already treats that as unset.
    tm = cfg.get("target_metric")
    alpha = None  # solved recall level; stays None unless target_metric is set
    searched: dict[str, float] = {}  # the global search's score at that alpha
    ceiling: float | None = None  # size ceiling of the label counts, when they are fixed
    if tm:
        # Well-formed rows only: under custom, build_rules reports a valueless
        # matrix ([CLM-206]) and a malformed one ([CLM-141]); the other modes
        # never read it.
        matrix = cfg.get("assignment_matrix")
        if any(isinstance(row, dict) and "recall_target" in row
               for row in (matrix if isinstance(matrix, (list, tuple)) else [])):
            clm_warn(log, 303)
        if tm.get("scope", "global") == "pair":
            # Exact single-pair MCC: size the target label so all of it
            # fits its cluster (recall 1), no numerical search.
            m_counts = _pair_label_counts(cfg, cluster_sizes, m_counts, N)
            alpha = 1.0
        else:
            # The label counts are settled before the search. When nothing can move
            # them, every labeling the search can deliver has these counts, so their
            # size ceiling holds at every recall level. Reported, not acted on.
            if not movers:
                ceiling = _SIZE_CEILINGS[tm["type"]](list(cluster_sizes.values()), m_counts)
            alpha = solve_alpha_for_target_metric(cluster_labels, coords, cfg, cluster_ids,
                                                   cluster_sizes, m_counts, seed=seed,
                                                   report=searched, ceiling=ceiling)
        rules = build_rules(cfg, cluster_ids, recall_target_override=alpha)
    else:
        rules = build_rules(cfg, cluster_ids)

    # Allocation and placement draw from their OWN stream, started fresh from the
    # same seed rather than continuing the one above. Two reasons, one of them a
    # defect: resolve_label_counts consumes draws for a dirichlet skew and not
    # for any other rule, so a shared stream reaches allocation in a state that
    # depends on which skew rule was chosen.
    alloc_rng = np.random.default_rng(seed)
    out = _run_allocation_pipeline(cluster_labels, coords, cfg, rules, cluster_ids,
                                    cluster_sizes, m_counts, alloc_rng)

    achieved_counts = np.bincount(out, minlength=cfg["num_classes"])
    log.info(f"CLM labels generated. Target counts: {m_counts.tolist()}, achieved: {achieved_counts.tolist()}.")

    # [CLM-304]: checked against the delivered counts, not predicted from the config.
    # proportional_to_marginal refills every label to its target, so competing_noise
    # moves the counts only when it gives a label more points than its target.
    if movers and not np.array_equal(achieved_counts, m_counts):
        clm_warn(log, 304, cause=" with ".join(movers), achieved=achieved_counts.tolist(),
                 target=m_counts.tolist())

    if tm:
        if tm.get("scope", "global") == "pair":
            sm = cfg["single_match"]
            # Only mcc reaches here ([CLM-123]); the ari branch is kept, unused.
            pair_fn = clustering_mcc_pair if tm["type"] == "mcc" else clustering_ari_pair
            achieved = pair_fn(cluster_labels, out, sm["cluster"], sm["label"])
            log.info(f"target_metric: final achieved pair {tm['type']}={achieved:.4f} "
                     f"(target was {tm['value']}, pair cluster={sm['cluster']}/label={sm['label']}).")

            # The same 'tolerance' key the global solver reads. Only 'max_iter' stays global-only:
            # the pair scope inverts in closed form and never iterates.
            pair_tol = tm.get("tolerance", 0.01)
            if abs(achieved - tm["value"]) > pair_tol:
                lstar_mask = (out == sm["label"])
                total = int(np.sum(lstar_mask))
                outside = int(np.sum(lstar_mask & (np.asarray(cluster_labels) != sm["cluster"])))
                clm_warn(log, 310, type=tm["type"], achieved=achieved, target=tm["value"],
                         tol=pair_tol, label=sm["label"], cluster=sm["cluster"],
                         outside=outside, total=total)
            if report is not None:
                report.update(scope="pair", type=tm["type"], requested=tm["value"],
                              achieved=achieved)
        else:
            final_metric = _METRIC_FUNCS[tm["type"]](cluster_labels, out)
            log.info(f"target_metric: final achieved {tm['type']}={final_metric:.4f} "
                     f"(target was {tm['value']}, alpha={alpha:.4f}).")
            if ceiling is None:
                bound = f"no size ceiling, since {' and '.join(movers)} can move the label counts"
            elif tm["type"] == "mcc":
                bound = (f"size ceiling {ceiling:.4f}, which no labeling with these label "
                         "counts exceeds")
            else:
                bound = (f"size bound {ceiling:.4f}, which no labeling with these label counts "
                         "exceeds (it is not always reached)")
            log.info(f"target_metric: highest {tm['type']} the search found is "
                     f"{searched['highest']:.4f} (alpha={searched['highest_alpha']:.4f}); {bound}.")
            if report is not None:
                report.update(scope="global", type=tm["type"], requested=tm["value"],
                              achieved=final_metric, alpha=alpha, highest=searched["highest"],
                              highest_alpha=searched["highest_alpha"], ceiling=ceiling)

            # A search that missed has already said so ([CLM-306]). [CLM-309] is for
            # a delivered labeling the search never scored, which takes a probe_seed
            # other than the run seed: on one stream the two values are identical.
            tol = tm.get("tolerance", 0.01)
            if (abs(final_metric - tm["value"]) > tol
                    and final_metric != searched["metric"]):
                clm_warn(log, 309, type=tm["type"], achieved=final_metric,
                         searched=searched["metric"], target=tm["value"], tol=tol,
                         alpha=alpha)

    return pd.Series(out, name="clm_label")
