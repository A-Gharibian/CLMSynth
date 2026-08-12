# clm_label_engine.py
"""
Assigns synthetic labels L(x) to each point x given its existing
ground-truth cluster c(x). Clusters and cluster membership are fixed
inputs from upstream, this module only decides which label each point
gets, per the clm_label config schema.
"""

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clm_errors import InfeasibleAllocationError, clm_error, clm_infeasible, clm_warn
from .metrics import clustering_ari, clustering_mcc, clustering_mcc_pair

log = logging.getLogger(__name__)

MAX_CARDINALITY = 64


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
    precedence, and `skew_rule` is the fallback when no proportions are given."""
    M = cfg["num_classes"]
    balance = cfg.get("balance", "balanced")

    if balance == "balanced":
        if cfg.get("proportions"):
            clm_warn(log, 301)
        proportions = [1.0 / M] * M
    elif cfg.get("proportions"):
        proportions = cfg["proportions"]
        # [CLM-121] Length must match M exactly. A longer list used to enlarge the
        # label space silently: m_counts is sized from `proportions`, and
        # _spillover_draws derived M from len(m_counts), so uniform/concentrated
        # spillover emitted label ids >= num_classes into the written dataset.
        # (proportional_to_marginal happened to fail with a numpy broadcasting
        # error instead, which is why this went unnoticed under the default rule.)
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
    try:                                   # accept Python/NumPy ints; reject
        is_int = (label == int(label))     # floats like 2.5, strings, None
    except (TypeError, ValueError):
        is_int = False
    if not is_int or not (0 <= label < M):
        raise clm_error(104, where=where, label=label, hi=M - 1)
    unknown = [k for k in clusters if k not in cluster_ids]
    if unknown:
        raise clm_error(105, where=where, unknown=unknown,
                        available=sorted(cluster_ids, key=str))


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
        clm_warn(log, 302)
        return [Rule(label=i, clusters=[cluster_ids[i]], recall_target=1.0) for i in range(K)]

    if mode == "single":
        if M < 2 or K < 2:
            raise clm_error(103, M=M, K=K)
        sm = cfg["single_match"]
        _check_pair(sm["label"], [sm["cluster"]], M, cluster_ids, "single_match")
        rt = recall_target_override if recall_target_override is not None else 1.0
        return [Rule(label=sm["label"], clusters=[sm["cluster"]], recall_target=rt)]

    if mode == "custom":
        rules = []
        for i, row in enumerate(cfg["assignment_matrix"]):
            _check_pair(row["label"], row["clusters"], M, cluster_ids,
                        f"assignment_matrix row {i}")
            rt = recall_target_override if recall_target_override is not None else row["recall_target"]
            rules.append(Rule(label=row["label"], clusters=row["clusters"], recall_target=rt))
        return rules

    raise clm_error(101, mode=mode)


def validate_matching_ids(cfg: dict, cluster_ids: list) -> None:
    """Run only the [CLM-104]/[CLM-105] id checks against ONE dataset's cluster ids.

    Deliberately tolerant of a malformed config: anything other than an id
    problem is left for the engine's own validation to report in its usual place.
    Split out of build_rules so the pipeline can apply them ahead of time, per
    dataset, without building rules or resolving recall targets. Unlike every
    other [CLM-1xx] code, 104 and 105 are statements about a *dataset* rather
    than about the configuration, under `byoc` each CSV brings its own cluster
    ids and nothing requires them to agree, so the pipeline checks every
    dataset up front and refuses the batch as a whole rather than discovering the
    mismatch mid-run.
    """
    M = cfg.get("num_classes")
    if not isinstance(M, int) or isinstance(M, bool):
        return

    mode = cfg.get("matching_mode")
    if mode == "single":
        sm = cfg.get("single_match") or {}
        if "label" in sm and "cluster" in sm:
            _check_pair(sm["label"], [sm["cluster"]], M, cluster_ids, "single_match")
    elif mode == "custom":
        for i, row in enumerate(cfg.get("assignment_matrix") or []):
            if isinstance(row, dict) and "label" in row and "clusters" in row:
                _check_pair(row["label"], row["clusters"], M, cluster_ids,
                            f"assignment_matrix row {i}")


def _ensure_coords(cfg: dict, coords, N: int) -> np.ndarray:
    """[CLM-125] guard: spatial placement needs real per-point geometry.
    Placement is requested by `centroid_dependence.enabled` or by any
    competing_noise entry whose favors is 'core'/'boundary' (the default is
    'boundary').
    """
    cd = cfg.get("centroid_dependence") or {}
    spatial_noise = [e for e in (cfg.get("competing_noise") or [])
                     if e.get("favors", "boundary") != "random"]
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
    # is a no-op; the caller gates the whole block on the same truthiness so the two
    # can never disagree.
    tm = cfg.get("target_metric")
    if not tm:
        return
    mode = cfg["matching_mode"]
    if mode not in ("single", "custom"):
        raise clm_error(111, mode=mode)
    if tm.get("type") not in ("mcc", "ari"):
        raise clm_error(112, type=tm.get("type"))
    if not (-1.0 <= tm.get("value", 0) <= 1.0):
        raise clm_error(113, value=tm.get("value"))
    scope = tm.get("scope", "global")
    if scope not in ("pair", "global"):
        raise clm_error(122, scope=scope)
    if scope == "pair":
        # The single-pair MCC inverts in closed form (see _pair_label_counts);
        # ARI has no such inverse, and the pair is taken from single_match.
        if tm["type"] != "mcc":
            raise clm_error(123)
        if mode != "single":
            raise clm_error(124)
        # Validate the (cluster, label) pair up front: _pair_label_counts indexes
        # cluster_sizes[cluster] and m_counts[label] BEFORE build_rules' own
        # _check_pair would run, so an unknown cluster / out-of-range label must
        # surface here as [CLM-105]/[CLM-104], not a raw KeyError/IndexError.
        sm = cfg["single_match"]
        _check_pair(sm["label"], [sm["cluster"]], cfg["num_classes"], cluster_ids, "single_match")
        # [CLM-130] The closed form is only exact while EVERY point of l* stays
        # inside k*: _pair_label_counts sizes l* for that assumption and there is
        # no search to correct a miss. Any setting that can emit l* elsewhere
        # silently delivers a different coefficient, so the combination is rejected.
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
            try:                    # accept 2 and 2.0, reject 1.5, True, "2", None
                is_int = not isinstance(v, bool) and v == int(v)
            except (TypeError, ValueError):
                is_int = False
            if not is_int or not (0 <= v < M):
                valid = False
                break
    if not valid:
        raise clm_error(128, hi=M - 1, given=given)


def _is_real(value) -> bool:
    """True for a real number, excluding bool (which is an int subclass and would
    otherwise sail through every range check as 0/1)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _validate_skew_cfg(cfg: dict) -> None:
    """[CLM-131] guard: skew parameters must be in range for the chosen skew_rule.

    Called from generate_clm_labels *before* resolve_label_counts, because that
    is where the parameters are consumed, a guard placed alongside the other
    validators would run after the counts it protects had already been computed.

    Only the parameters that will actually be read are checked, using the same
    predicate resolve_label_counts branches on, so a config that supplies explicit
    proportions (or asks for a balanced split) is not failed for a stale
    skew_params block it never consults.

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
        if not _is_real(ratio) or ratio < 0:
            problem = (f"geometric 'ratio' must be a number >= 0, got {ratio!r}; a "
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
        if not _is_real(alpha) or alpha <= 0:
            problem = (f"dirichlet 'alpha' must be a number > 0, got {alpha!r}; at "
                       "exactly 0 every draw is 0 and normalising them divides by zero")

    if problem is not None:
        raise clm_error(131, problem=problem, hi=M - 1)


def _validate_centroid_cfg(cfg: dict) -> None:
    """[CLM-129] guard: centroid_dependence.favors must be exactly 'core' or 'boundary'.
    """
    cd = cfg.get("centroid_dependence") or {}
    if not cd.get("enabled"):
        return                      # not read unless placement is switched on
    favors = cd.get("favors", "core")
    if favors not in ("core", "boundary"):
        raise clm_error(129, favors=favors)


def _pair_label_counts(cfg: dict, cluster_sizes: dict[int, int],
                       m_counts: np.ndarray, N: int) -> np.ndarray:
    """Exact single-pair MCC target via the single-dominant construction
    (target_metric.scope='pair', single mode).

    The target label l* is sized so that placing *all* of it inside its cluster
    k* (recall 1, so no leftover spills back in) makes the 2x2 Matthews phi of
    the (k*, l*) pair equal the requested value. Inverting
        phi = sqrt( m_c (N - n_k) / (n_k (N - m_c)) )
    for the label size gives, with n_k = |k*|,
        m_c* = phi^2 n_k N / (N - n_k (1 - phi^2)).
    Returns a copy of `m_counts` with l*'s count set to m_c* and the remaining
    labels rescaled to fill N - m_c*. No numerical search; the achieved phi
    equals the target to within one point of integer rounding.
    """
    tm = cfg["target_metric"]
    sm = cfg["single_match"]
    lstar, n_k = sm["label"], cluster_sizes[sm["cluster"]]
    phi = tm["value"]

    # Reachable range of the single-dominant construction: placing m_c in [1, n_k]
    # of label l* entirely inside k* yields a pair phi in [phi_min, 1.0], phi_min
    # at m_c = 1 (phi_max = 1 at m_c = n_k, an exact 1:1 pair). A request outside
    # the range is clamped to the nearest reachable value with [CLM-307]; the
    # subset construction can never produce a negative phi.
    phi_max = 1.0
    phi_min = float(np.sqrt((N - n_k) / (n_k * (N - 1)))) if (n_k >= 1 and N > 1) else 0.0
    if not (phi_min <= phi <= phi_max):
        clm_warn(log, 307, target=phi, phi_min=phi_min, phi_max=phi_max)
        phi = min(max(phi, phi_min), phi_max)

    phi2 = phi * phi
    denom = N - n_k * (1.0 - phi2)
    m_star = n_k if (phi >= 1.0 or denom <= 0) else int(round(phi2 * n_k * N / denom))
    m_star = min(max(m_star, 1), n_k)                # a present label, capped at the cluster
    clm_warn(log, 308, label=lstar, m=m_star)

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
            raise clm_infeasible(150, label=rule.label, rt=rule.recall_target, tp=tp_row,
                                 m=m_label, clusters=rule.clusters, capacity=capacity,
                                 max_recall=capacity / m_label)

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
        return rng.integers(0, M, size=n_spillover).tolist()
    if rule_name == "concentrated":
        targets = cfg.get("concentrated_labels", [int(np.argmax(m_counts))])
        return rng.choice(targets, size=n_spillover).tolist()
    raise clm_error(109, rule_name=rule_name)


# ---------------------------------------------------------------------------
# Spatial placement (centroid_dependence)
# ---------------------------------------------------------------------------

def _weighted_pick(idx_avail: np.ndarray, count: int, coreness_avail: np.ndarray,
                    profile: str, steepness: float, rng: np.random.Generator) -> np.ndarray:
    """Selects `count` of the available point indices by the configured
    profile, given each candidate's coreness score. Extracted verbatim from
    the placement loop so competing_noise overrides can reuse the exact same
    machinery with a per-label coreness sign."""
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

    Rule-claimed labels are placed first (centroid-weighted when
    `centroid_dependence` is enabled, per-label `favors_overrides` from
    competing_noise taking precedence), then the leftover rows consume the
    spillover draws. Returns the cluster's label column.
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
        # a zeros array in the else-branch: a placeholder that looks like real
        # geometry is exactly the silent-wrong-output shape this engine refuses
        # elsewhere ([CLM-125] exists for the same reason).
        assert distances is not None
        coreness = -distances if favors == "core" else distances
    else:
        coreness = np.zeros(n)

    ordered = sorted(label_targets.items(),
                     key=lambda kv: (((favors_overrides or {}).get(kv[0]) is None), -kv[1]))
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
            # truthy, so distances was computed above.
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

def _competing_demand(cfg: dict, remaining_capacity: dict[int, int], M: int):
    """
    OPTIONAL feature, active only when the config carries a 'competing_noise'
    list; with the key absent this function is never called and the engine
    behaves exactly as before it existed (to disable the feature entirely,
    remove its single call site in _run_allocation_pipeline).

    Each entry converts `share` of ONE cluster's UNCLAIMED points (leftover
    after all rules were allocated) into one specific competing label, placed
    core/boundary/random within that cluster:

        competing_noise:
          - {cluster: 1, label: 2, share: 1.0, favors: boundary}
    """
    extra: dict[int, dict[int, int]] = {}
    overrides: dict[int, dict[int, str]] = {}

    for entry in cfg["competing_noise"]:
        k, label = entry["cluster"], entry["label"]
        share = entry.get("share", 1.0)
        favors = entry.get("favors", "boundary")

        if k not in remaining_capacity:
            raise clm_error(119, cluster=k, available=sorted(remaining_capacity, key=str))
        if not (0 <= label < M):
            raise clm_error(118, label=label, hi=M - 1)
        if not (0.0 <= share <= 1.0):
            raise clm_error(117, share=share)
        if favors not in ("core", "boundary", "random"):
            raise clm_error(116, favors=favors)

        count = int(round(share * remaining_capacity[k]))
        if count == 0:
            clm_warn(log, 305, k=k, entry=entry)
            continue
        extra.setdefault(k, {})
        extra[k][label] = extra[k].get(label, 0) + count
        overrides.setdefault(k, {})[label] = favors

    for k, per_label in extra.items():
        claimed = sum(per_label.values())
        if claimed > remaining_capacity[k]:
            raise clm_infeasible(152, k=k, claimed=claimed, remaining=remaining_capacity[k])

    if extra:
        clm_warn(log, 304)
    return extra, overrides


# ---------------------------------------------------------------------------
# Shared allocation pipeline (used both by the probe search and the final commit)
# ---------------------------------------------------------------------------

def _run_allocation_pipeline(cluster_labels, coords, cfg, rules, cluster_ids,
                              cluster_sizes, m_counts, rng) -> np.ndarray:
    demand, remaining_capacity = allocate(cfg, rules, m_counts, cluster_sizes)

    # Optional structured competing-label noise (see _competing_demand).
    # With no 'competing_noise' key this block is skipped and the pipeline
    # is identical to the base engine.
    favors_overrides: dict[int, dict[int, str]] = {}
    if cfg.get("competing_noise"):
        extra, favors_overrides = _competing_demand(cfg, remaining_capacity,
                                                     cfg["num_classes"])
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


def _generate_for_alpha(cluster_labels, coords, cfg, alpha, rng,
                         cluster_ids, cluster_sizes, m_counts) -> np.ndarray | None:
    """One probe evaluation. Returns None if this alpha is infeasible."""
    try:
        rules = build_rules(cfg, cluster_ids, recall_target_override=alpha)
        return _run_allocation_pipeline(cluster_labels, coords, cfg, rules,
                                         cluster_ids, cluster_sizes, m_counts, rng)
    except InfeasibleAllocationError:
        return None


def solve_alpha_for_target_metric(cluster_labels, coords, cfg, cluster_ids,
                                   cluster_sizes, m_counts, seed: int = 0) -> float:
    """
    Finds alpha in [0, 1] (substituted as every rule's recall_target) such
    that the achieved global metric (clustering_mcc/clustering_ari between
    `cluster_labels` and the generated label array) is as close as possible
    to cfg['target_metric']['value'].

    No closed form: the achieved global metric depends on every rule's
    outcome jointly (unlike the single-pair solve of scope='pair', which
    inverts exactly in _pair_label_counts). This runs a
    coarse grid scan first, both to bracket the target and to guard
    against non-strict monotonicity, then bisects within the bracket.
    All probe evaluations share a fixed seed (common random numbers) so
    differences across candidates come from alpha, not randomization noise.

    Feasibility is monotonic in alpha by construction: tp_row = round(alpha
    * m_label) only grows with alpha, so once a rule's claim exceeds its
    clusters' capacity, it stays exceeded for every higher alpha. That
    guarantee does NOT extend to the achieved metric itself, hence the grid
    scan rather than a blind bisection.
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
    # re-rolled on an unrelated stream. That divergence is what [CLM-309]
    # reports, and this removes it rather than measuring it. Setting an explicit
    # probe_seed different from the run seed re-introduces it, deliberately.
    probe_seed = tm.get("probe_seed", seed)

    def achieved(alpha: float) -> float | None:
        rng = np.random.default_rng(probe_seed)
        labels = _generate_for_alpha(cluster_labels, coords, cfg, alpha, rng,
                                      cluster_ids, cluster_sizes, m_counts)
        return None if labels is None else metric_fn(cluster_labels, labels)

    grid = np.linspace(0.0, 1.0, 11)
    scored = [(a, achieved(a)) for a in grid]
    feasible = sorted((a, m) for a, m in scored if m is not None)

    if not feasible:
        raise clm_error(120)

    best_alpha, best_metric = min(feasible, key=lambda am: abs(am[1] - target))
    if abs(best_metric - target) <= tol:
        log.info("target_metric: grid search hit tolerance directly "
                 f"(alpha={best_alpha:.3f}, achieved={best_metric:.3f}, target={target:.3f}).")
        return best_alpha

    # Only the LOW end's metric is carried: the bisection decides which side to
    # keep by comparing mid against `lo_m`, so the other end needs its alpha and
    # nothing else.
    lo, lo_m = feasible[0]
    hi = feasible[-1][0]
    for (a1, m1), (a2, m2) in itertools.pairwise(feasible):
        if (m1 - target) * (m2 - target) <= 0:
            lo, lo_m, hi = a1, m1, a2
            break

    for i in range(max_iter):
        mid = (lo + hi) / 2
        mid_m = achieved(mid)
        if mid_m is None:
            # infeasible mid-bracket, infeasibility only tracks upward with
            # alpha (see docstring), so shrink from the top.
            hi = mid
            continue

        if abs(mid_m - target) < abs(best_metric - target):
            best_alpha, best_metric = mid, mid_m
        if abs(mid_m - target) <= tol:
            log.info(f"target_metric: converged in {i + 1} iteration(s) "
                     f"(alpha={mid:.4f}, achieved={mid_m:.4f}, target={target:.3f}).")
            return mid

        if (mid_m - target) * (lo_m - target) <= 0:
            hi = mid
        else:
            lo, lo_m = mid, mid_m

    clm_warn(log, 306, max_iter=max_iter, best_alpha=best_alpha,
             best_metric=best_metric, target=target, tol=tol)
    return best_alpha


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_clm_labels(cluster_labels: np.ndarray, coords: np.ndarray, cfg: dict, seed: int = 42) -> pd.Series:
    """Generates one synthetic label column for an existing clustering.

    The entry point of the engine

    `coords` may be None/empty only for labels-only configs; any spatial
    placement (centroid_dependence, or competing_noise favoring
    core/boundary) requires real feature vectors and raises [CLM-125].
    """
    rng = np.random.default_rng(seed)
    N = len(cluster_labels)
    M = cfg["num_classes"]
    if not (1 <= M <= MAX_CARDINALITY):
        raise clm_error(126, M=M, max_val=MAX_CARDINALITY)

    coords = _ensure_coords(cfg, coords, N)   # [CLM-125] placement needs geometry
    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    cluster_sizes = {k: int(np.sum(cluster_labels == k)) for k in cluster_ids}
    K = len(cluster_ids)
    if K > MAX_CARDINALITY:
        raise clm_error(127, K=K, max_val=MAX_CARDINALITY)

    # Must precede resolve_label_counts, which is what consumes skew_params: an
    # out-of-range value there does not raise, it returns negative label counts
    # that still sum to N ([CLM-131]).
    _validate_skew_cfg(cfg)

    m_counts = resolve_label_counts(cfg, N, rng)

    if cfg["matching_mode"] == "perfect":
        if cfg["num_classes"] != len(cluster_ids):
            raise clm_error(102, M=cfg["num_classes"], K=len(cluster_ids))
        m_counts = np.array([cluster_sizes[cluster_ids[i]] for i in range(cfg["num_classes"])])

    if cfg["matching_mode"] == "random":
        # Truthiness (not membership): a null/empty target_metric is "not set"
        if cfg.get("target_metric"):
            raise clm_error(114)
        if cfg.get("competing_noise"):
            raise clm_error(115)
        draws = np.repeat(np.arange(cfg["num_classes"]), m_counts)
        rng.shuffle(draws)
        return pd.Series(draws, name="clm_label")

    # Both run before the target-metric solver: a config error must surface as
    # itself, not as a solved score computed over an invalid labeling.
    _validate_spillover_cfg(cfg)
    _validate_centroid_cfg(cfg)
    _validate_target_metric_cfg(cfg, cluster_ids)

    # Resolve once and gate every branch on truthiness. `"target_metric" in cfg`
    # would be True for a null/empty value (`target_metric:` in YAML), then crash
    # at `.get(...)` on None; validation above already treats that as unset.
    tm = cfg.get("target_metric")
    alpha = None  # solved recall level; stays None unless target_metric is set
    if tm:
        if any("recall_target" in row for row in cfg.get("assignment_matrix", [])):
            clm_warn(log, 303)
        if tm.get("scope", "global") == "pair":
            # Exact closed-form single-pair MCC: size the target label so all of
            # it fits its cluster (recall 1), no numerical search.
            m_counts = _pair_label_counts(cfg, cluster_sizes, m_counts, N)
            alpha = 1.0
        else:
            alpha = solve_alpha_for_target_metric(cluster_labels, coords, cfg, cluster_ids,
                                                   cluster_sizes, m_counts, seed=seed)
        rules = build_rules(cfg, cluster_ids, recall_target_override=alpha)
    else:
        rules = build_rules(cfg, cluster_ids)

    # Allocation and placement draw from their OWN stream, started fresh from the
    # same seed rather than continuing the one above. Two reasons, one of them a
    # defect: resolve_label_counts consumes draws for a dirichlet skew and not
    # for any other rule, so a shared stream reaches allocation in a state that
    # depends on which skew rule was chosen. The target-metric
    # probes cannot reproduce that state, so the labeling that was scored and the
    # labeling that was written came out different, [CLM-309]. With both
    # starting from default_rng(seed), the winning probe IS the delivered result.
    alloc_rng = np.random.default_rng(seed)
    out = _run_allocation_pipeline(cluster_labels, coords, cfg, rules, cluster_ids,
                                    cluster_sizes, m_counts, alloc_rng)

    achieved_counts = np.bincount(out, minlength=cfg["num_classes"])
    log.info(f"CLM labels generated. Target counts: {m_counts.tolist()}, achieved: {achieved_counts.tolist()}.")

    if tm:
        if tm.get("scope", "global") == "pair":
            sm = cfg["single_match"]
            achieved = clustering_mcc_pair(cluster_labels, out, sm["cluster"], sm["label"])
            log.info(f"target_metric: final achieved pair mcc={achieved:.4f} "
                     f"(target was {tm['value']}, pair cluster={sm['cluster']}/label={sm['label']}).")

            # Honors the same 'tolerance' key the global solver reads. It was
            # hardcoded to 0.01 here, so a requested 0.001 was silently widened and
            # a requested 0.05 silently narrowed. Only 'max_iter' stays global-only:
            # the pair scope inverts in closed form and never iterates.
            pair_tol = tm.get("tolerance", 0.01)
            if abs(achieved - tm["value"]) > pair_tol:
                lstar_mask = (out == sm["label"])
                total = int(np.sum(lstar_mask))
                outside = int(np.sum(lstar_mask & (np.asarray(cluster_labels) != sm["cluster"])))
                clm_warn(log, 310, achieved=achieved, target=tm["value"], tol=pair_tol,
                         label=sm["label"], cluster=sm["cluster"],
                         outside=outside, total=total)
        else:
            final_metric = _METRIC_FUNCS[tm["type"]](cluster_labels, out)
            log.info(f"target_metric: final achieved {tm['type']}={final_metric:.4f} "
                     f"(target was {tm['value']}, alpha={alpha:.4f}).")

            tol = tm.get("tolerance", 0.01)
            if abs(final_metric - tm["value"]) > tol:
                clm_warn(log, 309, type=tm["type"], achieved=final_metric,
                         target=tm["value"], tol=tol, alpha=alpha)

    return pd.Series(out, name="clm_label")
