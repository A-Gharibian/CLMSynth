# SPDX-License-Identifier: MIT
# config_validation.py
"""
The checks on a clm_label block, and what they build for the engine: the matching
rules from single_match or assignment_matrix, and a competing_noise or
feature_dependence entry with its defaults. clm_label_engine calls them, in its
order; the stages its check before a dataset and generate_clm_labels share are
`_checked_block`, `_check_sizing_cfg` and `_check_matching_cfg`. Also the run's
check for keys it does not read in the rest of a config.
"""

import difflib
import logging
import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from .clm_errors import clm_error, clm_infeasible, clm_missing, clm_warn
from .config_tree import SETTINGS, applies, choices, default, drop_empty, in_bounds

__all__ = [
    "BALANCE_VALUES",
    "MATCHING_MODES",
    "MAX_CARDINALITY",
    "Rule",
    "build_rules",
    "validate_matching_ids",
]

log: logging.Logger = logging.getLogger(__name__)

MAX_CARDINALITY = int(cast(float, SETTINGS["clm_label.num_classes"].hi))
MATCHING_MODES = choices("clm_label.matching_mode")
BALANCE_VALUES = choices("clm_label.balance")

_FEATURE_ENTRY_EXAMPLE = "{label: 0, feature: Feature_1, favors: high}"

# The private clm_label key under which LabelGenerator.apply hands over the names of
# the coords columns: the engine takes an array, and a feature_dependence entry names
# its feature.
_FEATURE_COLUMNS = "_feature_columns"

_Number = Fraction | float | int


def _exact(x: _Number) -> Fraction:
    """A number as written: a float is its shortest decimal, so 0.7 is 7/10, not
    the float nearest it. Counts are rounded from these, so 0.7 and 0.3 of 1435
    points (1004.5 and 430.5) tie, and the tie rule decides, not float noise."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    # A numpy float prints the shortest decimal of its own precision: float() would
    # widen np.float32(0.6) to 0.6000000238418579.
    written: Any = x
    return Fraction(str(written if isinstance(written, np.floating) else float(written)))


# ---------------------------------------------------------------------------
# matching_mode -> a uniform rule list: (label, clusters, recall_target)
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """One matching rule: route `recall_target` of label's budget into `clusters`."""
    label: int
    clusters: list[int]
    recall_target: float


def _check_label(label: Any, M: int, where: str) -> None:
    """[CLM-104] A label is an integer in 0..M-1."""
    if not _is_int(label):
        raise clm_error(104, where=where, label=label, hi=M - 1,
                        problem="is not an integer; labels are indices in")
    if not (0 <= label < M):
        raise clm_error(104, where=where, label=label, hi=M - 1, problem="out of range")


def _check_pair(label: int, clusters: list[Any], M: int, cluster_ids: list[Any] | None,
                where: str) -> None:
    """Fail fast on out-of-range label / unknown cluster ids in a matching
    rule, before allocate() would otherwise surface them as an uncoded numpy
    IndexError (label >= M), a KeyError (unknown cluster). Without a dataset
    (`cluster_ids` None) the ids are not looked up ([CLM-105])."""
    _check_label(label, M, where)
    # [CLM-141] Configuration errors, so not [CLM-105], which skips one dataset.
    if not isinstance(clusters, (list, tuple)):
        raise clm_error(141, key=f"{where}: clusters", expected="a list of cluster ids",
                        value=clusters)
    if not clusters:
        raise clm_error(141, key=f"{where}: clusters",
                        expected="a non-empty list of cluster ids", value=clusters)
    for k in clusters:
        _check_cluster_id(k, f"{where}: cluster")
    if cluster_ids is None:
        return
    unknown = [k for k in clusters if k not in cluster_ids]
    if unknown:
        raise clm_error(105, where=where, unknown=unknown,
                        available=sorted(cluster_ids, key=str))


def _check_named_once(named: dict[tuple[Any, Any], int], label, clusters: list[Any], i: int) -> None:
    """[CLM-142] A cluster is named once per label across assignment_matrix.
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


def _single_match(cfg: dict[str, Any]) -> dict[str, Any]:
    """single_match with both required keys."""
    sm: dict[str, Any] | None = cfg.get("single_match")
    if sm is None:
        raise clm_missing(204)
    _check_mapping(sm, "single_match", "{cluster: 0, label: 0}")
    for key in ("cluster", "label"):
        if key not in sm:
            raise clm_missing(205, key=key)
    return sm


def build_rules(cfg: dict[str, Any], cluster_ids: list[int],
                 recall_target_override: float | None = None) -> list[Rule]:
    """Normalizes the matching mode into a validated list of `Rule` records.

    `recall_target_override` replaces every rule's recall with one global
    value; the target-metric solver passes its solved alpha through it.
    """
    return _rules(cfg, cluster_ids, recall_target_override)


def _rules(cfg: dict[str, Any], cluster_ids: list[int] | None,
           recall_target_override: float | None = None, warn: bool = True) -> list[Rule]:
    """build_rules. Without a dataset (`cluster_ids` None) the checks on its ids
    and their count K are left out, and 'perfect' has no rules."""
    mode = cfg["matching_mode"]
    M = cfg["num_classes"]
    K = None if cluster_ids is None else len(cluster_ids)
    shown_K = "unknown" if K is None else K

    if mode == "perfect":
        if recall_target_override is not None:
            raise clm_error(111, mode="perfect", M=M, K=shown_K)
        if K is not None and M != K:
            raise clm_error(102, M=M, K=K)
        # Only warn about keys the caller actually set.
        if warn and any(k in cfg for k in ("proportions", "balance", "skew_rule",
                                           "skew_params")):
            clm_warn(log, 302)
        if cluster_ids is None:
            return []
        return [Rule(label=i, clusters=[cluster_ids[i]], recall_target=1.0) for i in range(M)]

    if mode == "single":
        if M < 2 or (K is not None and K < 2):
            raise clm_error(103, M=M, K=shown_K)
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


def _check_recall_sums(rules: list[Rule]) -> None:
    """[CLM-153] without a dataset: one label's recall_target values add up past
    1.0, so its rules claim more than its budget on every dataset. The claim is
    named as a share of the budget, since the counts need the dataset's size."""
    total: dict[int, Fraction] = {}
    for rule in rules:
        total[rule.label] = total.get(rule.label, Fraction(0)) + _exact(rule.recall_target)
    for label, recall in total.items():
        if recall > 1:
            named = [rule for rule in rules if rule.label == label]
            raise clm_infeasible(153, label=label, claimed=f"{float(recall) * 100:g}% of its",
                                 budget="100%", n_rules=len(named),
                                 breakdown="; ".join(f"clusters {rule.clusters} at recall "
                                                     f"{rule.recall_target}" for rule in named))


def validate_matching_ids(cfg: dict[str, Any], cluster_ids: list[int]) -> None:
    """Run only the [CLM-104]/[CLM-105] id checks against ONE dataset's cluster ids.

    Deliberately tolerant of a malformed config. [CLM-105] is a statement about a
    *dataset*, unlike every other [CLM-1xx] code. [CLM-104] bounds labels by
    num_classes, so it is configuration-scoped; _single_run.py evaluates it once.
    """
    M = cfg.get("num_classes")
    if not isinstance(M, (int, np.integer)) or isinstance(M, bool):
        return
    M = int(M)

    mode = cfg.get("matching_mode")
    if mode == "single":
        sm = cfg.get("single_match") or {}
        if isinstance(sm, Mapping) and "label" in sm and "cluster" in sm:
            _check_pair(sm["label"], [sm["cluster"]], M, cluster_ids, "single_match")
    elif mode == "custom":
        matrix = cfg.get("assignment_matrix")
        for i, row in enumerate(matrix if isinstance(matrix, (list, tuple)) else []):
            if isinstance(row, Mapping) and "label" in row and "clusters" in row:
                _check_pair(row["label"], row["clusters"], M, cluster_ids,
                            f"assignment_matrix row {i}")


def _placement(cfg: dict[str, Any]) -> str | None:
    """What requests spatial placement, or None: `centroid_dependence.enabled`,
    any competing_noise entry whose favors is 'core'/'boundary' (the default is
    'boundary'), or a feature_dependence entry; under 'random' none, as random
    places nothing ([CLM-311], [CLM-115]). The blocks are first read here, so their
    shape ([CLM-141]) is checked here too, centroid_dependence's and
    feature_dependence's in every mode but 'random'.
    """
    random_mode = cfg["matching_mode"] == "random"
    cd = {} if random_mode else cfg.get("centroid_dependence") or {}
    _check_mapping(cd, "centroid_dependence", "{enabled: true, favors: core}")
    # [CLM-141] Checked here, where it is first read: a quoted "false" is truthy.
    if "enabled" in cd and not isinstance(cd["enabled"], (bool, np.bool_)):
        raise clm_error(141, key="centroid_dependence.enabled", expected="true or false",
                        value=cd["enabled"])
    noise = cfg.get("competing_noise") or []
    _check_list_of_mappings(noise, "competing_noise", "entry",
                            "{cluster: 1, label: 2, share: 0.5}")
    spatial_noise = [] if random_mode else [
        e for e in noise if e.get("favors", default("clm_label.competing_noise.favors")) != "random"]
    follows = [] if random_mode else cfg.get("feature_dependence") or []
    _check_list_of_mappings(follows, "feature_dependence", "entry", _FEATURE_ENTRY_EXAMPLE)
    if cd.get("enabled"):
        return "centroid_dependence.enabled"
    if spatial_noise:
        return "competing_noise with favors 'core'/'boundary'"
    if follows:
        return "feature_dependence"
    return None


def _tolerance(tm: Mapping[str, Any]) -> Any:
    """target_metric.tolerance; when absent, 0 under strict and 0.01 otherwise."""
    return tm.get("tolerance",
                  0.0 if tm.get("strict") else default("clm_label.target_metric.tolerance"))


def _validate_target_metric_cfg(cfg: dict[str, Any], cluster_ids: list[Any] | None) -> None:
    # An absent OR empty/null target_metric (e.g. `target_metric:` in YAML -> None)
    # is a no-op. Without a dataset (`cluster_ids` None), K is not checked.
    tm = cfg.get("target_metric")
    if not tm:
        return
    mode = cfg["matching_mode"]
    M = cfg["num_classes"]
    K = None if cluster_ids is None else len(cluster_ids)
    # Under single, [CLM-103] refuses M or K below 2.
    if mode not in ("single", "custom") or (
            mode == "custom" and (M < 2 or (K is not None and K < 2))):
        raise clm_error(111, mode=mode, M=M, K="unknown" if K is None else K)
    _check_mapping(tm, "target_metric", "{type: mcc, value: 0.5}")
    if tm.get("type") not in choices("clm_label.target_metric.type"):
        raise clm_error(112, type=tm.get("type"))
    if "value" not in tm:
        raise clm_missing(209)
    value = tm["value"]
    if not _is_real(value) or not in_bounds("clm_label.target_metric.value", value):
        raise clm_error(113, value=value if _is_real(value) else repr(value))
    if "strict" in tm and not isinstance(tm["strict"], (bool, np.bool_)):
        raise clm_error(141, key="target_metric.strict", expected="true or false",
                        value=tm["strict"])
    # [CLM-141] Under a NaN or infinite tolerance [CLM-309]/[CLM-310] could not fire.
    tol = _tolerance(tm)
    if not _is_finite_real(tol) or not in_bounds("clm_label.target_metric.tolerance", tol):
        raise clm_error(141, key="target_metric.tolerance", expected="a finite number >= 0",
                        value=tol)
    scope = tm.get("scope", default("clm_label.target_metric.scope"))
    if scope not in choices("clm_label.target_metric.scope"):
        raise clm_error(122, scope=scope)
    # max_iter is read only by the global search.
    max_iter = tm.get("max_iter", default("clm_label.target_metric.max_iter"))
    if scope == "global" and (isinstance(max_iter, (bool, np.bool_))
                              or not isinstance(max_iter, (int, np.integer))
                              or not in_bounds("clm_label.target_metric.max_iter", max_iter)):
        raise clm_error(141, key="target_metric.max_iter", expected="an integer >= 0",
                        value=max_iter)
    # probe_seed too is read only by the global search; empty is the run seed, as
    # absent.
    probe_seed = tm.get("probe_seed")
    if scope == "global" and probe_seed is not None and (
            isinstance(probe_seed, (bool, np.bool_))
            or not isinstance(probe_seed, (int, np.integer))
            or not in_bounds("clm_label.target_metric.probe_seed", probe_seed)):
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
        spill = _spillover_rule(cfg)
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


def _spillover_rule(cfg: dict[str, Any]) -> Any:
    """spillover_rule as written; absent or empty, proportional_to_marginal."""
    rule = cfg.get("spillover_rule")
    return default("clm_label.spillover_rule") if rule is None else rule


def _validate_spillover_cfg(cfg: dict[str, Any]) -> None:
    """
    [CLM-109] guard: spillover_rule must name a known rule.
    [CLM-128] guard: concentrated_labels must name existing labels.
    """
    rule_name = _spillover_rule(cfg)
    if rule_name not in choices("clm_label.spillover_rule"):
        raise clm_error(109, rule_name=rule_name)
    if rule_name != "concentrated":
        return
    given = cfg.get("concentrated_labels")
    if given is None:
        return                      # documented default: the single largest label

    M = cfg["num_classes"]
    valid = isinstance(given, (list, tuple)) and len(given) > 0
    if valid:
        for v in given:
            try:                    # accept 2 and 2.0, reject 1.5, True, "2", None, inf
                is_int = not isinstance(v, (bool, np.bool_)) and v == int(v)
            except (TypeError, ValueError, OverflowError):
                is_int = False
            if not is_int or not (0 <= v < M):
                valid = False
                break
    if not valid:
        raise clm_error(128, hi=M - 1, given=given)


def _is_real(value) -> bool:
    """Return whether the value is real, excluding `bool` because it is an
    `int` subclass."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_finite_real(value) -> bool:
    """_is_real, excluding NaN, inf and integers too large for a float. Every
    comparison with NaN is False, so a NaN passes any range check that refuses only
    the out-of-range side."""
    if not _is_real(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_mapping(value, key: str, example: str) -> None:
    """[CLM-141] A block read key by key must be a mapping."""
    if not isinstance(value, Mapping):
        raise clm_error(141, key=key, expected=f"a mapping such as {example}", value=value)


def _check_list_of_mappings(value, key: str, item: str, example: str) -> None:
    """[CLM-141] A list block whose entries are mappings."""
    if not isinstance(value, (list, tuple)):
        raise clm_error(141, key=key, expected=f"a list of mappings such as [{example}]",
                        value=value)
    for i, entry in enumerate(value):
        _check_mapping(entry, f"{key} {item} {i}", example)


def _is_int(value: object) -> bool:
    """An integer, and not a bool: a label, a cluster id or a whole-number setting."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_cluster_id(k, key: str) -> None:
    """[CLM-141] One cluster id: an integer."""
    if not _is_int(k):
        raise clm_error(141, key=key, expected="one cluster id (an integer)", value=k)


def _keys_under(prefix: str) -> frozenset[str]:
    """The keys directly under `prefix` that the settings table names."""
    return frozenset(path[len(prefix):].partition(".")[0]
                     for path in SETTINGS if path.startswith(prefix))


# The keys each block reads: the settings table's, and those it does not hold.
_SECTION_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "global_settings": _keys_under("global_settings."),
    "label_generation": _keys_under("label_generation.") | {"clm_label"},
}
_BLOCK_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "clm_label": _keys_under("clm_label.") | {"proportions", "concentrated_labels",
                                              "single_match", "assignment_matrix"},
    "skew_params": _keys_under("clm_label.skew_params."),
    "single_match": frozenset({"cluster", "label"}),
    "target_metric": _keys_under("clm_label.target_metric."),
    "centroid_dependence": _keys_under("clm_label.centroid_dependence."),
}
_ENTRY_KEYS: Final[Mapping[str, tuple[str, frozenset[str]]]] = {
    "assignment_matrix": ("row", frozenset({"label", "clusters", "recall_target"})),
    "competing_noise": ("entry", _keys_under("clm_label.competing_noise.")
                        | {"cluster", "label"}),
    "feature_dependence": ("entry", _keys_under("clm_label.feature_dependence.")
                           | {"label", "feature", "value"}),
}
# Keys a block takes without naming them in its messages.
_UNNAMED: Final[Mapping[str, frozenset[str]]] = {
    "label_generation": frozenset({"noise"}),
    "clm_label": frozenset({_FEATURE_COLUMNS}),
}


def _refuse_unknown(block: Mapping[Any, Any], takes: Set[str], where: str,
                    kind: str = "key") -> None:
    """[CLM-141] for the first key of `block` that is not in `takes`, naming the
    closest one that is. A key with no value is left alone, as an empty one it
    reads is."""
    unnamed = _UNNAMED.get(where, frozenset())
    for key, value in block.items():
        if value is None or key in takes or key in unnamed:
            continue
        known = sorted(takes)
        close = difflib.get_close_matches(str(key), known, n=1)
        if close:
            others = ", ".join(k for k in known if k != close[0])
            expected = f"'{close[0]}' or another {kind} it reads ({others})"
        else:
            expected = f"a {kind} it reads ({', '.join(known)})"
        raise clm_error(141, key=f"a {kind} of {where}", expected=expected, value=key)


def _check_config_keys(config: Mapping[Any, Any], suite: str,
                       suite_takes: Set[str] | None) -> None:
    """[CLM-141] for a key the run does not read, outside clm_label, which the engine
    checks: the config's sections, global_settings, label_generation, and `suite`
    when `suite_takes` lists its keys. Any `<name>_suite` is a section, read only
    when it is the run's. A part of another shape is left to its own check."""
    sections = {*_SECTION_KEYS, suite,
                *(k for k in config if isinstance(k, str) and k.endswith("_suite"))}
    _refuse_unknown(config, sections, "the config", "section")
    for name, takes in _SECTION_KEYS.items():
        block = config.get(name)
        if isinstance(block, Mapping):
            _refuse_unknown(block, takes, name)
    block = config.get(suite)
    if suite_takes is not None and isinstance(block, Mapping):
        _refuse_unknown(block, suite_takes, suite)


def _check_clm_keys(cfg: Mapping[Any, Any]) -> None:
    """[CLM-141] for a key the clm_label block does not read, in the block and in
    each block and entry inside it, whatever the mode. A part of another shape is
    left to its own check."""
    _refuse_unknown(cfg, _BLOCK_KEYS["clm_label"], "clm_label")
    for name in ("skew_params", "single_match", "target_metric", "centroid_dependence"):
        block = cfg.get(name)
        if isinstance(block, Mapping):
            _refuse_unknown(block, _BLOCK_KEYS[name], name)
    for name, (item, takes) in _ENTRY_KEYS.items():
        entries = cfg.get(name)
        if isinstance(entries, (list, tuple)):
            for i, entry in enumerate(entries):
                if isinstance(entry, Mapping):
                    _refuse_unknown(entry, takes, f"{name} {item} {i}")


def _validate_balance_cfg(cfg: dict[str, Any]) -> None:
    """[CLM-132] guard: balance must be exactly 'balanced' or 'unbalanced'.
    """
    if "balance" in cfg and cfg["balance"] not in BALANCE_VALUES:
        raise clm_error(132, balance=cfg["balance"])


def _validate_skew_cfg(cfg: dict[str, Any]) -> None:
    """[CLM-131] guard: skew parameters must be in range for the chosen skew_rule.

    Called from generate_clm_labels *before* resolve_label_counts,
    Unknown skew_rule values stay [CLM-107], raised by _skewed_proportions itself.
    """
    if cfg.get("balance", default("clm_label.balance")) == "balanced" or cfg.get("proportions"):
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
        ratio = params.get("ratio", default("clm_label.skew_params.ratio"))
        if not _is_finite_real(ratio) or not in_bounds("clm_label.skew_params.ratio", ratio):
            problem = (f"geometric 'ratio' must be a finite number >= 0, got {ratio!r}; a "
                       "negative ratio alternates sign across labels")

    elif rule == "dominant_minority":
        if M < 2:
            problem = ("dominant_minority needs num_classes >= 2, got 1; the rule "
                       "divides the remaining share by (num_classes - 1)")
        else:
            share = params.get("dominant_share", default("clm_label.skew_params.dominant_share"))
            index = params.get("dominant_index", default("clm_label.skew_params.dominant_index"))
            if not _is_real(share) or not in_bounds("clm_label.skew_params.dominant_share", share):
                problem = (f"dominant_minority 'dominant_share' must be a number in "
                           f"[0, 1], got {share!r}")
            elif isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                problem = (f"dominant_minority 'dominant_index' must be an integer, got "
                           f"{index!r}; it indexes the proportions list directly")
            elif not in_bounds("clm_label.skew_params.dominant_index", index) or index >= M:
                problem = (f"dominant_minority 'dominant_index' must be in 0..{M - 1}, "
                           f"got {index!r}")

    elif rule == "dirichlet":
        alpha = params.get("alpha", default("clm_label.skew_params.alpha"))
        if not _is_finite_real(alpha) or not in_bounds("clm_label.skew_params.alpha", alpha):
            problem = (f"dirichlet 'alpha' must be a finite number > 0, got {alpha!r}; at "
                       "exactly 0 every draw is 0 and normalising them divides by zero")
        # Each draw is about alpha, so from alpha ~ float max / M their sum overflows
        # and every share comes out 0. numpy's running sum gets there up to a few
        # ULPs below the exact quotient, hence the margin.
        elif float(alpha) > np.finfo(float).max / M * (1 - 1e-14):
            problem = (f"dirichlet 'alpha' must be below about "
                       f"{np.finfo(float).max / M:.3g} with num_classes {M}, got {alpha!r}; "
                       f"the {M} draws add up past the largest float, so every share "
                       "comes out 0")

    if problem is not None:
        raise clm_error(131, problem=problem, hi=M - 1)


def _validate_centroid_cfg(cfg: dict[str, Any]) -> None:
    """[CLM-129] favors, [CLM-110] profile and [CLM-141] steepness, each checked
    wherever placement reads it.
    """
    cd = cfg.get("centroid_dependence") or {}
    enabled = cd.get("enabled", default("clm_label.centroid_dependence.enabled"))
    if enabled:
        favors = cd.get("favors", default("clm_label.centroid_dependence.favors"))
        if favors not in choices("clm_label.centroid_dependence.favors"):
            raise clm_error(129, favors=favors)
    if not applies("clm_label.centroid_dependence.profile", cfg):
        return
    profile = cd.get("profile", default("clm_label.centroid_dependence.profile"))
    if profile not in choices("clm_label.centroid_dependence.profile"):
        raise clm_error(110, profile=profile)
    steepness = cd.get("steepness", default("clm_label.centroid_dependence.steepness"))
    if profile == "exponential" and (
            not _is_finite_real(steepness)
            or not in_bounds("clm_label.centroid_dependence.steepness", steepness)):
        raise clm_error(141, key="centroid_dependence.steepness",
                        expected="a finite number >= 0", value=steepness)


def _validate_split_cfg(cfg: dict[str, Any]) -> None:
    """[CLM-108] guard: split_rule must name a known rule wherever 'custom' reads it.
    """
    if cfg.get("matching_mode") != "custom":
        return
    split_rule = cfg.get("split_rule", default("clm_label.split_rule"))
    if split_rule not in choices("clm_label.split_rule"):
        raise clm_error(108, split_rule=split_rule)


def _noise_entry(i: int, entry: Mapping[str, Any],
                 M: int) -> tuple[Any, int | np.integer, Any, str]:
    """One competing_noise entry as (cluster, label, share, favors), defaults filled
    in, after the checks that hold for every dataset."""
    for key in ("cluster", "label"):
        if key not in entry:
            raise clm_missing(207, where=f"competing_noise entry {i}", key=key)
    k, label = entry["cluster"], entry["label"]
    share = entry.get("share", default("clm_label.competing_noise.share"))
    favors = entry.get("favors", default("clm_label.competing_noise.favors"))

    # [CLM-141], not [CLM-119]: a config error, not a dataset's.
    _check_cluster_id(k, f"competing_noise entry {i}: cluster")
    if not _is_int(label):
        raise clm_error(118, label=label, hi=M - 1,
                        problem="is not an integer; labels are indices in")
    if not (0 <= label < M):
        raise clm_error(118, label=label, hi=M - 1, problem="out of range")
    if not _is_real(share) or not in_bounds("clm_label.competing_noise.share", share):
        raise clm_error(117, share=share if _is_real(share) else repr(share))
    if favors not in choices("clm_label.competing_noise.favors"):
        raise clm_error(116, favors=favors)
    return k, label, share, favors


def _plain_pair(label, k) -> bool:
    """An integer label and an integer cluster id: what build_rules accepts.
    Anything else is left for it to report ([CLM-104], [CLM-141])."""
    return _is_int(label) and _is_int(k)


def _named_targets(cfg: dict[str, Any]) -> dict[tuple[Any, Any], str]:
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


def _validate_competing_noise_cfg(cfg: dict[str, Any]) -> None:
    """Each competing_noise entry, checked once before allocation and the
    target_metric search: [CLM-207], [CLM-141], [CLM-118], [CLM-117] and
    [CLM-116], then [CLM-144] and [CLM-142]. All of them hold for every dataset.
    [CLM-119] is about a dataset's ids: the allocation checks it before the rules
    claim any point.
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


def _feature_entry(i: int, entry: Mapping[str, Any],
                   M: int) -> tuple[int, str, str, float | None, str]:
    """One feature_dependence entry as (label, feature, favors, value, profile),
    defaults filled in, after the checks that hold for every dataset. `value` is
    None unless favors is 'value', the only place it is read."""
    where = f"feature_dependence entry {i}"
    for key in ("label", "feature"):
        if key not in entry:
            raise clm_missing(207, where=where, key=key)
    label, feature = entry["label"], entry["feature"]
    _check_label(label, M, where)
    if not isinstance(feature, str) or not feature:
        raise clm_error(141, key=f"{where}: feature",
                        expected="the name of one feature column", value=feature)
    favors = entry.get("favors", default("clm_label.feature_dependence.favors"))
    if (not isinstance(favors, str)
            or favors not in choices("clm_label.feature_dependence.favors")):
        raise clm_error(141, key=f"{where}: favors", expected="'high', 'low' or 'value'",
                        value=favors)
    value = None
    if favors == "value":
        value = entry.get("value")
        if not _is_finite_real(value):
            raise clm_error(141, key=f"{where}: value",
                            expected="a finite number, which favors 'value' requires",
                            value=value)
    profile = entry.get("profile", default("clm_label.feature_dependence.profile"))
    if (not isinstance(profile, str)
            or profile not in choices("clm_label.feature_dependence.profile")):
        raise clm_error(141, key=f"{where}: profile", expected="'linear' or 'step'",
                        value=profile)
    return label, feature, favors, value, profile


def _validate_feature_dependence_cfg(cfg: dict[str, Any]) -> None:
    """Each feature_dependence entry, checked once before allocation and the
    target_metric search: [CLM-207], [CLM-104] and [CLM-141], then [CLM-141] for a
    second entry on a label and for centroid_dependence enabled beside entries. All
    of them hold for every dataset. [CLM-125] is about a dataset's columns
    (_followed_features).
    """
    entries = cfg.get("feature_dependence") or []
    _check_list_of_mappings(entries, "feature_dependence", "entry", _FEATURE_ENTRY_EXAMPLE)
    first: dict[int, int] = {}
    for i, entry in enumerate(entries):
        label = _feature_entry(i, entry, cfg["num_classes"])[0]
        if label in first:
            # Two features give a label no single score.
            raise clm_error(141, key=f"feature_dependence entry {i}: label",
                            expected=f"a label no earlier entry names (entry {first[label]} "
                                     f"has label {label}, and a label follows one feature)",
                            value=label)
        first[label] = i
    cd = cfg.get("centroid_dependence") or {}
    if entries and cd.get("enabled"):
        raise clm_error(141, key="centroid_dependence.enabled",
                        expected="false while feature_dependence has entries (points are "
                                 "placed by their distance to the centroid or by a feature, "
                                 "not both)",
                        value=cd["enabled"])


def _refuse_arrays(cfg: dict[str, Any]) -> None:
    """An array in a setting that takes text, a list or a mapping, refused where the
    setting is read, with the code its own check gives a value of the wrong type:
    `cfg.get(key) or ...` and `value in choices` ask it for one truth value, which
    an array of several values does not have."""
    mode = cfg["matching_mode"]

    def array(key: str) -> bool:
        return isinstance(cfg.get(key), np.ndarray)

    if mode != "perfect":
        if array("balance"):
            raise clm_error(132, balance=cfg["balance"])
        if array("proportions") and cfg.get("balance", default("clm_label.balance")) != "balanced":
            raise clm_error(141, key="proportions",
                            expected="a list of finite numbers >= 0, one per label",
                            value=cfg["proportions"])
        if (array("skew_rule") and cfg.get("balance", default("clm_label.balance")) != "balanced"
                and not cfg.get("proportions")):
            raise clm_error(107, skew_rule=cfg["skew_rule"])
    if mode != "random" and array("spillover_rule"):
        raise clm_error(109, rule_name=cfg["spillover_rule"])
    if mode == "custom" and array("split_rule"):
        raise clm_error(108, split_rule=cfg["split_rule"])
    if array("centroid_dependence"):
        _check_mapping(cfg["centroid_dependence"], "centroid_dependence",
                       "{enabled: true, favors: core}")
    if array("competing_noise"):
        _check_list_of_mappings(cfg["competing_noise"], "competing_noise", "entry",
                                "{cluster: 1, label: 2, share: 0.5}")
    if array("feature_dependence"):
        _check_list_of_mappings(cfg["feature_dependence"], "feature_dependence", "entry",
                                _FEATURE_ENTRY_EXAMPLE)
    if array("target_metric"):
        _check_mapping(cfg["target_metric"], "target_metric", "{type: mcc, value: 0.5}")


def _check_classes_and_mode(cfg: dict[str, Any]) -> None:
    """[CLM-201]/[CLM-126] num_classes, then [CLM-202]/[CLM-101] matching_mode, then
    an array where a setting is read (`_refuse_arrays`)."""
    if "num_classes" not in cfg:
        raise clm_missing(201)
    M = cfg["num_classes"]
    # bool is an int subclass: `num_classes: yes` would otherwise pass as 1.
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)):
        raise clm_error(126, M=repr(M), max_val=MAX_CARDINALITY)
    if not in_bounds("clm_label.num_classes", M):
        raise clm_error(126, M=M, max_val=MAX_CARDINALITY)

    # The mode decides which of the other keys are read at all, so it is checked
    # before any of them: an error in a key the mode may never read must not stand
    # in for a missing or misspelled mode.
    # Written with no value (`matching_mode:`) is left out too, not an unknown mode.
    if cfg.get("matching_mode") is None:
        raise clm_missing(202)
    mode = cfg["matching_mode"]
    if isinstance(mode, np.ndarray) or mode not in MATCHING_MODES:
        raise clm_error(101, mode=mode)
    _refuse_arrays(cfg)


def _check_random_cfg(cfg: dict[str, Any], warn: bool = True) -> None:
    """Under 'random': [CLM-114], [CLM-115], and [CLM-311] for an enabled
    centroid_dependence and for a feature_dependence of any shape."""
    # Validity (not membership): a null/empty target_metric is "not set"
    if cfg.get("target_metric"):
        raise clm_error(114)
    if cfg.get("competing_noise"):
        raise clm_error(115)
    cd = cfg.get("centroid_dependence")
    if warn and cd and (not isinstance(cd, Mapping) or cd.get("enabled")):
        clm_warn(log, 311, block="centroid_dependence", by="its distance to a centroid")
    if warn and cfg.get("feature_dependence"):
        clm_warn(log, 311, block="feature_dependence", by="the value of a feature")
    if warn and _given(cfg.get("concentrated_labels")):
        clm_warn(log, 311, block="concentrated_labels", by="a spillover rule")


def _given(value: Any) -> bool:
    """Whether a setting holds a value: empty (null, an empty list) is absent."""
    return value is not None and not (isinstance(value, (list, tuple)) and len(value) == 0)


def _warn_unread(cfg: dict[str, Any]) -> None:
    """[CLM-312] for a setting given where it is not read, of those the template
    writes only when asked; run after the checks, which have refused bad shapes."""
    if (_given(cfg.get("concentrated_labels"))
            and _spillover_rule(cfg) != "concentrated"):
        clm_warn(log, 312, setting="concentrated_labels",
                 when="under spillover_rule 'concentrated'")
    cd = cfg.get("centroid_dependence") or {}
    if (cd.get("steepness") is not None
            and cd.get("profile", default("clm_label.centroid_dependence.profile"))
            != "exponential"):
        clm_warn(log, 312, setting="centroid_dependence.steepness",
                 when="under profile 'exponential'")
    tm = cfg.get("target_metric")
    if tm and tm.get("scope", default("clm_label.target_metric.scope")) == "pair":
        for key in ("max_iter", "probe_seed"):
            if tm.get(key) is not None:
                clm_warn(log, 312, setting=f"target_metric.{key}", when="under scope 'global'")


def _warn_recall_targets_ignored(cfg: dict[str, Any]) -> None:
    """[CLM-303] under target_metric, for rows that set their own recall_target."""
    # Well-formed rows only: under custom, build_rules reports a valueless
    # matrix ([CLM-206]) and a malformed one ([CLM-141]); the other modes
    # never read it.
    matrix = cfg.get("assignment_matrix")
    if any(isinstance(row, Mapping) and "recall_target" in row
           for row in (matrix if isinstance(matrix, (list, tuple)) else [])):
        clm_warn(log, 303)


# ---------------------------------------------------------------------------
# The stages the engine's check before a dataset (_check_config) and
# generate_clm_labels share, each run by both in this order.
# ---------------------------------------------------------------------------

def _checked_block(cfg: Any) -> dict[str, Any]:
    """The block itself: a mapping, its empty settings dropped as absent, no key it
    does not read, then num_classes and matching_mode. Returns the block without
    its empty settings."""
    _check_mapping(cfg, "clm_label", "{num_classes: 4, matching_mode: perfect}")
    block: dict[str, Any] = drop_empty(cfg)
    _check_clm_keys(block)
    _check_classes_and_mode(block)
    return block


def _check_sizing_cfg(cfg: dict[str, Any]) -> None:
    """Outside 'perfect', before the label counts are resolved: balance, then
    skew_rule and skew_params."""
    _validate_balance_cfg(cfg)
    _validate_skew_cfg(cfg)


def _check_matching_cfg(cfg: dict[str, Any], cluster_ids: list[int] | None,
                        warn: bool = True) -> None:
    """Outside 'perfect' and 'random', before any rule is built: spillover,
    centroid_dependence, target_metric (against `cluster_ids` when known), split,
    competing_noise and feature_dependence, then [CLM-312] for a setting given
    where it is not read, unless `warn` is False."""
    _validate_spillover_cfg(cfg)
    _validate_centroid_cfg(cfg)
    _validate_target_metric_cfg(cfg, cluster_ids)
    _validate_split_cfg(cfg)
    _validate_competing_noise_cfg(cfg)
    _validate_feature_dependence_cfg(cfg)
    if warn:
        _warn_unread(cfg)
