# SPDX-License-Identifier: MIT
# config_tree.py
"""What the run accepts for each setting, and what it uses when the setting is
absent. One record per setting, in the order the run's config is written. Only
settings with a default, choices, bounds or a type the command checks have a record.

An empty setting (YAML `key:`, JSON null) is absent, except `balance`, refused, and
the seeds, for which empty draws one."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

__all__ = ["REQUIRED", "SETTINGS", "Kind", "Setting", "applies", "choices", "default",
           "drop_empty", "in_bounds"]


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


# The default of a setting the run refuses to do without.
REQUIRED: Final = _Required()

# What the command checks a value against once the config is loaded, for the
# settings no [CLM-###] code covers: global_settings, label_generation and the
# <source>_suite keys. Empty passes: what it means is decided where it is read.
Kind = Literal["text", "path", "flag", "whole", "count", "names"]

# competing_noise favors values that place an entry by its distance to a centroid.
_PLACED_BY_DISTANCE: Final = frozenset({"core", "boundary"})


def _profile_read(clm: Mapping[str, Any]) -> bool:
    """Whether the centroid profile is read: centroid_dependence is enabled, or a
    competing_noise entry is placed by distance, and the mode is not random, which
    places nothing ([CLM-311], [CLM-115])."""
    if clm.get("matching_mode") == "random":
        return False
    cd = clm.get("centroid_dependence")
    if isinstance(cd, Mapping) and cd.get("enabled"):
        return True
    noise = clm.get("competing_noise")
    # Only text can be one of the values; any other type is the engine's to refuse.
    return any(isinstance(entry, Mapping)
               and isinstance(favors := entry.get(
                   "favors", default("clm_label.competing_noise.favors")), str)
               and favors in _PLACED_BY_DISTANCE
               for entry in (noise if isinstance(noise, (list, tuple)) else ()))


@dataclass(frozen=True)
class Setting:
    """One setting. `default` is a value, REQUIRED, or None when what an absent
    setting means is decided where it is read. `choices`, or `lo` and `hi`, are
    what the run accepts; `lo_strict` excludes `lo` itself. `missing_code` and
    `invalid_code` are the [CLM-###] codes for a required setting left out and a
    value refused. `applies_when` takes the clm_label block and says whether the
    run reads the setting; None: whenever its block is read. `kind` is checked by
    the command after loading; None: the engine checks it, with its code."""

    path: str
    default: Any = None
    choices: tuple[str, ...] | None = None
    lo: float | None = None
    hi: float | None = None
    lo_strict: bool = False
    missing_code: int | None = None
    invalid_code: int | None = None
    applies_when: Callable[[Mapping[str, Any]], bool] | None = None
    kind: Kind | None = None


_TREE: Final = (
    Setting("global_settings.data_source", "fabricated_data", kind="text"),
    Setting("global_settings.output_dir", "OUTPUT", kind="path"),

    # 0 or less writes no label.
    Setting("label_generation.n_labels", 1, hi=10_000, kind="whole"),
    Setting("label_generation.source_labeling", "labels0", kind="text"),
    # Label i is drawn with seed + i.
    Setting("label_generation.seed", 42, lo=0, kind="whole"),

    Setting("clm_label.num_classes", REQUIRED, lo=1, hi=64,
            missing_code=201, invalid_code=126),
    Setting("clm_label.balance", "balanced", choices=("balanced", "unbalanced"),
            invalid_code=132),
    Setting("clm_label.skew_rule", None,
            choices=("geometric", "dominant_minority", "dirichlet"),
            missing_code=203, invalid_code=107),
    Setting("clm_label.skew_params.ratio", 0.5, lo=0, invalid_code=131),
    Setting("clm_label.skew_params.dominant_index", 0, lo=0, invalid_code=131),
    Setting("clm_label.skew_params.dominant_share", 0.5, lo=0, hi=1, invalid_code=131),
    Setting("clm_label.skew_params.alpha", 1.0, lo=0, lo_strict=True, invalid_code=131),
    Setting("clm_label.matching_mode", REQUIRED,
            choices=("perfect", "single", "random", "custom"),
            missing_code=202, invalid_code=101),
    Setting("clm_label.split_rule", "proportional_to_size",
            choices=("proportional_to_size", "equal"), invalid_code=108,
            applies_when=lambda clm: clm.get("matching_mode") == "custom"),
    Setting("clm_label.spillover_rule", "proportional_to_marginal",
            choices=("proportional_to_marginal", "uniform", "concentrated"), invalid_code=109,
            applies_when=lambda clm: clm.get("matching_mode") != "random"),
    Setting("clm_label.competing_noise.share", 1.0, lo=0, hi=1, invalid_code=117),
    Setting("clm_label.competing_noise.favors", "boundary",
            choices=("boundary", "core", "random"), invalid_code=116),
    Setting("clm_label.target_metric.type", REQUIRED, choices=("mcc", "ari"),
            missing_code=112, invalid_code=112),
    Setting("clm_label.target_metric.value", REQUIRED, lo=-1, hi=1,
            missing_code=209, invalid_code=113),
    Setting("clm_label.target_metric.scope", "global", choices=("pair", "global"),
            invalid_code=122),
    # 0 under strict.
    Setting("clm_label.target_metric.tolerance", 0.01, lo=0, invalid_code=141),
    Setting("clm_label.target_metric.strict", False, invalid_code=141),
    Setting("clm_label.target_metric.max_iter", 40, lo=0, invalid_code=141,
            applies_when=lambda clm: _scope(clm) == "global"),
    # Absent, each label's search runs on that label's own seed.
    Setting("clm_label.target_metric.probe_seed", None, lo=0, invalid_code=141,
            applies_when=lambda clm: _scope(clm) == "global"),
    Setting("clm_label.centroid_dependence.enabled", False, invalid_code=141,
            applies_when=lambda clm: clm.get("matching_mode") != "random"),
    Setting("clm_label.centroid_dependence.favors", "core", choices=("core", "boundary"),
            invalid_code=129,
            applies_when=lambda clm: bool((clm.get("centroid_dependence") or {})
                                          .get("enabled"))),
    Setting("clm_label.centroid_dependence.profile", "linear",
            choices=("linear", "exponential", "step"), invalid_code=110,
            applies_when=_profile_read),
    Setting("clm_label.centroid_dependence.steepness", 3.0, lo=0, invalid_code=141,
            applies_when=lambda clm: _profile_read(clm) and (
                (clm.get("centroid_dependence") or {}).get(
                    "profile", default("clm_label.centroid_dependence.profile"))
                == "exponential")),
    Setting("clm_label.feature_dependence.favors", "high", choices=("high", "low", "value"),
            invalid_code=141),
    Setting("clm_label.feature_dependence.profile", "linear", choices=("linear", "step"),
            invalid_code=141),
)

SETTINGS: Final[Mapping[str, Setting]] = MappingProxyType({s.path: s for s in _TREE})


def _scope(clm: Mapping[str, Any]) -> Any:
    tm = clm.get("target_metric")
    return (tm.get("scope", default("clm_label.target_metric.scope"))
            if isinstance(tm, Mapping) else None)


def default(path: str) -> Any:
    """What the run uses when the setting at `path` is absent."""
    return SETTINGS[path].default


def applies(path: str, clm: Mapping[str, Any]) -> bool:
    """Whether the run reads the setting at `path` under the clm_label block `clm`."""
    condition = SETTINGS[path].applies_when
    return True if condition is None else condition(clm)


def choices(path: str) -> tuple[str, ...]:
    """The values the run accepts for the setting at `path`."""
    found = SETTINGS[path].choices
    if found is None:
        raise KeyError(f"{path} takes no fixed choices")
    return found


def in_bounds(path: str, value: Any) -> bool:
    """Whether a number is within the run's bounds for `path`. NaN never is."""
    s = SETTINGS[path]
    if value != value:
        return False
    above = s.lo is None or (value > s.lo if s.lo_strict else value >= s.lo)
    below = s.hi is None or value <= s.hi
    return bool(above and below)


def _leaves(prefix: str) -> frozenset[str]:
    """The names of the settings directly under `prefix`."""
    return frozenset(path[len(prefix):] for path in SETTINGS
                     if path.startswith(prefix) and "." not in path[len(prefix):])


# The clm_label settings an empty value leaves absent, by the block that holds them:
# "" for the block itself, a name for a mapping inside it or for each entry of a list.
_EMPTY_IS_ABSENT: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "": _leaves("clm_label.") - {"balance"},
    **{block: _leaves(f"clm_label.{block}.")
       for block in ("skew_params", "target_metric", "centroid_dependence",
                     "competing_noise", "feature_dependence")}})


def drop_empty(clm: Any) -> Any:
    """The clm_label block with every empty setting left out, so that it reads as
    absent: an optional setting takes its default and a required one reports its
    missing-key code. Its mappings become dicts. Anything malformed is returned as it
    stands, for the checks to refuse."""
    if not isinstance(clm, Mapping):
        return clm
    out = _without_empty(clm, _EMPTY_IS_ABSENT[""])
    for block in ("skew_params", "target_metric", "centroid_dependence", "single_match"):
        if isinstance(out.get(block), Mapping):
            out[block] = _without_empty(out[block], _EMPTY_IS_ABSENT.get(block, frozenset()))
    for block in ("assignment_matrix", "competing_noise", "feature_dependence"):
        entries = out.get(block)
        if isinstance(entries, (list, tuple)):
            keys = _EMPTY_IS_ABSENT.get(block, frozenset())
            out[block] = [_without_empty(e, keys) if isinstance(e, Mapping) else e
                          for e in entries]
    return out


def _without_empty(block: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if not (v is None and k in keys)}
