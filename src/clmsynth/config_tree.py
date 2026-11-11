# SPDX-License-Identifier: MIT
# config_tree.py
"""What the run accepts for each setting, and what it uses when the setting is
absent. One record per setting, in the order the run's config is written. Only
settings with a default, choices or bounds have a record."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

__all__ = ["REQUIRED", "SETTINGS", "Setting", "choices", "default", "in_bounds"]


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


# The default of a setting the run refuses to do without.
REQUIRED: Final = _Required()


@dataclass(frozen=True)
class Setting:
    """One setting. `default` is a value, REQUIRED, or None when what an absent
    setting means is decided where it is read. `choices`, or `lo` and `hi`, are
    what the run accepts; `lo_strict` excludes `lo` itself."""

    path: str
    default: Any = None
    choices: tuple[str, ...] | None = None
    lo: float | None = None
    hi: float | None = None
    lo_strict: bool = False


_TREE: Final = (
    Setting("global_settings.data_source", "fabricated_data"),
    Setting("global_settings.output_dir", "OUTPUT"),

    Setting("label_generation.n_labels", 1),
    Setting("label_generation.source_labeling", "labels0"),

    Setting("clm_label.num_classes", REQUIRED, lo=1, hi=64),
    Setting("clm_label.balance", "balanced", choices=("balanced", "unbalanced")),
    Setting("clm_label.skew_rule", None,
            choices=("geometric", "dominant_minority", "dirichlet")),
    Setting("clm_label.skew_params.ratio", 0.5, lo=0),
    Setting("clm_label.skew_params.dominant_index", 0, lo=0),
    Setting("clm_label.skew_params.dominant_share", 0.5, lo=0, hi=1),
    Setting("clm_label.skew_params.alpha", 1.0, lo=0, lo_strict=True),
    Setting("clm_label.matching_mode", REQUIRED,
            choices=("perfect", "single", "random", "custom")),
    Setting("clm_label.split_rule", "proportional_to_size",
            choices=("proportional_to_size", "equal")),
    Setting("clm_label.spillover_rule", "proportional_to_marginal",
            choices=("proportional_to_marginal", "uniform", "concentrated")),
    Setting("clm_label.competing_noise.share", 1.0, lo=0, hi=1),
    Setting("clm_label.competing_noise.favors", "boundary",
            choices=("core", "boundary", "random")),
    Setting("clm_label.target_metric.type", REQUIRED, choices=("mcc", "ari")),
    Setting("clm_label.target_metric.value", REQUIRED, lo=-1, hi=1),
    Setting("clm_label.target_metric.scope", "global", choices=("pair", "global")),
    # 0 under strict.
    Setting("clm_label.target_metric.tolerance", 0.01, lo=0),
    Setting("clm_label.target_metric.strict", False),
    Setting("clm_label.target_metric.max_iter", 40, lo=0),
    # Absent, each label's search runs on that label's own seed.
    Setting("clm_label.target_metric.probe_seed", None, lo=0),
    Setting("clm_label.centroid_dependence.enabled", False),
    Setting("clm_label.centroid_dependence.favors", "core", choices=("core", "boundary")),
    Setting("clm_label.centroid_dependence.profile", "linear",
            choices=("linear", "exponential", "step")),
    Setting("clm_label.centroid_dependence.steepness", 3.0, lo=0),
    Setting("clm_label.feature_dependence.favors", "high", choices=("high", "low", "value")),
    Setting("clm_label.feature_dependence.profile", "linear", choices=("linear", "step")),
)

SETTINGS: Final[Mapping[str, Setting]] = MappingProxyType({s.path: s for s in _TREE})


def default(path: str) -> Any:
    """What the run uses when the setting at `path` is absent."""
    return SETTINGS[path].default


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
