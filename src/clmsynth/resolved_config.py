# SPDX-License-Identifier: MIT
# resolved_config.py
"""The configuration a run used, as the run read it.

`resolve_config` rebuilds the config a run was given into the one it acted on:
every key the pipeline and the engine read, with its value or its default; the
seeds as numbers; `batteries` and `datasets` as the lists that ran. A key the run
never reads, or reads and ignores ([CLM-301], [CLM-302], [CLM-303]), is left out.
main.py writes the result into the run folder, and running that file again
reproduces the run's CSVs.

Only well-formed blocks are resolved. A missing required key stays missing and a
malformed value stays as written, so the engine refuses the resolved config
exactly as it refused the original.
"""

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from .clm_label_engine import BALANCE_VALUES, MATCHING_MODES

__all__ = [
    "BALANCE",
    "CENTROID",
    "DATA_SOURCE",
    "NOISE_ENTRY",
    "N_LABELS",
    "OUTPUT_DIR",
    "RESOLVED_CONFIG_NAME",
    "SKEW_PARAMS",
    "SOURCE_LABELING",
    "SPILLOVER_RULE",
    "SPLIT_RULE",
    "TARGET_METRIC",
    "plain",
    "resolve_config",
]

RESOLVED_CONFIG_NAME = "resolved_config.yaml"

# Pipeline defaults, read by main.py.
DATA_SOURCE = "clustbench"
OUTPUT_DIR = "OUTPUT"
N_LABELS = 0
SOURCE_LABELING = "labels0"

# Engine defaults, restated from clm_label_engine.
BALANCE = "balanced"
SPLIT_RULE = "proportional_to_size"
SPILLOVER_RULE = "proportional_to_marginal"
SKEW_PARAMS: Final[Mapping[str, Mapping[str, Any]]] = {
    "geometric": {"ratio": 0.5},
    "dominant_minority": {"dominant_index": 0, "dominant_share": 0.5},
    "dirichlet": {"alpha": 1.0},
}
CENTROID: Final[Mapping[str, Any]] = {"enabled": False, "profile": "linear", "favors": "core",
                                      "steepness": 3.0}
NOISE_ENTRY: Final[Mapping[str, Any]] = {"share": 1.0, "favors": "boundary"}
TARGET_METRIC: Final[Mapping[str, Any]] = {"scope": "global", "tolerance": 0.01, "max_iter": 40}


def resolve_config(config: Mapping[str, Any], source: str, batteries: list[str],
                   datasets_cfg: Any, jobs: list[tuple[str, str]], fetch_seed: int | None,
                   fetch_kwargs: Mapping[str, Any], label_seed: int) -> dict[str, Any]:
    """The config `run_pipeline` acted on, from the config it was given and the
    values it resolved: the batteries and (battery, dataset) jobs, the source seed
    (None for a source that reads none), the kwargs forwarded to the fetcher, and
    the label seed."""
    global_settings = config.get("global_settings") or {}
    output_dir = (global_settings.get("output_dir", OUTPUT_DIR)
                  if isinstance(global_settings, Mapping) else OUTPUT_DIR)

    # The names that ran, in run order. Each battery keeps the names it holds, so
    # one list reproduces every battery's selection. An empty list would read as
    # "all", so a selection that matched nothing stays as written.
    ran = list(dict.fromkeys(dataset for _, dataset in jobs))
    suite: dict[str, Any] = {"batteries": list(batteries), "datasets": ran or datasets_cfg}
    if fetch_seed is not None:
        suite["seed"] = fetch_seed
    # Forwarded to the fetcher as written; the fetchers' own defaults are not restated.
    suite.update(fetch_kwargs)

    return {
        "global_settings": {"data_source": source, "output_dir": output_dir},
        f"{source}_suite": suite,
        "label_generation": _resolve_label_generation(config.get("label_generation") or {},
                                                      label_seed),
    }


def _resolve_label_generation(label_cfg: Mapping[str, Any], label_seed: int) -> dict[str, Any]:
    n_labels = label_cfg.get("n_labels", N_LABELS)
    out: dict[str, Any] = {
        "n_labels": n_labels,
        "source_labeling": label_cfg.get("source_labeling", SOURCE_LABELING),
        "seed": label_seed,
    }
    # `noise` is accepted and never read. clm_label is read only to generate a
    # label; an n_labels of the wrong type keeps it, for the run to refuse.
    generates = not (_is_int(n_labels) and n_labels <= 0)
    if generates and "clm_label" in label_cfg:
        out["clm_label"] = _resolve_clm(label_cfg["clm_label"])
    return out


def _resolve_clm(clm: Any) -> Any:
    """The clm_label block as generate_clm_labels reads it for its matching_mode."""
    if not isinstance(clm, Mapping) or clm.get("matching_mode") not in MATCHING_MODES:
        return clm          # refused as it stands: [CLM-141], [CLM-202] or [CLM-101]
    mode = clm["matching_mode"]
    out = _present(clm, "num_classes")
    if mode != "perfect":   # 'perfect' sizes labels from the clusters ([CLM-302])
        out.update(_resolve_counts(clm))
    out["matching_mode"] = mode

    if mode == "random":
        # Only the refusals read past the counts: a competing_noise or target_metric
        # is refused ([CLM-115], [CLM-114]). centroid_dependence is not read ([CLM-311]).
        for key in ("competing_noise", "target_metric"):
            if clm.get(key):
                out[key] = clm[key]
        return out

    if mode == "single":
        sm = clm.get("single_match")
        if isinstance(sm, Mapping):
            out["single_match"] = _present(sm, "cluster", "label")
        elif sm is not None:
            out["single_match"] = sm
    tm = clm.get("target_metric")
    if mode == "custom":
        out.update(_resolve_matrix(clm, bool(tm)))
        out["split_rule"] = clm.get("split_rule", SPLIT_RULE)

    spill = clm.get("spillover_rule")
    if spill is None:       # absent or empty: the default runs
        spill = SPILLOVER_RULE
    out["spillover_rule"] = spill
    # Absent, it is the largest label of each dataset: decided per dataset, not here.
    if spill == "concentrated" and clm.get("concentrated_labels") is not None:
        out["concentrated_labels"] = clm["concentrated_labels"]

    noise = clm.get("competing_noise")
    noise_entries = _mappings(noise) if noise else []
    if noise:
        out["competing_noise"] = (
            [{**_present(e, "cluster", "label"),
              "share": e.get("share", NOISE_ENTRY["share"]),
              "favors": e.get("favors", NOISE_ENTRY["favors"])} for e in noise_entries]
            if noise_entries is not None else noise)
    if tm:
        out["target_metric"] = _resolve_target(tm) if mode != "perfect" else tm
    noise_placed = any(e.get("favors", NOISE_ENTRY["favors"]) in ("core", "boundary")
                       for e in noise_entries or [])
    out["centroid_dependence"] = _resolve_centroid(clm.get("centroid_dependence"), noise_placed)
    return out


def _resolve_counts(clm: Mapping[str, Any]) -> dict[str, Any]:
    """balance, then proportions or skew_rule, whichever resolve_label_counts reads."""
    balance = clm.get("balance", BALANCE)
    if balance not in BALANCE_VALUES:
        return _present(clm, "proportions", "balance", "skew_rule", "skew_params")  # [CLM-132]
    if balance == "balanced":
        return {"balance": balance}                          # proportions ignored: [CLM-301]
    if clm.get("proportions"):
        return {"proportions": clm["proportions"], "balance": balance}
    out: dict[str, Any] = {"balance": balance}
    params = clm.get("skew_params")
    params_ok = params is None or isinstance(params, dict)
    if "skew_rule" not in clm:
        # [CLM-203], unless skew_params is not a mapping: [CLM-131] is checked first.
        if not params_ok:
            out["skew_params"] = params
        return out
    rule = clm["skew_rule"]
    out["skew_rule"] = rule
    if params_ok and isinstance(rule, str) and rule in SKEW_PARAMS:
        given = params or {}
        out["skew_params"] = {k: given.get(k, v) for k, v in SKEW_PARAMS[rule].items()}
    elif "skew_params" in clm:
        out["skew_params"] = params                          # [CLM-131] or [CLM-107]
    return out


def _resolve_matrix(clm: Mapping[str, Any], solved: bool) -> dict[str, Any]:
    """assignment_matrix rows; recall_target is solved for under target_metric ([CLM-303])."""
    if "assignment_matrix" not in clm or clm["assignment_matrix"] is None:
        return {}                                            # [CLM-206]
    rows = _mappings(clm["assignment_matrix"])
    if rows is None:
        return {"assignment_matrix": clm["assignment_matrix"]}
    keys = ("clusters", "label") if solved else ("clusters", "label", "recall_target")
    return {"assignment_matrix": [_present(row, *keys) for row in rows]}


def _resolve_target(tm: Any) -> Any:
    """target_metric; max_iter and probe_seed are read by the global search only."""
    if not isinstance(tm, Mapping):
        return tm
    scope = tm.get("scope", TARGET_METRIC["scope"])
    if scope not in ("pair", "global"):
        return tm                                            # [CLM-122]
    out = _present(tm, "type", "value")
    out["scope"] = scope
    strict = isinstance(tm.get("strict"), (bool, np.bool_)) and bool(tm["strict"])
    out["tolerance"] = tm.get("tolerance", 0.0 if strict else TARGET_METRIC["tolerance"])
    out.update(_present(tm, "strict"))
    if scope == "global":
        out["max_iter"] = tm.get("max_iter", TARGET_METRIC["max_iter"])
        # Absent or empty, each label's search runs on that label's own seed: not
        # one number.
        if tm.get("probe_seed") is not None:
            out["probe_seed"] = tm["probe_seed"]
    return out


def _resolve_centroid(cd: Any, noise_placed: bool) -> Any:
    """centroid_dependence: favors when enabled, the profile when anything is
    placed by distance, steepness for the exponential profile only."""
    if cd and not isinstance(cd, Mapping):
        return cd                                            # [CLM-141]
    cd = cd or {}
    enabled = cd.get("enabled", CENTROID["enabled"])
    out: dict[str, Any] = {"enabled": enabled}
    if enabled:
        out["favors"] = cd.get("favors", CENTROID["favors"])
    if enabled or noise_placed:
        out["profile"] = cd.get("profile", CENTROID["profile"])
        if out["profile"] == "exponential":
            out["steepness"] = cd.get("steepness", CENTROID["steepness"])
    return out


def _present(block: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    """The given keys that `block` has, in that order."""
    return {k: block[k] for k in keys if k in block}


def _mappings(value: Any) -> list[Mapping[str, Any]] | None:
    """A list block's entries when it is a list of mappings, else None."""
    if isinstance(value, (list, tuple)) and all(isinstance(e, Mapping) for e in value):
        return list(value)
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def plain(value: Any) -> Any:
    """`value` with numpy scalars and arrays, tuples and mappings turned into the
    Python types a YAML or JSON writer accepts. A dict run may pass any of them."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
