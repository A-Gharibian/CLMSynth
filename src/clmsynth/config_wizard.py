# config_wizard.py
"""
Configuration Generator (CLI wizard).
    python -m clmsynth.config_wizard      (or the `clmsynth-wizard` console script)

Run as a module, not as a file.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .dataset_sources import SOURCE_DATASETS, SOURCE_METADATA, is_heavy
from .questions import SCHEMA

# Windows' MAX_PATH. Duplicated here on purpose rather than imported from
# visualization.py: that module pulls matplotlib and seaborn, and importing the
# constant would drag both into this module's import graph, and therefore into
# any help command built on the same schema. test_07_text_wizard asserts neither
# is imported. Keep the value in step with visualization._MAX_PATH by hand.
_MAX_PATH = 260

# --------------------------------------------------------------------------- #
# Input helpers: each explains, shows a [default], and re-asks on bad input.
# --------------------------------------------------------------------------- #

def _explain(text):
    if text:
        print("\n" + text)


def _read(prompt, default) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    raw = input(f"  {prompt}{suffix}: ").strip()
    return raw if raw else ("" if default is None else str(default))


def ask_str(prompt, default=None, explain=None) -> str:
    """Asks for a non-empty string."""
    _explain(explain)
    while True:
        v = _read(prompt, default)
        if v:
            return v
        print("    (a value is required)")


def ask_int(prompt, default=None, minv=None, maxv=None, explain=None) -> int:
    """Asks for a whole number, optionally bounded by `minv` and/or `maxv`."""
    _explain(explain)
    while True:
        v = _read(prompt, default)
        try:
            i = int(v)
        except ValueError:
            print("    (enter a whole number)")
            continue
        if minv is not None and i < minv:
            print(f"    (must be at least {minv})")
            continue
        if maxv is not None and i > maxv:
            print(f"    (must be at most {maxv})")
            continue
        return i


def ask_float(prompt, default=None, lo=None, hi=None, explain=None, lo_strict=False) -> float:
    """Asks for a number, optionally bounded to [lo, hi]. With `lo_strict` the
    lower bound is exclusive (value must be strictly greater than `lo`), which is
    what the engine wants for a dirichlet `alpha` (exactly 0 divides by zero)."""
    _explain(explain)
    while True:
        v = _read(prompt, default)
        try:
            f = float(v)
        except ValueError:
            print("    (enter a number)")
            continue
        below = lo is not None and (f <= lo if lo_strict else f < lo)
        if below or (hi is not None and f > hi):
            print(f"    (must be between {lo} and {hi})")
            continue
        return f


def ask_bool(prompt, default=True, explain=None) -> bool:
    """Asks a yes/no question."""
    _explain(explain)
    while True:
        v = _read(prompt + " (yes/no)", "yes" if default else "no").lower()
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False
        print("    (answer yes or no)")


def ask_choice(prompt, choices, default=None, explain=None) -> str:
    """Asks to pick one of `choices`, by number or by name."""
    _explain(explain)
    for i, c in enumerate(choices, 1):
        print(f"    {i}) {c}")
    while True:
        v = _read(prompt, default)
        if v in choices:
            return v
        if v.isdigit() and 1 <= int(v) <= len(choices):
            return choices[int(v) - 1]
        print(f"    (choose 1-{len(choices)} or type the name)")


def _parse_ids(raw):
    """Cluster ids may be ints or strings; keep each as int if possible."""
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            out.append(tok)
    return out


def ask_cluster_ids(prompt, explain=None) -> list:
    """Asks for one or more cluster ids, re-asking on empty input.
    """
    _explain(explain)
    while True:
        ids = _parse_ids(input(f"  {prompt}: "))
        if ids:
            return ids
        print("    (at least one cluster id is required)")


def ask_cluster_id(prompt, explain=None):
    """Asks for exactly ONE cluster id; extra ids are ignored."""
    return ask_cluster_ids(prompt, explain)[0]


def ask_ints(prompt, explain=None) -> list[int]:
    """Asks for a comma-separated list of whole numbers (e.g. label ids)."""
    _explain(explain)
    while True:
        raw = input(f"  {prompt} (comma-separated): ").strip()
        try:
            vals = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("    (enter whole numbers separated by commas)")
            continue
        if vals:
            return vals
        print("    (at least one value is required)")


def ask_floats(prompt, explain=None) -> list[float]:
    """Asks for a comma-separated list of numbers."""
    _explain(explain)
    while True:
        raw = input(f"  {prompt} (comma-separated): ").strip()
        try:
            return [float(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("    (enter numbers separated by commas)")


def ask_strs(prompt, explain=None) -> list[str]:
    """Comma-separated list of names."""
    _explain(explain)
    while True:
        raw = input(f"  {prompt} (comma-separated): ")
        vals = [s.strip() for s in raw.split(",") if s.strip()]
        if vals:
            return vals
        print("    (at least one name is required)")


def ask_selection(prompt, options, default=None, explain=None):
    """Picks by number, name, or 'all'."""
    _explain(explain)
    while True:
        raw = _read(prompt, default)
        if raw == "all":
            return "all"
        picked, unknown = [], []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit() and 1 <= int(tok) <= len(options):
                picked.append(options[int(tok) - 1])
            elif tok in options:
                picked.append(tok)
            else:
                unknown.append(tok)
        if unknown:
            print(f"    (not in this group: {', '.join(unknown)})")
            continue
        if picked:
            return picked
        print(f"    (choose 1-{len(options)}, name(s), or 'all')")


def ask_from(key, **override):
    """Drives the schema question named `key` through the matching prompt helper.

    Prompt, explain text, default, bounds and choices come from `questions.SCHEMA`;
    the wizard keeps the control flow. `override` supplies the runtime-dependent
    bits a static schema cannot carry, a formatted `prompt`, a `default` computed
    from the data, or a `maxv`/`hi` that depends on M.
    """
    q = SCHEMA[key]
    prompt = override.get("prompt", q.prompt)
    default = override.get("default", q.default)
    explain = override.get("explain", q.explain)
    if q.kind == "str":
        return ask_str(prompt, default, explain=explain)
    if q.kind == "int":
        return ask_int(prompt, default, minv=override.get("minv", q.lo),
                       maxv=override.get("maxv", q.hi), explain=explain)
    if q.kind == "float":
        return ask_float(prompt, default, lo=override.get("lo", q.lo),
                         hi=override.get("hi", q.hi), explain=explain, lo_strict=q.lo_strict)
    if q.kind == "bool":
        return ask_bool(prompt, default, explain=explain)
    if q.kind == "choice":
        return ask_choice(prompt, q.choices, default, explain=explain)
    if q.kind == "int_list":
        return ask_ints(prompt, explain=explain)
    if q.kind == "float_list":
        return ask_floats(prompt, explain=explain)
    if q.kind == "str_list":
        return ask_strs(prompt, explain=explain)
    if q.kind == "selection":
        return ask_selection(prompt, override["options"], default, explain=explain)
    if q.kind == "id":
        return ask_cluster_id(prompt, explain=explain)
    if q.kind == "ids":
        return ask_cluster_ids(prompt, explain=explain)
    raise ValueError(f"unknown question kind {q.kind!r} for {key!r}")  # pragma: no cover


def _worst_case_path_len(gs, source, suite) -> int:
    """Length of the longest file a run of this config could write:

        <output_dir>/DDMMYY_<source>_HHMMSS/png/<source>__<battery>__<dataset>__Label_<n>.png

    Plot names are the longest and so cross Windows' MAX_PATH first. Pure string
    arithmetic on values already in hand, no fetch and no run, which is what keeps
    the check inside the wizard's rule-based remit."""
    base = len(str(Path(gs["output_dir"]).resolve()))
    run_stem = f"DDMMYY_{source}_HHMMSS"
    batteries = suite.get("batteries")
    if batteries == "all" or batteries is None:
        batteries = list(SOURCE_DATASETS.get(source, {}).keys())
    datasets = suite.get("datasets")
    if datasets == "all":
        names = [d for b in batteries for d in SOURCE_DATASETS.get(source, {}).get(b, [])]
    elif isinstance(datasets, list):
        names = datasets
    else:
        names = []
    longest_ds = max((str(n) for n in names), key=len, default="dataset")
    longest_bat = max((str(b) for b in batteries), key=len, default="battery")
    leaf = f"png/{source}__{longest_bat}__{longest_ds}__Label_9.png"
    return base + 1 + len(run_stem) + 1 + len(leaf)


def _warn_if_paths_may_be_too_long(gs, source, suite) -> None:
    """Warns, before anything is written, if the deepest output path could pass
    Windows' MAX_PATH. A warning, not a refusal: the path is legitimate on other
    platforms, and on Windows with long-path support enabled."""
    try:
        total = _worst_case_path_len(gs, source, suite)
    except Exception:                        # best-effort guidance, never blocks the wizard
        return
    if total > _MAX_PATH:
        print(f"\n  Note: the longest file this run could write is about {total} characters,\n"
              f"  past Windows' {_MAX_PATH}-character path limit. Plots (.png) have the longest\n"
              "  names and would fail first, while CSV and TXT still succeed. Shorten\n"
              "  'output_dir' or move it nearer the drive root to stay under the limit.")


def section(title) -> None:
    """Prints a banner separating the wizard's numbered sections."""
    print("\n" + "=" * 62 + f"\n{title}\n" + "=" * 62)


def _peek_cluster_count(suite):
    """Best-effort: read the first CSV and count clusters, to guide the user
    (e.g. so 'perfect' mode gets the right number of labels). Silent on failure."""
    try:
        import pandas as pd
        name = (suite.get("datasets") or [None])[0]
        idir = suite.get("input_dir")
        path = (Path(idir) / f"{name}.csv") if idir else Path(f"{name}.csv")
        col = suite.get("cluster_column")
        return int(pd.read_csv(path, usecols=[col])[col].nunique())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1. Data source
# --------------------------------------------------------------------------- #

def build_source():
    """Wizard section 1: choose the data source and its suite settings."""
    section("1. Data source, where the clusters come from")
    source = ask_from("global_settings.data_source")
    gs = {"data_source": source, "output_dir": ask_from("global_settings.output_dir")}
    suite = _byoc_suite() if source == "byoc" else _registry_suite(source)
    _warn_if_paths_may_be_too_long(gs, source, suite)
    known_k = _peek_cluster_count(suite) if source == "byoc" else None
    return source, gs, suite, known_k


def _byoc_suite():
    input_dir = ask_from("byoc_suite.input_dir")
    datasets = ask_from("byoc_suite.datasets")
    cluster_column = ask_from("byoc_suite.cluster_column")
    standardize = ask_from("byoc_suite.standardize")
    seed = ask_from("byoc_suite.seed")
    return {"batteries": ["local"], "input_dir": input_dir, "datasets": datasets,
            "cluster_column": cluster_column, "standardize": standardize, "seed": seed}


def _registry_suite(source):
    if source == "mdcgen":
        print("\n  (mdcgen needs the 'mdcgenpy' package installed; see the README.)")
    batteries = list(SOURCE_DATASETS.get(source, {}).keys())
    print(f"\nAvailable '{source}' groups:")
    for i, b in enumerate(batteries, 1):
        desc = SOURCE_METADATA.get(source, {}).get(b, {}).get("description", "")
        heavy = "  [HEAVY: slow/large]" if is_heavy(source, b) else ""
        print(f"    {i}) {b}{heavy}")
        if desc:
            print(f"         {desc}")
    while True:
        raw = _read("Pick a group by number/name, or 'all'", "1")
        if raw == "all":
            return {"batteries": "all", "datasets": "all", "seed": ask_int("Random seed", 42)}
        if raw in batteries:
            battery = raw
            break
        if raw.isdigit() and 1 <= int(raw) <= len(batteries):
            battery = batteries[int(raw) - 1]
            break
        print(f"    (choose 1-{len(batteries)}, a name, or 'all')")

    ds = SOURCE_DATASETS[source][battery]
    print(f"\nDatasets in '{battery}' ({len(ds)} total):")
    for i, d in enumerate(ds[:40], 1):
        print(f"    {i}) {d}")
    if len(ds) > 40:
        print(f"    ... and {len(ds) - 40} more")
    datasets = ask_from("registry_suite.datasets", options=ds)
    seed = ask_from("registry_suite.seed")
    return {"batteries": [battery], "datasets": datasets, "seed": seed}


# --------------------------------------------------------------------------- #
# 2. Label generation
# --------------------------------------------------------------------------- #

def build_label_generation(source) -> dict[str, Any]:
    """Wizard section 2: label count, source labeling, and seed."""
    section("2. Labels, how many and against which clusters")
    n_labels = ask_from("label_generation.n_labels")
    if source == "byoc":
        # byoc always stores your single cluster column internally as 'labels0',
        # so there is nothing to pick here, asking again only invites the
        # mistake of re-typing the CSV column name.
        source_labeling = "labels0"
        print("\n  (Your cluster column is the ground truth; nothing more to choose here.)")
    else:
        source_labeling = ask_from("label_generation.source_labeling")
    seed = ask_from("label_generation.seed")
    return {"n_labels": n_labels, "source_labeling": source_labeling, "noise": 0.1, "seed": seed}


# --------------------------------------------------------------------------- #
# 3. Cluster-label matching (clm_label)
# --------------------------------------------------------------------------- #

def _final_check(clm: dict[str, Any]) -> None:
    tm = clm.get("target_metric") or {}
    lstar = (clm.get("single_match") or {}).get("label")
    try:
        if (clm.get("matching_mode") == "single"
                and tm.get("type") == "mcc"
                and tm.get("scope") == "pair"
                and float(tm.get("value", 0)) == 1.0
                and clm.get("spillover_rule", "proportional_to_marginal") == "proportional_to_marginal"
                and not any(e.get("label") == lstar
                            for e in (clm.get("competing_noise") or []))):
            print("NFL-169")
    except (TypeError, ValueError):
        return


def build_clm(known_k=None) -> dict[str, Any]:
    """Wizard section 3: the clm_label block (mode, balance, rules, extras)."""
    section("3. Cluster-label matching, the core settings")
    if known_k:
        print(f"\n  (Your data has {known_k} clusters.)")
    M = ask_from("clm_label.num_classes")
    mode = ask_from("clm_label.matching_mode")
    clm: dict[str, Any] = {"num_classes": M, "matching_mode": mode}

    if mode == "perfect":
        if known_k and M != known_k:
            print(f"\n  Note: perfect mode needs M = your cluster count ({known_k}); setting M = {known_k}.")
            clm["num_classes"] = known_k
        else:
            print("\n  (perfect copies the clusters exactly; label sizes come from the clusters,\n"
                  "   so balance/proportions do not apply. It errors if M != the cluster count.)")
    else:
        _add_balance(clm, clm["num_classes"])

    use_target = False
    if mode in ("single", "custom"):
        use_target = ask_from("clm_label.target_metric._enabled")
        if use_target:
            ttype = ask_from("clm_label.target_metric.type")
            tm = {"type": ttype, "value": ask_from("clm_label.target_metric.value")}
            # For an MCC target in 'single' mode you can score just your one chosen
            # cluster-label pair (exact and instant) or the whole labeling (the usual
            # multiclass score, found by search).
            if ttype == "mcc" and mode == "single":
                tm["scope"] = ask_from("clm_label.target_metric.scope")
            # Asked for BOTH scopes: tolerance is the band the delivered labeling
            # is checked against afterward, not only the global solver's
            # convergence test, so the pair scope reads it too.
            tm["tolerance"] = ask_from("clm_label.target_metric.tolerance")
            clm["target_metric"] = tm

    if mode == "single":
        clm["single_match"] = {
            "cluster": ask_from("clm_label.single_match.cluster"),
            # Cap at M-1 here; the engine's [CLM-104] aborts the whole run.
            "label": ask_from("clm_label.single_match.label", maxv=M - 1),
        }
    elif mode == "custom":
        clm["assignment_matrix"] = _build_rules(omit_recall=use_target, M=M)
        # split_rule only bites when a rule spans more than one cluster, which
        # 'single' never does, so it stays a custom-only question.
        clm["split_rule"] = ask_from("clm_label.split_rule")

    # Spillover applies to BOTH single and custom: either way the rules leave
    # unclaimed points behind. It used to be asked only under 'custom', so a
    # 'single' run silently took the default.
    if mode in ("single", "custom"):
        if (clm.get("target_metric") or {}).get("scope") == "pair":
            # scope: pair sizes the target label to sit entirely inside its cluster;
            # only proportional_to_marginal keeps its leftover pool empty. uniform or
            # concentrated could place the label outside the cluster and break the
            # closed form ([CLM-130]), so it is pinned rather than offered.
            clm["spillover_rule"] = "proportional_to_marginal"
            print("\n  (Spillover fixed to 'proportional_to_marginal': a pair-MCC target keeps\n"
                  "   the target label entirely inside its cluster.)")
        else:
            clm["spillover_rule"] = ask_from("clm_label.spillover_rule")
            if clm["spillover_rule"] == "concentrated":
                clm["concentrated_labels"] = ask_from("clm_label.concentrated_labels")

    if mode in ("single", "custom") and ask_from("clm_label.competing_noise._enabled"):
        entries: list[dict[str, Any]] = []
        while True:
            print(f"\n  Competing-noise entry {len(entries) + 1}:")
            entries.append({
                "cluster": ask_from("clm_label.competing_noise.cluster"),
                "label": ask_from("clm_label.competing_noise.label", maxv=M - 1),
                "share": ask_from("clm_label.competing_noise.share"),
                "favors": ask_from("clm_label.competing_noise.favors"),
            })
            if not ask_from("clm_label.competing_noise._add_another"):
                break
        clm["competing_noise"] = entries

    if mode != "random" and ask_from("clm_label.centroid_dependence._enabled"):
        prof = ask_from("clm_label.centroid_dependence.profile")
        cd = {"enabled": True, "profile": prof,
              "favors": ask_from("clm_label.centroid_dependence.favors")}
        if prof == "exponential":
            cd["steepness"] = ask_from("clm_label.centroid_dependence.steepness")
        clm["centroid_dependence"] = cd

    _final_check(clm)
    return clm


def _add_balance(clm, M):
    balance = ask_from("clm_label.balance")
    clm["balance"] = balance
    if balance != "unbalanced":
        return
    if ask_from("clm_label._explicit_proportions"):
        while True:
            props = ask_from("clm_label.proportions", prompt=f"Fraction for each of the {M} labels")
            if len(props) == M and abs(sum(props) - 1.0) < 1e-6:
                clm["proportions"] = props
                return
            print(f"    (need exactly {M} numbers that add up to 1.0)")
    rule = ask_from("clm_label.skew_rule")
    clm["skew_rule"] = rule
    if rule == "geometric":
        clm["skew_params"] = {"ratio": ask_from("clm_label.skew_params.ratio")}
    elif rule == "dominant_minority":
        clm["skew_params"] = {
            "dominant_index": ask_from("clm_label.skew_params.dominant_index", maxv=M - 1),
            "dominant_share": ask_from("clm_label.skew_params.dominant_share")}
    else:
        clm["skew_params"] = {"alpha": ask_from("clm_label.skew_params.alpha")}


def _build_rules(omit_recall, M):
    print("\n  Custom rules: each rule sends a share of ONE label into chosen cluster(s).")
    n = ask_from("clm_label.assignment_matrix._count")
    rules = []
    for i in range(n):
        print(f"\n  Rule {i + 1}:")
        rule: dict[str, Any] = {
            "label": ask_from("clm_label.assignment_matrix.label", maxv=M - 1),
            "clusters": ask_from("clm_label.assignment_matrix.clusters")}
        if not omit_recall:
            rule["recall_target"] = ask_from("clm_label.assignment_matrix.recall_target")
        rules.append(rule)
    return rules


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _run() -> None:
    """Runs the wizard: all sections, save the YAML, optionally run main.py."""
    print("=" * 62)
    print("  CLMSynth, User-Friendly Configuration Generator")
    print("=" * 62)
    print("\n  What this does: your data already has CLUSTERS. This tool adds a SECOND")
    print("  label on top and lets you dial how strongly it agrees with those clusters")
    print(" , from an exact copy (score MCC = 1.0) down to no relationship (MCC ~ 0).")
    print("  You choose how many labels, how big each is, and where they sit.")
    print("\n  Press Enter to accept the [default] shown in brackets.")

    source, gs, suite, known_k = build_source()
    lg = build_label_generation(source)
    lg["clm_label"] = build_clm(known_k)
    config = {"global_settings": gs, f"{source}_suite": suite, "label_generation": lg}

    section("4. Save & run")
    out = ask_from("output.path")
    if not out.lower().endswith((".yaml", ".yml")):
        out += ".yaml"                       # always a .yaml file, distinct from any output folder
    if Path(out).resolve() == Path(gs["output_dir"]).resolve():
        out = "config_" + out                # last-ditch guard against a folder/file name clash
    Path(out).write_text(yaml.dump(config, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"\n  Wrote '{out}'.")
    print(f"  Run it any time with:  python -m clmsynth.main {out}")
    if ask_from("run.now"):
        # check=False, deliberately: the pipeline reports its own failures with a
        # coded message, and a CalledProcessError traceback on top of that would
        # bury it. But the exit code was previously discarded entirely, so a run
        # that failed looked exactly like one that succeeded.
        result = subprocess.run([sys.executable, "-m", "clmsynth.main", out], check=False)
        if result.returncode != 0:
            print(f"\n  The run exited with code {result.returncode}; see the messages above. "
                  f"Your configuration is saved, so you can edit '{out}' and retry with:"
                  f"\n      python -m clmsynth.main {out}")


def main() -> None:
    """Entry point; Ctrl-C cancels cleanly."""
    try:
        _run()
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None
    except EOFError:
        print("\nCancelled.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
