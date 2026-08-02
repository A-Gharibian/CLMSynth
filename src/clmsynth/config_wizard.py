# config_wizard.py
"""
Configuration Generator (CLI wizard).
    python -m clmsynth.config_wizard      (or the `clmsynth-wizard` console script)

Run as a module, not as a file.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .dataset_sources import SOURCE_DATASETS, SOURCE_METADATA, is_heavy


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


def ask_int(prompt, default=None, minv=None, explain=None) -> int:
    """Asks for a whole number, optionally bounded below by `minv`."""
    _explain(explain)
    while True:
        v = _read(prompt, default)
        try:
            i = int(v)
        except ValueError:
            print("    (enter a whole number)"); continue
        if minv is not None and i < minv:
            print(f"    (must be at least {minv})"); continue
        return i


def ask_float(prompt, default=None, lo=None, hi=None, explain=None) -> float:
    """Asks for a number, optionally bounded to [lo, hi]."""
    _explain(explain)
    while True:
        v = _read(prompt, default)
        try:
            f = float(v)
        except ValueError:
            print("    (enter a number)"); continue
        if (lo is not None and f < lo) or (hi is not None and f > hi):
            print(f"    (must be between {lo} and {hi})"); continue
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


def ask_cluster_ids(prompt, explain=None) -> List:
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


def ask_ints(prompt, explain=None) -> List[int]:
    """Asks for a comma-separated list of whole numbers (e.g. label ids)."""
    _explain(explain)
    while True:
        raw = input(f"  {prompt} (comma-separated): ").strip()
        try:
            vals = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("    (enter whole numbers separated by commas)"); continue
        if vals:
            return vals
        print("    (at least one value is required)")


def ask_floats(prompt, explain=None) -> List[float]:
    """Asks for a comma-separated list of numbers."""
    _explain(explain)
    while True:
        raw = input(f"  {prompt} (comma-separated): ").strip()
        try:
            return [float(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("    (enter numbers separated by commas)")


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
    source = ask_choice(
        "Data source", ["clustbench", "mdcgen", "fabricated_data", "byoc"], "clustbench",
        explain="Choose where the base clusters come from:\n"
                "  clustbench      : real benchmark datasets downloaded online (Gagolewski suite)\n"
                "  mdcgen          : synthetic clusters generated on the fly (needs the mdcgenpy package)\n"
                "  fabricated_data : tiny offline example data (no internet, no extra packages)\n"
                "  byoc            : your OWN CSV file (features + one cluster-id column)")
    gs = {"data_source": source,
          "output_dir": ask_str("Where to save results (folder)", "OUTPUT",
                                 explain="Where results go. Usually just press Enter to keep 'OUTPUT';\n"
                                         "each run still gets its own timestamped subfolder inside it.\n"
                                         "(Pick a plain folder name, not the same name as a file.)")}
    suite = _byoc_suite() if source == "byoc" else _registry_suite(source)
    known_k = _peek_cluster_count(suite) if source == "byoc" else None
    return source, gs, suite, known_k


def _byoc_suite():
    input_dir = ask_str("Folder that holds your CSV file(s)", "INPUT",
                        explain="Bring-your-own-clusters: point at your CSV(s).")
    raw = input("  CSV file name(s) WITHOUT '.csv', comma-separated: ").strip()
    datasets = [s.strip() for s in raw.split(",") if s.strip()] or ["my_clusters"]
    cluster_column = ask_str("Name of the single cluster-id column", "cluster",
                             explain="Exactly ONE column holds the cluster id of each row.\n"
                                     "Every other numeric column is treated as a feature.")
    standardize = ask_bool("Rescale features to 0..1 on import?", default=False,
                           explain="Min-max standardization puts every feature on the same 0..1 scale.\n"
                                   "Use it when your features have very different units/ranges so no\n"
                                   "single one dominates the geometry. It rescales the saved CSV too.")
    seed = ask_int("Random seed", 42, explain="Same seed -> same result every run.")
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
            battery = raw; break
        if raw.isdigit() and 1 <= int(raw) <= len(batteries):
            battery = batteries[int(raw) - 1]; break
        print(f"    (choose 1-{len(batteries)}, a name, or 'all')")

    ds = SOURCE_DATASETS[source][battery]
    print(f"\nDatasets in '{battery}' ({len(ds)} total):")
    for i, d in enumerate(ds[:40], 1):
        print(f"    {i}) {d}")
    if len(ds) > 40:
        print(f"    ... and {len(ds) - 40} more")
    raw = _read("Pick dataset(s) by number/name, comma-separated, or 'all'", "all")
    if raw == "all":
        datasets = "all"
    else:
        datasets = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok.isdigit() and 1 <= int(tok) <= len(ds):
                datasets.append(ds[int(tok) - 1])
            elif tok in ds:
                datasets.append(tok)
        datasets = datasets or "all"
    seed = ask_int("Random seed", 42, explain="Used by mdcgen/fabricated_data; ignored by clustbench.")
    return {"batteries": [battery], "datasets": datasets, "seed": seed}


# --------------------------------------------------------------------------- #
# 2. Label generation
# --------------------------------------------------------------------------- #

def build_label_generation(source) -> Dict[str, Any]:
    """Wizard section 2: label count, source labeling, and seed."""
    section("2. Labels, how many and against which clusters")
    n_labels = ask_int("How many synthetic labels to generate", 1, minv=1,
                       explain="Each makes one Label_0, Label_1, ... column (a different random\n"
                               "seed each), so you can compare several labellings of the same data.")
    if source == "byoc":
        # byoc always stores your single cluster column internally as 'labels0',
        # so there is nothing to pick here, asking again only invites the
        # mistake of re-typing the CSV column name.
        source_labeling = "labels0"
        print("\n  (Your cluster column is the ground truth; nothing more to choose here.)")
    else:
        source_labeling = ask_str("Which ground-truth labeling to match", "labels0",
                                  explain="Most datasets have 'labels0'. Leave as-is unless you "
                                          "know there are more (e.g. 'labels1').")
    seed = ask_int("Random seed for label generation", 42)
    return {"n_labels": n_labels, "source_labeling": source_labeling, "noise": 0.1, "seed": seed}


# --------------------------------------------------------------------------- #
# 3. Cluster-label matching (clm_label)
# --------------------------------------------------------------------------- #

def _final_check(clm: Dict[str, Any]) -> None:
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


def build_clm(known_k=None) -> Dict[str, Any]:
    """Wizard section 3: the clm_label block (mode, balance, rules, extras)."""
    section("3. Cluster-label matching, the core settings")
    if known_k:
        print(f"\n  (Your data has {known_k} clusters.)")
    M = ask_int("Number of labels (M)", 3, minv=2,
                explain="How many DIFFERENT label values to create. This is NOT the total number\n"
                        "of labels, every datapoint always gets a label, it's how many classes\n"
                        "they split into (e.g. 3 -> values 0, 1, 2), sized to your whole dataset.\n"
                        "At least 2: one label for everything has no matching to measure, and\n"
                        "some modes/skews are undefined at M=1.")
    mode = ask_choice(
        "Matching mode", ["perfect", "single", "random", "custom"], "custom",
        explain="How should the new label relate to your clusters?\n"
                "  perfect : an exact copy, one label per cluster (score MCC = 1.0).\n"
                "            Needs M = your cluster count.\n"
                "  single  : ONE cluster becomes one label; all other points are unrelated.\n"
                "  random  : the label ignores the clusters completely (MCC ~ 0), a baseline.\n"
                "  custom  : you write rules (send a share of a label into chosen clusters).\n"
                "            Use for partial/realistic agreement, or to aim at a target score.")
    clm: Dict[str, Any] = {"num_classes": M, "matching_mode": mode}

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
        use_target = ask_bool("Aim for a specific agreement score instead of setting it by hand?",
                              default=False,
                              explain="Normally YOU set how much the label matches (recall, below).\n"
                                      "Say yes to instead name a target MCC/ARI and let the tool solve\n"
                                      "for it. Caveat: a target can be impossible for your data (e.g.\n"
                                      "MCC=1 with fewer labels than clusters), then it gets as close\n"
                                      "as it can and tells you it fell short.\n"
                                      "Note: the spillover and competing-noise choices you make later\n"
                                      "are held fixed and still shape the recall the solver lands on.")
        if use_target:
            ttype = ask_choice("Target score", ["mcc", "ari"], "mcc",
                               explain="Both measure agreement (1 = identical, 0 = unrelated);\n"
                                       "either is fine, mcc is the more common choice here.")
            tm = {"type": ttype, "value": ask_float("Target value (0..1)", 0.6, lo=-1.0, hi=1.0)}
            # For an MCC target in 'single' mode you can score just your one chosen
            # cluster-label pair (exact and instant) or the whole labeling (the usual
            # multiclass score, found by search).
            if ttype == "mcc" and mode == "single":
                tm["scope"] = ask_choice(
                    "Measure that MCC on", ["pair", "global"], "global",
                    explain="How much of the picture should the score judge?\n"
                            "  pair   : only your one chosen cluster and its label, each against\n"
                            "           everything else. The tool hits this exactly and instantly.\n"
                            "  global : the whole set of labels against all the clusters at once\n"
                            "           (the standard multiclass MCC). Found by search; it can\n"
                            "           fall short if the value is impossible for your data.")
            if tm.get("scope") != "pair":
                tm["tolerance"] = ask_float("How close counts as 'reached'", 0.01, lo=0.0, hi=1.0)
            clm["target_metric"] = tm

    if mode == "single":
        clm["single_match"] = {
            "cluster": ask_cluster_id("Which cluster id becomes the aligned label? "
                                      "(an id from your cluster column, e.g. 1)"),
            "label": ask_int("Which label value (0..M-1) it becomes", 0, minv=0),
        }
    elif mode == "custom":
        clm["assignment_matrix"] = _build_rules(omit_recall=use_target)
        # split_rule only bites when a rule spans more than one cluster, which
        # 'single' never does, so it stays a custom-only question.
        clm["split_rule"] = ask_choice(
            "If a rule targets several clusters, how to split the label between them",
            ["proportional_to_size", "equal"], "proportional_to_size",
            explain="proportional_to_size : bigger clusters get more of the label (usual choice).\n"
                    "equal                : each targeted cluster gets the same amount.")

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
            clm["spillover_rule"] = ask_choice(
                "After the rules, how to fill the leftover points",
                ["proportional_to_marginal", "uniform", "concentrated"], "proportional_to_marginal",
                explain="Rules rarely use every point; the rest still need a label:\n"
                        "  proportional_to_marginal : fill them so final sizes EXACTLY match your\n"
                        "                             proportions (recommended, keeps your split).\n"
                        "  uniform                  : spread evenly, this CHANGES your proportions.\n"
                        "  concentrated             : dump all leftovers into one label.")
            if clm["spillover_rule"] == "concentrated":
                clm["concentrated_labels"] = ask_ints(
                    "Which label value(s) should absorb the leftovers",
                    explain="All leftover points go to these labels. Leave one value for the\n"
                            "usual case; the default without this is the single largest label.")

    if mode in ("single", "custom") and ask_bool(
            "Give one cluster's leftover points a specific competing label (structured noise)?",
            default=False,
            explain="Optional 'structured noise': points NOT claimed by your rules normally get\n"
                    "filler labels from the spillover rule. This instead forces a share of ONE\n"
                    "cluster's leftover points to a single competing label, placed at that\n"
                    "cluster's edge (or center). Consequences to be aware of:\n"
                    "  - it BYPASSES your proportions: final label sizes will no longer match\n"
                    "    them (same caveat as the uniform/concentrated spillover rules);\n"
                    "  - it CHANGES the agreement score: structured noise scores differently\n"
                    "    from random noise, comparing the two is exactly what it is for;\n"
                    "  - it only shapes points not already claimed by the rules above."):
        entries = []
        while True:
            print(f"\n  Competing-noise entry {len(entries) + 1}:")
            entries.append({
                "cluster": ask_cluster_id("  which cluster id gets the competing label"),
                "label": ask_int("    which label value (0..M-1) competes there", 0, minv=0),
                "share": ask_float("    share of that cluster's LEFTOVER points to convert (0..1)",
                                    1.0, lo=0.0, hi=1.0),
                "favors": ask_choice("    where the competing label sits in the cluster",
                                      ["boundary", "core", "random"], "boundary",
                                      explain="boundary : the cluster's rim (the usual 'hard case').\n"
                                              "core     : the cluster's center.\n"
                                              "random   : anywhere in the cluster."),
            })
            if not ask_bool("  Add another competing-noise entry?", default=False):
                break
        clm["competing_noise"] = entries

    if mode != "random" and ask_bool(
            "Control WHERE inside each cluster the labelled points sit (center vs edge)?",
            default=False,
            explain="Optional, spatial only: choose whether the labelled points sit near the\n"
                    "cluster CENTER (core) or its EDGE (boundary). This does NOT change the score,\n"
                    "only which points carry the label. Useful for testing how a metric reacts\n"
                    "to geometry."):
        prof = ask_choice("How strongly to prefer them", ["linear", "exponential", "step"], "linear",
                          explain="linear      : gentle, even preference from center to edge.\n"
                                  "exponential : strong, points very near the target dominate\n"
                                  "              (tune with 'steepness').\n"
                                  "step        : a hard cut, take the N nearest (or farthest),\n"
                                  "              nothing in between.")
        cd = {"enabled": True, "profile": prof,
              "favors": ask_choice("Put the labelled points at the", ["core", "boundary"], "core",
                                   explain="core     : the center of each cluster.\n"
                                           "boundary : the edge/rim of each cluster.")}
        if prof == "exponential":
            cd["steepness"] = ask_float("Steepness (higher = more extreme; typical 2-8)", 3.0, lo=0.0)
        clm["centroid_dependence"] = cd

    _final_check(clm)
    return clm


def _add_balance(clm, M):
    balance = ask_choice("Label balance", ["balanced", "unbalanced"], "balanced",
                         explain="balanced   : every label is roughly the same size.\n"
                                 "unbalanced : some labels are bigger than others (like real data,\n"
                                 "             where one class is often much more common).")
    clm["balance"] = balance
    if balance != "unbalanced":
        return
    if ask_bool("Type the exact sizes yourself?", default=True,
                explain="Yes: you give the fraction for each label (they must add up to 1).\n"
                        "No: a 'skew rule' invents an uneven split for you."):
        while True:
            props = ask_floats(f"Fraction for each of the {M} labels",
                               explain="One number per label, adding up to 1.0. They become the label\n"
                                       "sizes, scaled to your data (e.g. 0.5 of 6500 points = 3250).")
            if len(props) == M and abs(sum(props) - 1.0) < 1e-6:
                clm["proportions"] = props
                return
            print(f"    (need exactly {M} numbers that add up to 1.0)")
    rule = ask_choice("Skew rule", ["geometric", "dominant_minority", "dirichlet"], "geometric",
                      explain="How to make the label sizes uneven:\n"
                              "  geometric         : a smooth shrinking series (e.g. 50, 25, 12, ...).\n"
                              "                      One 'ratio' knob, smaller ratio = steeper drop.\n"
                              "  dominant_minority : ONE label takes a big share you choose; the rest\n"
                              "                      split what's left equally (one common, several rare).\n"
                              "  dirichlet         : sizes drawn at RANDOM but repeatable from the seed.\n"
                              "                      'alpha' sets how uneven: <1 = very lopsided,\n"
                              "                      >1 = nearly equal. Good for sampling many imbalances.")
    clm["skew_rule"] = rule
    if rule == "geometric":
        clm["skew_params"] = {"ratio": ask_float("ratio (0..1; smaller = steeper drop-off)", 0.5, lo=0.0, hi=1.0)}
    elif rule == "dominant_minority":
        clm["skew_params"] = {
            "dominant_index": ask_int("which label is the big one (0..M-1)", 0, minv=0),
            "dominant_share": ask_float("its share of all points (0..1, e.g. 0.6)", 0.6, lo=0.0, hi=1.0)}
    else:
        clm["skew_params"] = {"alpha": ask_float("alpha (smaller = more lopsided; try 0.3 or 1.0)", 1.0, lo=0.0)}


def _build_rules(omit_recall):
    print("\n  Custom rules: each rule sends a share of ONE label into chosen cluster(s).")
    n = ask_int("How many rules", 1, minv=1)
    rules = []
    for i in range(n):
        print(f"\n  Rule {i + 1}:")
        rule: Dict[str, Any] = {
            "label": ask_int("    which label value (0..M-1)", 0, minv=0),
            "clusters": ask_cluster_ids("  into which cluster id(s), comma-separated")}
        if not omit_recall:
            rule["recall_target"] = ask_float(
                "    recall_target, share of THAT label's points to put here "
                "(1.0 = all/strong, lower = partial)", 0.8, lo=0.0, hi=1.0)
        rules.append(rule)
    return rules


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
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
    out = ask_str("Save this config to (a .yaml file)", "test_data_config.yaml")
    if not out.lower().endswith((".yaml", ".yml")):
        out += ".yaml"                       # always a .yaml file, distinct from any output folder
    if Path(out).resolve() == Path(gs["output_dir"]).resolve():
        out = "config_" + out                # last-ditch guard against a folder/file name clash
    Path(out).write_text(yaml.dump(config, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"\n  Wrote '{out}'.")
    print(f"  Run it any time with:  python -m clmsynth.main {out}")
    if ask_bool("Run the pipeline now?", default=True):
        subprocess.run([sys.executable, "-m", "clmsynth.main", out])


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
