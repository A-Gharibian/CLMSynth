"""Regenerate the troubleshooting catalog.

For every diagnostic in the troubleshooting reference, writes a config that
triggers it, runs the pipeline, captures the log, and asserts the expected code
actually appears. Output is grouped one folder per main table:

    troubleshooting_catalog/
        ValueError_1xx/            101..131   fatal, aborts the run (exit 2)
        InfeasibleAllocation_15x/  150..153   per-dataset skip (exit 0)
        KeyError_2xx/              201..209   per-dataset skip (exit 0)
        Warnings_3xx/              301..310   non-fatal, run succeeds (exit 0)

One code in the 1xx band is an exception to "fatal, aborts the run": since 0.6.3
[CLM-105] judges the DATASET's cluster ids rather than the config, so for
non-byoc sources it is logged per dataset and the run continues at exit 1.
[CLM-104] is config-scoped and aborts at exit 2. The catalog still files them
under ValueError_1xx, because that is what the engine raises; only the
pipeline's handling of them differs.

Everything the catalog writes is RELATIVE to the catalog folder, and every case
is run with that folder as the working directory. So a shipped fixture is
reproducible verbatim:

    cd docs/troubleshooting_catalog
    python -m clmsynth.main ValueError_1xx/CLM-150.yaml

Captured logs are normalized -- timestamps, run-folder names and path separators
removed -- so that regenerating produces a byte-identical log whenever the
behavior is unchanged, and a diff shows only what actually moved.

Usage (project interpreter, any working directory):

    python tools/CLM_catalog_gen.py [OUT_DIR]

OUT_DIR defaults to docs/troubleshooting_catalog. Pass a different one to
generate a candidate catalog for comparison before replacing the shipped one.
"""

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from clmsynth.clm_errors import CODES

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "docs" / "troubleshooting_catalog"
DATA_REL = "_data"            # byoc inputs, relative to OUT
SCRATCH_REL = "_scratch"      # per-case run output, deleted at the end
DATA = OUT / DATA_REL
SCRATCH = OUT / SCRATCH_REL

TABLES = {
    "1xx": "ValueError_1xx",
    "15x": "InfeasibleAllocation_15x",
    "2xx": "KeyError_2xx",
    "3xx": "Warnings_3xx",
}


def _checked_rmtree(target: Path, **kwargs) -> None:
    """Refuse to delete a non-catalog target."""
    if not target.exists():
        return
    if target != OUT and OUT not in target.parents:
        raise SystemExit(f"refusing to delete '{target}': it is outside '{OUT}'")
    if (target == OUT and any(OUT.iterdir())
            and not any((OUT / name).is_dir() for name in TABLES.values())):
        raise SystemExit(
            f"refusing to delete '{OUT}': it holds no catalog. Expected one of "
            f"{', '.join(sorted(TABLES.values()))} inside it. "
            "Pass an empty or non-existent OUT_DIR to build a new catalog.")
    shutil.rmtree(target, **kwargs)


# --------------------------------------------------------------------------
# Base configuration: offline fabricated_data, N=800, K=4 (ids 0..3, 200 each)
# --------------------------------------------------------------------------

def base():
    return {
        "global_settings": {"data_source": "fabricated_data", "output_dir": "OUTPUT"},
        "fabricated_data_suite": {
            "batteries": ["fabricated"],
            "datasets": ["baseline_4class"],
            "seed": 42,
        },
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": {
                "num_classes": 4,
                "proportions": [0.25, 0.25, 0.25, 0.25],
                "balance": "unbalanced",
                "skew_rule": "geometric",
                "matching_mode": "custom",
                "assignment_matrix": [
                    {"clusters": [i], "label": i, "recall_target": 0.8} for i in range(4)
                ],
                "split_rule": "proportional_to_size",
                "spillover_rule": "proportional_to_marginal",
                "centroid_dependence": {"enabled": True, "profile": "linear", "favors": "core"},
            },
        },
    }


def byoc(stem, n_clusters=65, n_per=3, string_ids=False):
    """Write a byoc CSV and return a config pointing at it."""
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for k in range(n_clusters):
        for _ in range(n_per):
            cid = f"C{k}" if string_ids else k
            rows.append([float(rng.normal(k, 0.3)), float(rng.normal(k, 0.3)), cid])
    with open(DATA / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Feature_1", "Feature_2", "cluster"])
        w.writerows(rows)

    cfg = base()
    cfg["global_settings"]["data_source"] = "byoc"
    del cfg["fabricated_data_suite"]
    cfg["byoc_suite"] = {
        "batteries": ["local"],
        "input_dir": DATA_REL,
        "datasets": [stem],
        "cluster_column": "cluster",
        "standardize": False,
        "seed": 42,
    }
    return cfg


def clm(cfg):
    return cfg["label_generation"]["clm_label"]


# --------------------------------------------------------------------------
# One builder per diagnostic.  Each returns a config dict.
# --------------------------------------------------------------------------

def c(**patch):
    """Base config with clm_label keys patched/removed (None removes)."""
    cfg = base()
    for k, v in patch.items():
        if v is None:
            clm(cfg).pop(k, None)
        else:
            clm(cfg)[k] = v
    return cfg


def single(**patch):
    patch.setdefault("single_match", {"cluster": 0, "label": 0})
    return c(matching_mode="single", assignment_matrix=None, **patch)


def labels_only(**patch):
    """Base config over the zero-feature preset: cluster ids, no geometry.

    The only dataset in the project with no feature columns, and therefore the
    only way a configuration can reach the [CLM-125] placement guard.
    """
    cfg = c(**patch)
    cfg["fabricated_data_suite"]["datasets"] = ["labels_only_4class"]
    return cfg


CASES = [
    # ---- 1xx : ValueError, fatal, aborts the run ------------------------
    ("1xx", 101, "matching_mode is not one of the four modes",
     lambda: c(matching_mode="bogus_mode")),
    # balance='balanced' with no proportions: otherwise base()'s four-entry
    # proportions against num_classes=3 trips [CLM-121] first, which is a
    # different code and masks this one entirely.
    ("1xx", 102, "perfect with M != K",
     lambda: c(matching_mode="perfect", num_classes=3, assignment_matrix=None,
               proportions=None, balance="balanced")),
    ("1xx", 103, "single with M < 2",
     lambda: c(matching_mode="single", num_classes=1, proportions=[1.0],
               single_match={"cluster": 0, "label": 0}, assignment_matrix=None)),
    ("1xx", 104, "single_match.label out of range 0..M-1",
     lambda: single(single_match={"cluster": 0, "label": 9})),
    ("1xx", 105, "assignment_matrix cluster id not in the data",
     lambda: c(assignment_matrix=[{"clusters": [99], "label": 0, "recall_target": 0.8}])),
    ("1xx", 106, "explicit proportions do not sum to 1",
     lambda: c(proportions=[0.5, 0.5, 0.5, 0.5])),
    ("1xx", 107, "unknown skew_rule (unbalanced, no proportions)",
     lambda: c(proportions=None, skew_rule="bogus_rule")),
    ("1xx", 108, "unknown split_rule (rule spans >1 cluster so it is read)",
     lambda: c(split_rule="bogus_split",
               assignment_matrix=[{"clusters": [0, 1], "label": 0, "recall_target": 0.5}])),
    ("1xx", 109, "unknown spillover_rule",
     lambda: c(spillover_rule="bogus_spill")),
    ("1xx", 110, "unknown centroid_dependence.profile",
     lambda: c(centroid_dependence={"enabled": True, "profile": "bogus", "favors": "core"})),
    ("1xx", 111, "target_metric under perfect",
     lambda: c(matching_mode="perfect", assignment_matrix=None,
               target_metric={"type": "mcc", "value": 0.5})),
    ("1xx", 112, "target_metric.type not mcc/ari",
     lambda: c(target_metric={"type": "bogus", "value": 0.5})),
    ("1xx", 113, "target_metric.value outside [-1, 1]",
     lambda: c(target_metric={"type": "mcc", "value": 5.0})),
    ("1xx", 114, "target_metric under random",
     lambda: c(matching_mode="random", assignment_matrix=None,
               target_metric={"type": "mcc", "value": 0.5})),
    ("1xx", 115, "competing_noise under random",
     lambda: c(matching_mode="random", assignment_matrix=None,
               competing_noise=[{"cluster": 0, "label": 1, "share": 0.5, "favors": "boundary"}])),
    ("1xx", 116, "competing_noise.favors not core/boundary/random",
     lambda: c(competing_noise=[{"cluster": 0, "label": 1, "share": 0.5, "favors": "bogus"}])),
    ("1xx", 117, "competing_noise.share outside [0, 1]",
     lambda: c(competing_noise=[{"cluster": 0, "label": 1, "share": 5.0, "favors": "boundary"}])),
    ("1xx", 118, "competing_noise.label out of range",
     lambda: c(competing_noise=[{"cluster": 0, "label": 9, "share": 0.5, "favors": "boundary"}])),
    ("1xx", 119, "competing_noise.cluster not in the data",
     lambda: c(competing_noise=[{"cluster": 99, "label": 1, "share": 0.5, "favors": "boundary"}])),
    # No rule touches cluster 0, so its free capacity is 200 at EVERY alpha; two
    # competing_noise entries then over-claim it regardless of the recall level,
    # which is what makes every probed alpha infeasible. (Putting the noise on a
    # rule-claimed cluster does not work: at alpha=1 the rule fills it, leaving
    # nothing to claim, so the entries degrade to a [CLM-305] no-op and that
    # alpha stays feasible.)
    ("1xx", 120, "target_metric: no feasible alpha at any recall level",
     lambda: c(target_metric={"type": "mcc", "value": 0.5},
               assignment_matrix=[{"clusters": [i], "label": i, "recall_target": 0.8}
                                  for i in (1, 2, 3)],
               competing_noise=[
                   {"cluster": 0, "label": 1, "share": 0.7, "favors": "random"},
                   {"cluster": 0, "label": 2, "share": 0.7, "favors": "random"},
               ])),
    # Guarded since 0.6.0. This case previously expected NumPy's "could not be
    # broadcast" from the spillover pool, which is what the length mismatch used
    # to surface as -- and only under the default spillover rule; uniform and
    # concentrated wrote the undeclared labels straight into the dataset instead.
    ("1xx", 121, "proportions length != num_classes",
     lambda: c(proportions=[0.4, 0.3, 0.3],
               assignment_matrix=[{"clusters": [i], "label": i, "recall_target": 0.4}
                                  for i in range(3)])),
    ("1xx", 122, "target_metric.scope not pair/global",
     lambda: c(target_metric={"type": "mcc", "value": 0.5, "scope": "bogus"})),
    ("1xx", 123, "scope pair with type ari",
     lambda: single(target_metric={"type": "ari", "value": 0.5, "scope": "pair"})),
    ("1xx", 124, "scope pair outside single mode",
     lambda: c(target_metric={"type": "mcc", "value": 0.5, "scope": "pair"})),
    # The only config that can reach this guard. Every other dataset in every
    # source carries at least one feature column, so coords are never empty;
    # labels_only_4class carries cluster ids and nothing else, which is a valid
    # run in itself until spatial placement is asked for on top of it.
    ("1xx", 125, "spatial placement requested over a dataset with no features",
     lambda: labels_only(centroid_dependence={"enabled": True, "profile": "linear",
                                               "favors": "core"})),
    ("1xx", 126, "num_classes outside [1, 64]",
     lambda: c(num_classes=100, proportions=None, balance="balanced")),
    ("1xx", 127, "dataset cluster count K > 64 (byoc, 65 clusters)",
     lambda: byoc("k65_clusters", n_clusters=65)),
    ("1xx", 128, "concentrated_labels holds an id outside 0..M-1",
     lambda: c(spillover_rule="concentrated", concentrated_labels=[99])),
    ("1xx", 129, ("centroid_dependence.favors not core/boundary "
                  "(any other value previously meant boundary, silently)"),
     lambda: c(centroid_dependence={"enabled": True, "profile": "linear", "favors": "CORE"})),
    ("1xx", 130, "scope 'pair' with a spillover rule that can emit the target label",
     lambda: single(spillover_rule="uniform",
                    target_metric={"type": "mcc", "value": 0.6, "scope": "pair"})),
    # alpha exactly 0 is the break point: every dirichlet draw is 0 and the
    # engine's own normalization divides by their sum. Chosen over the negative
    # ratio / share cases because those did not raise at all before 0.6.3 -- they
    # returned negative label counts that still summed to N.
    ("1xx", 131, "skew_params out of range for the chosen skew_rule",
     lambda: c(proportions=None, skew_rule="dirichlet", skew_params={"alpha": 0.0})),

    # ---- 15x : InfeasibleAllocationError, per-dataset skip ---------------
    ("15x", 150, "one rule's budget exceeds its clusters' capacity",
     lambda: c(proportions=[0.5, 0.2, 0.2, 0.1])),
    ("15x", 151, "two rules jointly over-claim one cluster",
     lambda: c(assignment_matrix=[
         {"clusters": [0], "label": 0, "recall_target": 0.8},
         {"clusters": [0], "label": 1, "recall_target": 0.8},
     ])),
    ("15x", 153, "two assignment_matrix rules name the same label and over-claim it",
     lambda: c(assignment_matrix=[{"clusters": [0], "label": 0, "recall_target": 0.6},
                                  {"clusters": [1], "label": 0, "recall_target": 0.6}])),
    ("15x", 152, "competing_noise entries over-claim one cluster's leftovers",
     lambda: c(competing_noise=[
         {"cluster": 0, "label": 1, "share": 0.7, "favors": "random"},
         {"cluster": 0, "label": 2, "share": 0.7, "favors": "random"},
     ])),

    # ---- 2xx : MissingConfigKey, per-dataset skip, exit 0 ----------------
    ("2xx", 201, "num_classes missing", lambda: c(num_classes=None)),
    ("2xx", 202, "matching_mode missing", lambda: c(matching_mode=None)),
    ("2xx", 203, "skew_rule missing (unbalanced, no proportions)",
     lambda: c(proportions=None, skew_rule=None)),
    ("2xx", 204, "single_match missing under single",
     lambda: c(matching_mode="single", assignment_matrix=None, single_match=None)),
    ("2xx", 205, "cluster missing inside single_match",
     lambda: single(single_match={"label": 0})),
    ("2xx", 206, "assignment_matrix missing under custom",
     lambda: c(assignment_matrix=None)),
    ("2xx", 207, "clusters missing inside a rule",
     lambda: c(assignment_matrix=[{"label": 0, "recall_target": 0.8}])),
    ("2xx", 208, "recall_target missing inside a rule (no target_metric)",
     lambda: c(assignment_matrix=[{"clusters": [0], "label": 0}])),
    ("2xx", 209, "value missing inside target_metric",
     lambda: c(target_metric={"type": "mcc"})),

    # ---- 3xx : warnings, run succeeds ------------------------------------
    ("3xx", 301, "balance balanced with explicit proportions",
     lambda: c(balance="balanced")),
    ("3xx", 302, "perfect ignores proportions/balance/skew_rule",
     lambda: c(matching_mode="perfect", assignment_matrix=None)),
    ("3xx", 303, "target_metric present with per-rule recall_target",
     lambda: c(target_metric={"type": "mcc", "value": 0.5, "tolerance": 0.2})),
    ("3xx", 304, "competing_noise active (counts deviate from proportions)",
     lambda: c(competing_noise=[{"cluster": 0, "label": 1, "share": 0.5, "favors": "boundary"}])),
    ("3xx", 305, "competing_noise on a cluster with no unclaimed points",
     lambda: c(assignment_matrix=[{"clusters": [0], "label": 0, "recall_target": 1.0}],
               proportions=[0.25, 0.25, 0.25, 0.25],
               competing_noise=[{"cluster": 0, "label": 1, "share": 0.5, "favors": "random"}])),
    ("3xx", 306, "target_metric did not converge within tolerance",
     lambda: c(num_classes=3, proportions=[0.34, 0.33, 0.33],
               assignment_matrix=[{"clusters": [i], "label": i, "recall_target": 0.8}
                                  for i in range(3)],
               target_metric={"type": "mcc", "value": 0.99, "tolerance": 0.001, "max_iter": 15})),
    ("3xx", 307, "scope pair target outside the pair's reachable range",
     lambda: single(target_metric={"type": "mcc", "value": 0.01, "scope": "pair"})),
    ("3xx", 308, "scope pair resizes the target label",
     lambda: single(target_metric={"type": "mcc", "value": 0.6, "scope": "pair"})),
    ("3xx", 309, "delivered target_metric outside tolerance",
     lambda: c(target_metric={"type": "mcc", "value": 0.5, "tolerance": 0.0001})),
    # The pair counterpart of 309. The target label is sized to an integer number
    # of points, so the achieved coefficient lands on a discrete ladder and
    # essentially never equals the request exactly; tolerance 0 makes that
    # visible. Before 0.6.3 this key was ignored on the pair path and the check
    # was hardcoded to 0.01.
    ("3xx", 310, "delivered pair mcc outside tolerance",
     lambda: single(target_metric={"type": "mcc", "value": 0.6, "scope": "pair",
                                    "tolerance": 0.0})),
]


_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ", re.M)
_RUN_DIR = re.compile(r"\d{6}_[A-Za-z]+_\d{6}(?:_\d+)?")


def normalise(log: str) -> str:
    """Strip everything that changes between two runs of identical behavior.

    Without this the catalog cannot be diffed: wall-clock timestamps and the
    clock-derived run-folder name differ on every regeneration, so all 51 logs
    would show as modified even when nothing about the diagnostics moved. What
    remains is the log level, the message, and the shape of the run -- which is
    the part a reader is comparing their own output against.
    """
    log = _TIMESTAMP.sub("", log)
    log = _RUN_DIR.sub("{run}", log)
    log = log.replace("\\", "/")
    # Absolute paths are machine-specific.
    return log.replace(str(OUT).replace("\\", "/"), "{catalog}")


def run_case(table, code, desc, builder, expect):
    folder = OUT / TABLES[table]
    folder.mkdir(parents=True, exist_ok=True)
    cfg = builder()

    # Relative to OUT, which is also the working directory below, so the shipped
    # fixture is exactly what ran and a reader can rerun it by cd-ing here.
    run_dir = SCRATCH / f"CLM-{code}"
    _checked_rmtree(run_dir)
    cfg["global_settings"]["output_dir"] = f"{SCRATCH_REL}/CLM-{code}"

    cfg_path = folder / f"CLM-{code}.yaml"
    rel_cfg = f"{TABLES[table]}/CLM-{code}.yaml"
    header = (f"# [CLM-{code}] {desc}\n"
              f"# Regenerated by CLM_catalog_gen.py against the shipped engine.\n"
              f"# Reproduce: cd to this catalog's root, then\n"
              f"#     python -m clmsynth.main {rel_cfg}\n")
    cfg_path.write_text(header + yaml.dump(cfg, sort_keys=False, default_flow_style=False),
                        encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "clmsynth.main", rel_cfg],
        cwd=OUT, capture_output=True, text=True, check=False,
    )
    log = normalise((proc.stdout or "") + (proc.stderr or ""))
    (folder / f"CLM-{code}.log").write_text(
        f"$ python -m clmsynth.main {rel_cfg}\n"
        f"# exit code: {proc.returncode}\n\n{log}", encoding="utf-8")

    # Default expectation is the BRACKETED form the engine emits. A bare
    # "CLM-123" also occurs in the config filename and the scratch output path,
    # which produced false positives on the first pass.
    return (expect in log), proc.returncode, log


def check_builder_coverage():
    """Every registry code must have a builder here.

    Returns the codes with no builder, and the builders with no code.

    The counterpart of the shipped test asserting every code has a *fixture*.
    Both are needed, and neither implies the other: regenerating in 0.6.3 found
    four fixtures in the catalog (`CLM-128`, `129`, `130`, `153`) with no builder
    in this file, so a regeneration would have silently deleted them, and two
    builders still encoding pre-0.6.0 behavior. Nothing compared the two, which
    is why both drifts survived.

    Checked before anything is written, so a missing builder stops the run rather
    than producing a catalog that is quietly short of a code.
    """
    built = {case[1] for case in CASES}
    built_without_code = {c for c in built if c not in CODES}
    return sorted(set(CODES) - built), sorted(built_without_code)


def main():
    missing, built_without_code = check_builder_coverage()
    if missing:
        print(f"ABORT: {len(missing)} registry code(s) have no builder in this file: "
              f"{missing}")
        print("Regenerating now would drop them from the catalog. Add a CASES entry "
              "for each, then re-run.")
        return 1
    print(f"builder coverage: {len(CODES)} registry codes, all built.\n")
    if built_without_code:
        # Add --strict here to return 1 instead.
        print(f"WARNING: {len(built_without_code)} builder(s) have no registry code "
              f"({built_without_code}). Regenerating writes a shipped fixture for a "
              "diagnostic the engine can no longer emit. Drop the CASES entry, or "
              "re-register the code.\n")


    # ignore_errors: on Windows the folder can be held open by a shell sitting in
    # it, and failing the whole run over that is not worth it -- every fixture is
    # rewritten below anyway.
    _checked_rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        table, code, desc, builder = case[:4]
        expect = case[4] if len(case) > 4 else f"[CLM-{code}]"
        try:
            hit, rc, _log = run_case(table, code, desc, builder, expect)
        except Exception as exc:                     # broad by design: reporting
            results.append((table, code, "BUILD-ERROR", "-", f"{type(exc).__name__}: {exc}"))
            continue
        note = f"matched {expect!r}" if hit else f"EXPECTED {expect!r} NOT FOUND"
        results.append((table, code, "ok" if hit else "MISS", rc, note))

    _checked_rmtree(SCRATCH)

    print(f"{'table':<6}{'code':<7}{'status':<9}{'exit':<6}note")
    print("-" * 72)
    miss = 0
    for table, code, status, rc, note in results:
        if status != "ok":
            miss += 1
        print(f"{table:<6}{code:<7}{status:<9}{rc!s:<6}{note}")
    print("-" * 72)
    print(f"{len(results) - miss}/{len(results)} produced their expected diagnostic")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
