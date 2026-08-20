"""Smoke test.

Scope is deliberately narrow, and everything outside it belongs to another
suite:

Runs on every supported interpreter (3.11-3.14); nothing here uses syntax or
stdlib newer than 3.11.

    python -m pytest tests/test_smoke.py -v
"""

import copy
import importlib
import pkgutil

import pandas as pd

from clmsynth.generate_config import generate_base_config
from clmsynth.main import load_config, run_pipeline
from clmsynth.metrics import clustering_mcc

# The dataset the offline source generates for this preset.
EXPECTED_ROWS = 800
EXPECTED_COLUMNS = [
    "Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5", "Feature_6",
    "Cluster_0", "Label_0",
]
# Four equal clusters, four equal labels, every rule at recall 0.8: the achieved
# agreement lands on the configured recall.
EXPECTED_MCC = 0.8
MCC_TOLERANCE = 0.01

# A representative config, not a minimal one: `perfect` matching would exercise
# only the cluster->label bijection and could pass while the allocation math is
# silently wrong. This routes 80% of each label into its own cluster and lets
# spillover and centroid placement run, so the paths the project actually exists
# for are all live.
SMOKE_CONFIG = {
    "global_settings": {
        "data_source": "fabricated_data",
        "output_dir": "OUTPUT",          # unused: run_pipeline writes to the dirs it is given
    },
    "fabricated_data_suite": {
        "batteries": ["fabricated"],
        "datasets": ["baseline_4class"],  # n=800, 4 clusters, integer ids 0..3
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
            "skew_rule": "geometric",     # unused: explicit proportions take precedence
            "matching_mode": "custom",
            "assignment_matrix": [
                {"clusters": [0], "label": 0, "recall_target": 0.8},
                {"clusters": [1], "label": 1, "recall_target": 0.8},
                {"clusters": [2], "label": 2, "recall_target": 0.8},
                {"clusters": [3], "label": 3, "recall_target": 0.8},
            ],
            "split_rule": "proportional_to_size",
            "spillover_rule": "proportional_to_marginal",
            "centroid_dependence": {
                "enabled": True,
                "profile": "linear",
                "favors": "core",
            },
        },
    },
}

# The upstream payload `generate_base_config` renders into a config, expressed
# as the minimal set of facts the template needs.
#
# `test_06_diagnostics.py` defines a MINIMAL_PAYLOAD with the same contents; 06 mutates its copy per case
# (`overrides`/`drop`) to provoke render-time warnings.
MINIMAL_PAYLOAD = {
    "data_source": "fabricated_data",
    "batteries": ["fabricated"], "datasets": ["baseline_4class"], "source_seed": 42,
    "n_labels": 1, "source_labeling": "labels0", "label_seed": 42,
    "num_classes": 4, "proportions": [0.25, 0.25, 0.25, 0.25],
    "balance": "unbalanced", "skew_rule": "geometric",
    "matching_mode": "custom",
    "assignment_matrix": [{"clusters": [i], "label": i, "recall_target": 0.8}
                          for i in range(4)],
    "split_rule": "proportional_to_size",
    "spillover_rule": "proportional_to_marginal",
    "centroid_enabled": True, "centroid_profile": "linear", "centroid_favors": "core",
}


# ---------------------------------------------------------------------------
# 1. The package loads
# ---------------------------------------------------------------------------

def test_every_module_imports():
    """Import every module in the package.

    Catches import-time errors of any kind, including names used only in
    annotations, 3.14 defers those (PEP 649) while 3.11-3.13 evaluate them
    eagerly, so a missing import can be invisible on the dev interpreter and
    fatal everywhere else.
    """
    import clmsynth

    failures = []
    for mod in pkgutil.iter_modules(clmsynth.__path__):
        name = f"clmsynth.{mod.name}"
        try:
            importlib.import_module(name)
        except Exception as exc:                      # broad by design: reporting
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)


def test_public_api_is_importable():
    """Every name promised by __all__ actually resolves on the package."""
    import clmsynth

    missing = [n for n in clmsynth.__all__ if not hasattr(clmsynth, n)]
    assert not missing, f"__all__ names not present on the package: {missing}"


# ---------------------------------------------------------------------------
# 2. The pipeline runs and writes the expected CSV
# ---------------------------------------------------------------------------

def test_pipeline_produces_expected_csv(tmp_path):
    """Run the offline source end to end and check the written CSV.
    """
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    # deepcopy: the module-level config is shared, and run_pipeline pops keys
    # off the suite block it is handed.
    n_ok = run_pipeline("fabricated_data", copy.deepcopy(SMOKE_CONFIG),
                        csv_dir, png_dir, txt_dir)
    assert n_ok == 1, f"expected 1 processed dataset, got {n_ok}"

    csvs = sorted(csv_dir.glob("*.csv"))
    assert len(csvs) == 1, f"expected one CSV, got {[p.name for p in csvs]}"
    df = pd.read_csv(csvs[0])

    # -- shape and columns --
    assert list(df.columns) == EXPECTED_COLUMNS, list(df.columns)
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert not df.isnull().to_numpy().any(), "CSV contains empty cells"

    # -- the two generated columns hold what they promise --
    # Cluster ids must be integers 0..K-1, like every other source. They were
    # strings ("Class_0", ...) until 0.3.0, which made clm_label configs
    # non-portable between sources.
    assert sorted(df["Cluster_0"].unique().tolist()) == [0, 1, 2, 3], \
        "cluster ids are not integers 0..3: {}".format(sorted(df["Cluster_0"].unique().tolist()))
    labels = set(df["Label_0"].unique().tolist())
    assert labels <= {0, 1, 2, 3}, f"labels outside the configured label space: {labels}"

    # -- the CLM contract --
    mcc = float(clustering_mcc(df["Cluster_0"], df["Label_0"]))
    assert abs(mcc - EXPECTED_MCC) < MCC_TOLERANCE, \
        f"MCC {mcc:.4f} is not the configured {EXPECTED_MCC}"


# ---------------------------------------------------------------------------
# 3. The documented quick start, end to end
# ---------------------------------------------------------------------------

def test_rendered_config_runs(tmp_path):
    """`clmsynth-config` then `clmsynth`: the two-command flow the README opens with.

        python -m clmsynth.generate_config    # payload -> config
        python -m clmsynth.main               # config  -> dataset
    """
    rendered = tmp_path / "rendered_config.yaml"
    # dict(): the module-level payload is shared, and a renderer that mutated
    # what it was handed would otherwise leak into any test added after this one.
    generate_base_config(dict(MINIMAL_PAYLOAD), output_path=str(rendered))
    assert rendered.is_file(), "generate_base_config wrote nothing"

    # It must be YAML the pipeline's own loader accepts, not just parseable text.
    config = load_config(str(rendered))
    assert config["global_settings"]["data_source"] == "fabricated_data"

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir) == 1
    written = sorted(csv_dir.glob("*.csv"))
    assert len(written) == 1
    assert "Label_0" in pd.read_csv(written[0]).columns
