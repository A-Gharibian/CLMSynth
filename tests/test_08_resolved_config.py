"""Category 8: The resolved config.

The run folder holds the configuration as the run read it, not a copy of the
file it was given (`resolved_config.py`). What is checked, in order:

  reproduction     Running a resolved config again writes byte-identical CSVs,
                   across matching modes, skew rules, targets, competing noise,
                   placement, drawn seeds and a dataset the run skipped. It is
                   also the guard for the engine defaults that resolved_config.py
                   restates: one that drifts changes the labels and fails here.
  idempotence      Resolving a resolved config changes nothing.
  left out         Keys the run never reads, or reads and ignores ([CLM-301],
                   [CLM-302], [CLM-303], [CLM-311]), are not in it.
  as written       A missing required key stays missing and a malformed value
                   stays as written, so a run refuses it the same way.
  search defaults  `max_iter` and `tolerance` seldom change a search that
                   converges, so they are checked against [CLM-306], which
                   prints both.
  either format    The resolved config is written in the format of the path it
                   is given, and reads back the same in each.
  the CLI          `main` writes the resolved config in the format of its input,
                   and no copy of the input.
"""

import json
import logging
import sys
from typing import Any

import numpy as np
import pytest

import clmsynth.main
from clmsynth.clm_label_engine import generate_clm_labels
from clmsynth.main import run_pipeline
from clmsynth.resolved_config import RESOLVED_CONFIG_NAME, TARGET_METRIC, resolve_config


def pipeline_config(clm, datasets=("baseline_4class",), suite_seed: Any = 7,
                    label_seed: Any = 11, n_labels=1):
    """A fabricated_data run over `clm`, with keys no run reads around it.

    `"absent"` leaves a seed key out; None asks for a drawn seed.
    """
    label = {"n_labels": n_labels, "source_labeling": "labels0", "noise": 0.3, "clm_label": clm}
    suite: dict[str, Any] = {"batteries": ["fabricated"], "datasets": list(datasets)}
    if label_seed != "absent":
        label["seed"] = label_seed
    if suite_seed != "absent":
        suite["seed"] = suite_seed
    return {
        "global_settings": {"data_source": "fabricated_data", "output_dir": "OUT", "unused": 1},
        "fabricated_data_suite": suite,
        "clustbench_suite": {"batteries": "all"},
        "label_generation": label,
    }


def run_once(config, folder, yaml=None):
    """(datasets processed, {csv name: bytes}, the resolved config it wrote).
    Written as JSON, which needs nothing installed, or as YAML when given `yaml`."""
    dirs = [folder / sub for sub in ("csv", "png", "txt")]
    for d in dirs:
        d.mkdir(parents=True)
    out = folder / (RESOLVED_CONFIG_NAME if yaml else "resolved_config.json")
    n_ok = run_pipeline("fabricated_data", config, dirs[0], dirs[1], dirs[2],
                        render_plots=False, config_out=out)
    csvs = {p.name: p.read_bytes() for p in sorted(dirs[0].iterdir())}
    text = out.read_text(encoding="utf-8")
    return n_ok, csvs, yaml.safe_load(text) if yaml else json.loads(text)


# baseline_4class: 800 points in four clusters of 200. Every case generates its
# labels; a skipped label would make the comparison below vacuous.
REPRODUCTION_CASES = [
    pytest.param(pipeline_config({
        "num_classes": 4, "matching_mode": "perfect", "balance": "unbalanced",
        "proportions": [0.4, 0.3, 0.2, 0.1], "skew_rule": "geometric", "typo": 3,
    }), id="perfect-with-ignored-count-keys"),
    pytest.param(pipeline_config({
        "num_classes": 4, "matching_mode": "single", "balance": "balanced",
        "proportions": [0.4, 0.3, 0.2, 0.1], "single_match": {"cluster": 0, "label": 0},
        "assignment_matrix": [{"clusters": [1], "label": 1, "recall_target": 0.5}],
        "split_rule": "equal",
    }), id="single-with-ignored-proportions-matrix-split"),
    pytest.param(pipeline_config({
        "num_classes": 3, "matching_mode": "custom", "balance": "unbalanced",
        "proportions": [0.5, 0.3, 0.2],
        "assignment_matrix": [{"clusters": [0, 1], "label": 0, "recall_target": 0.8},
                              {"clusters": [2], "label": 1, "recall_target": 0.7}],
        "split_rule": "equal", "spillover_rule": "uniform",
        "competing_noise": [{"cluster": 3, "label": 2}],
        "centroid_dependence": {"enabled": True, "profile": "exponential"},
    }), id="custom-noise-exponential-placement"),
    pytest.param(pipeline_config({
        "num_classes": 3, "matching_mode": "custom",
        "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.9},
                              {"clusters": [1], "label": 1, "recall_target": 0.9}],
        "target_metric": {"type": "mcc", "value": 0.5},
    }, n_labels=2), id="custom-global-target-two-labels"),
    pytest.param(pipeline_config({
        "num_classes": 3, "matching_mode": "single", "single_match": {"cluster": 1, "label": 2},
        "target_metric": {"type": "mcc", "value": 0.6, "scope": "pair", "max_iter": 5,
                          "probe_seed": 3},
    }), id="single-pair-target"),
    pytest.param(pipeline_config({
        "num_classes": 3, "matching_mode": "random", "balance": "unbalanced",
        "skew_rule": "dirichlet", "spillover_rule": "uniform",
        "centroid_dependence": {"enabled": False, "favors": "boundary"},
    }), id="random-dirichlet-skew"),
    pytest.param(pipeline_config({
        "num_classes": 3, "matching_mode": "single", "balance": "unbalanced",
        "skew_rule": "geometric", "skew_params": {"alpha": 4.0},
        "single_match": {"cluster": 0, "label": 2},
    }), id="geometric-skew-default-ratio"),
    pytest.param(pipeline_config({
        "num_classes": 4, "matching_mode": "single", "spillover_rule": "concentrated",
        "single_match": {"cluster": 0, "label": 0},
    }), id="concentrated-default-labels"),
    pytest.param(pipeline_config({
        "num_classes": 4, "matching_mode": "single", "single_match": {"cluster": 0, "label": 0},
        "competing_noise": [{"cluster": 1, "label": 2, "share": 0.5, "favors": "core"}],
        "centroid_dependence": {"profile": "step", "steepness": 9},
    }), id="noise-placed-with-placement-off"),
    pytest.param(pipeline_config({
        "num_classes": 4, "matching_mode": "single", "single_match": {"cluster": 0, "label": 1},
    }, suite_seed=None, label_seed=None), id="drawn-seeds"),
    pytest.param(pipeline_config({
        "num_classes": 4, "matching_mode": "single", "single_match": {"cluster": 0, "label": 1},
    }, suite_seed="absent", label_seed="absent"), id="absent-seeds"),
    # labels_only_4class has no feature vectors, so [CLM-125] skips it both times.
    pytest.param(pipeline_config({
        "num_classes": 2, "matching_mode": "single", "single_match": {"cluster": 0, "label": 0},
        "balance": "unbalanced", "proportions": [0.2, 0.8],
        "centroid_dependence": {"enabled": True, "favors": "boundary"},
    }, datasets=("baseline_2class", "baseline_4class", "labels_only_4class")),
        id="three-datasets-one-skipped"),
    # Empty is absent: each label's search runs on that label's seed, and the
    # resolved config leaves the key out. It used to write null, a fresh stream.
    pytest.param(pipeline_config({
        "num_classes": 3, "matching_mode": "custom",
        "assignment_matrix": [{"clusters": [0], "label": 0}, {"clusters": [1], "label": 1}],
        "target_metric": {"type": "ari", "value": 0.3, "tolerance": 0, "probe_seed": None},
    }, n_labels=2), id="global-target-empty-probe-seed"),
]


@pytest.mark.parametrize("config", REPRODUCTION_CASES)
def test_a_resolved_config_repeats_the_run(tmp_path, config):
    """The contract, and the drift guard for the restated defaults."""
    n_first, first, resolved = run_once(config, tmp_path / "first")
    assert first, "the run wrote no CSV"
    assert all(b"Label_0" in body.splitlines()[0] for body in first.values()), \
        "a label was not generated, so the comparison proves nothing"

    n_again, again, resolved_again = run_once(resolved, tmp_path / "again")

    assert (n_again, again) == (n_first, first), "the resolved config wrote different CSVs"
    assert resolved_again == resolved, "resolving a resolved config changed it"


def resolved_label_generation(clm, n_labels=1) -> Any:
    config = pipeline_config(clm, n_labels=n_labels)
    resolved = resolve_config(config, "fabricated_data", ["fabricated"], "all",
                              [("fabricated", "baseline_4class")], 7, {}, 11)
    return resolved["label_generation"]


MISSING = object()


def lookup(block: Any, path: str) -> Any:
    """`block` at a dotted path ("target_metric.max_iter", "rows.0.label"), or MISSING."""
    for part in path.split("."):
        if isinstance(block, list) and part.isdigit() and int(part) < len(block):
            block = block[int(part)]
        elif isinstance(block, dict) and part in block:
            block = block[part]
        else:
            return MISSING
    return block


@pytest.mark.parametrize("clm, left_out", [
    pytest.param({"num_classes": 2, "matching_mode": "perfect", "balance": "unbalanced",
                  "proportions": [0.5, 0.5], "skew_rule": "geometric", "skew_params": {}},
                 ["balance", "proportions", "skew_rule", "skew_params"], id="perfect-CLM-302"),
    pytest.param({"num_classes": 2, "matching_mode": "single", "balance": "balanced",
                  "proportions": [0.5, 0.5], "single_match": {"cluster": 0, "label": 0}},
                 ["proportions"], id="balanced-CLM-301"),
    pytest.param({"num_classes": 2, "matching_mode": "single", "balance": "unbalanced",
                  "proportions": [0.5, 0.5], "skew_rule": "dirichlet",
                  "single_match": {"cluster": 0, "label": 0}},
                 ["skew_rule", "skew_params"], id="proportions-over-skew"),
    pytest.param({"num_classes": 2, "matching_mode": "custom",
                  "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.9}],
                  "target_metric": {"type": "mcc", "value": 0.5}},
                 ["assignment_matrix.0.recall_target"], id="solved-recall-CLM-303"),
    pytest.param({"num_classes": 2, "matching_mode": "single",
                  "single_match": {"cluster": 0, "label": 0, "recall_target": 0.5},
                  "assignment_matrix": [], "split_rule": "equal",
                  "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair",
                                    "max_iter": 5, "probe_seed": 3}},
                 ["single_match.recall_target", "assignment_matrix", "split_rule",
                  "target_metric.max_iter", "target_metric.probe_seed"], id="single-pair"),
    pytest.param({"num_classes": 2, "matching_mode": "custom",
                  "assignment_matrix": [{"clusters": [0], "label": 0}],
                  "target_metric": {"type": "mcc", "value": 0.5, "probe_seed": None}},
                 ["target_metric.probe_seed"], id="empty-probe-seed"),
    pytest.param({"num_classes": 2, "matching_mode": "random", "spillover_rule": "uniform",
                  "split_rule": "equal",
                  "centroid_dependence": {"enabled": True, "favors": "core",
                                          "profile": "step"}},
                 ["spillover_rule", "split_rule", "centroid_dependence"], id="random-CLM-311"),
    pytest.param({"num_classes": 2, "matching_mode": "single",
                  "single_match": {"cluster": 0, "label": 0}, "concentrated_labels": [1],
                  "centroid_dependence": {"enabled": True, "profile": "linear",
                                          "steepness": 2.0},
                  "typo": 1},
                 ["concentrated_labels", "centroid_dependence.steepness", "typo"],
                 id="read-only-where-it-applies"),
])
def test_keys_the_run_does_not_act_on_are_left_out(clm, left_out):
    """The file says what the run acted on: a key that changed nothing would
    read as if it had."""
    clm_out = resolved_label_generation(clm)["clm_label"]
    for path in left_out:
        assert lookup(clm_out, path) is MISSING, f"{path} is in the resolved config"


def test_keys_outside_the_engine_are_left_out_too():
    """`noise` is accepted and unused, another source's suite is never read, and
    with no label to generate `clm_label` is not read either."""
    config = pipeline_config({"num_classes": 2, "matching_mode": "perfect"})
    resolved = resolve_config(config, "fabricated_data", ["fabricated"], "all",
                              [("fabricated", "baseline_4class")], 7, {}, 11)

    assert "noise" not in resolved["label_generation"]
    assert "clustbench_suite" not in resolved
    assert resolved["global_settings"] == {"data_source": "fabricated_data", "output_dir": "OUT"}
    assert "clm_label" not in resolved_label_generation(
        {"num_classes": 2, "matching_mode": "perfect"}, n_labels=0)


def test_defaults_are_filled_where_they_are_read():
    clm_out = resolved_label_generation({
        "num_classes": 3, "matching_mode": "custom",
        "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.9}],
        "competing_noise": [{"cluster": 1, "label": 2}],
        "centroid_dependence": {"enabled": True, "profile": "exponential"},
        "target_metric": {"type": "ari", "value": 0.4},
    })["clm_label"]

    assert clm_out["balance"] == "balanced"
    assert clm_out["split_rule"] == "proportional_to_size"
    assert clm_out["spillover_rule"] == "proportional_to_marginal"
    assert clm_out["competing_noise"] == [{"cluster": 1, "label": 2, "share": 1.0,
                                           "favors": "boundary"}]
    assert clm_out["centroid_dependence"] == {"enabled": True, "favors": "core",
                                              "profile": "exponential", "steepness": 3.0}
    assert clm_out["target_metric"] == {"type": "ari", "value": 0.4, "scope": "global",
                                        "tolerance": 0.01, "max_iter": 40}


@pytest.mark.parametrize("clm, path, expected", [
    pytest.param({"matching_mode": "perfect"}, "num_classes", MISSING, id="missing-CLM-201"),
    pytest.param({"num_classes": 2, "matching_mode": "single", "balance": "unbalanced",
                  "single_match": {"cluster": 0, "label": 0}},
                 "skew_rule", MISSING, id="missing-CLM-203"),
    pytest.param({"num_classes": 2, "matching_mode": "single", "balance": "unbalanced",
                  "skew_params": [0.5], "single_match": {"cluster": 0, "label": 0}},
                 "skew_params", [0.5], id="malformed-CLM-131-before-CLM-203"),
    pytest.param({"num_classes": 2, "matching_mode": "single", "balance": "Balanced",
                  "single_match": {"cluster": 0, "label": 0}},
                 "balance", "Balanced", id="malformed-CLM-132"),
    pytest.param({"num_classes": 2, "matching_mode": "single", "centroid_dependence": "yes",
                  "single_match": {"cluster": 0, "label": 0}},
                 "centroid_dependence", "yes", id="malformed-CLM-141"),
    pytest.param({"num_classes": 2, "matching_mode": "single",
                  "single_match": {"cluster": 0, "label": 0},
                  "competing_noise": {"cluster": 1, "label": 1}},
                 "competing_noise", {"cluster": 1, "label": 1}, id="not-a-list-CLM-141"),
    pytest.param({"num_classes": 2, "matching_mode": "custom",
                  "assignment_matrix": [{"clusters": [0], "label": 0}],
                  "target_metric": {"type": "mcc", "value": 0.5, "probe_seed": -1}},
                 "target_metric.probe_seed", -1, id="malformed-probe-seed-CLM-141"),
])
def test_what_a_run_refuses_stays_as_written(clm, path, expected):
    """Filling a required key in, or repairing a malformed one, would make the
    resolved config run where the original was refused."""
    found = lookup(resolved_label_generation(clm)["clm_label"], path)
    if expected is MISSING:
        assert found is MISSING, f"{path} was filled in: {found!r}"
    else:
        assert found == expected


def test_an_unknown_matching_mode_keeps_the_whole_block():
    clm = {"num_classes": 2, "matching_mode": "perfectly", "typo": 1}
    assert resolved_label_generation(clm)["clm_label"] == clm


def test_the_restated_search_defaults_are_the_engines(caplog):
    """A target above the size ceiling never converges, and [CLM-306] then
    reports the max_iter and tolerance the engine used."""
    clusters = np.repeat([0, 1, 2, 3], [100, 80, 60, 40])
    coords = np.random.default_rng(0).normal(size=(len(clusters), 2))
    cfg = {"num_classes": 3, "matching_mode": "custom",
           "assignment_matrix": [{"clusters": [0], "label": 0}, {"clusters": [1], "label": 1}],
           "target_metric": {"type": "mcc", "value": 0.99}}

    with caplog.at_level(logging.WARNING, logger="clmsynth.clm_label_engine"):
        generate_clm_labels(clusters, coords, cfg, seed=1)

    message = next(r.getMessage() for r in caplog.records if "[CLM-306]" in r.getMessage())
    assert f" of {TARGET_METRIC['max_iter']} iterations" in message, message
    assert f"tol={TARGET_METRIC['tolerance']})" in message, message


def test_a_dict_run_writes_no_config_unless_asked(tmp_path):
    """`config_out` is opt-in, so existing callers of run_pipeline see no new file."""
    dirs = [tmp_path / sub for sub in ("csv", "png", "txt")]
    for d in dirs:
        d.mkdir()
    config = pipeline_config({"num_classes": 4, "matching_mode": "perfect"})

    assert run_pipeline("fabricated_data", config, dirs[0], dirs[1], dirs[2], False) == 1
    assert not [p for p in tmp_path.rglob("*") if p.suffix in (".yaml", ".json")]


@pytest.mark.parametrize("config", [
    # Floats, nested rules and a two-label search; no drawn seed, which would
    # differ between the two runs.
    REPRODUCTION_CASES[2], REPRODUCTION_CASES[3], REPRODUCTION_CASES[6],
])
def test_the_resolved_config_reads_the_same_in_either_format(tmp_path, config, yaml):
    _, as_yaml_csvs, as_yaml = run_once(config, tmp_path / "yaml", yaml)
    _, as_json_csvs, as_json = run_once(config, tmp_path / "json")

    assert as_json == as_yaml
    assert as_json_csvs == as_yaml_csvs

    _, again, _ = run_once(as_yaml, tmp_path / "again")
    assert again == as_yaml_csvs, "the YAML resolved config did not repeat the run"


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_the_run_folder_holds_the_resolved_config_not_a_copy(tmp_path, monkeypatch, request,
                                                             suffix):
    """Through the CLI: the one config file in the run folder is the resolved
    one, in info/, named after the run and in the format of the input."""
    config = pipeline_config({"num_classes": 4, "matching_mode": "perfect"})
    config["global_settings"]["output_dir"] = str(tmp_path / "out")
    config_path = tmp_path / f"my_config{suffix}"
    if suffix == ".json":
        config_path.write_text(json.dumps(config), encoding="utf-8")
    else:
        config_path.write_text("# only the original carries this\n"
                               + request.getfixturevalue("yaml").safe_dump(config),
                               encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth", str(config_path), "--no-viz"])

    clmsynth.main.main()

    (run_dir,) = (tmp_path / "out").iterdir()
    name = f"{run_dir.name}{suffix}"
    assert not [p.name for p in run_dir.iterdir() if p.is_file()]
    configs = [p.name for p in (run_dir / "info").iterdir() if p.suffix in (".yaml", ".json")]
    assert configs == [name]
    text = (run_dir / "info" / name).read_text(encoding="utf-8")
    if suffix == ".json":
        resolved = json.loads(text)
    else:
        assert text.startswith("# The configuration this run read")
        assert "only the original carries this" not in text
        resolved = request.getfixturevalue("yaml").safe_load(text)
    assert "noise" not in resolved["label_generation"]
    assert resolved["fabricated_data_suite"]["datasets"] == ["baseline_4class"]
