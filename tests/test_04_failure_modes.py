"""Category 4: Failure modes and recovery.

What the pipeline does when something goes wrong. `06_diagnostics` covers the
configuration half, a bad config produces the right coded error. This covers
what happens *around* that: one bad dataset must not take down a batch, a
failed plot must not be mistaken for a failed dataset, and a config that cannot
be loaded at all must exit distinguishably.

What is checked:

  batch isolation      a dataset raising mid-labeling is skipped alone; every
                       other dataset in the batch, including ones after it in
                       iteration order, still completes (finding F3)
  plot failure         reported distinguishably (False + a [PLOT-FAIL] log
                       line) and does NOT count as a dataset failure, since
                       the CSV is the deliverable and plotting is best-effort
                       (finding N3)
  error boundary       a failure inside `plt.subplots()` escapes the function's
                       own handler, unlike every other plotting failure
                       (finding N4, characterized not fixed)
  exit codes           a missing or malformed config exits 1, distinct from
                       exit 2 for a coded config error and 0 for success
  network timeout      `CLUSTBENCH_TIMEOUT` is passed to `urlopen`

"""

import logging
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

import clmsynth.main
from clmsynth import dataset_sources
from clmsynth.label_context import DatasetContext
from clmsynth.main import load_config, run_pipeline
from clmsynth.visualization import plot_feature_scatter

# The Agg backend is selected in conftest, before this module is imported, so a
# headless runner cannot fail for display reasons. This module renders for real
# in places, it tests plot FAILURE, so it deliberately does not use the
# `no_plots` fixture.


def byoc_config(input_dir, output_dir, datasets):
    """A batch over several BYOC datasets, matched on cluster 0.

    `scope: pair` validates `single_match.cluster` against each dataset's own
    ids up front, which is what lets one dataset in the batch fail while its
    neighbors succeed.
    """
    return {
        "global_settings": {"data_source": "byoc", "output_dir": str(output_dir)},
        "byoc_suite": {
            "batteries": ["local"], "input_dir": str(input_dir), "datasets": datasets,
            "cluster_column": "cluster", "standardize": False, "seed": 42,
        },
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": {
                "num_classes": 2, "matching_mode": "single",
                "single_match": {"cluster": 0, "label": 0},
                "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Batch isolation
# ---------------------------------------------------------------------------

def write_byoc_batch(inputs, specs):
    """One CSV per (name, cluster-0 size); every dataset has 40 rows, ids {0, 1}."""
    rng = np.random.default_rng(0)
    for name, n_zero in specs:
        pd.DataFrame({
            "f1": rng.normal(size=40), "f2": rng.normal(size=40),
            "cluster": [0] * n_zero + [1] * (40 - n_zero),
        }).to_csv(inputs / f"{name}.csv", index=False)


def infeasible_config(input_dir, output_dir, datasets):
    """Label 0 must fit entirely inside cluster 0, which not every dataset can do."""
    return {
        "global_settings": {"data_source": "byoc", "output_dir": str(output_dir)},
        "byoc_suite": {
            "batteries": ["local"], "input_dir": str(input_dir), "datasets": datasets,
            "cluster_column": "cluster", "standardize": False, "seed": 42,
        },
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": {
                "num_classes": 2, "balance": "balanced", "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": [0], "recall_target": 1.0}],
                "split_rule": "equal", "spillover_rule": "proportional_to_marginal",
            },
        },
    }


def test_infeasible_allocation_skips_only_the_labelling(tmp_path, monkeypatch):
    """Finding F3: one dataset's failure must not take the batch with it.

    `poison_b`'s cluster 0 holds 3 points where label 0's budget is 20, so it
    raises `InfeasibleAllocationError` ([CLM-150]) while its neighbors succeed.
    That is per-dataset by design, another dataset's cluster sizes may well
    satisfy the same rules, so the batch continues. `good_c` comes *after* the
    failure in iteration order, which is what proves it continued rather than
    merely having finished everything before it.

    Note what "skipped" means here, because it is narrower than it sounds: only
    the *labeling* is skipped. The dataset is still written, still counted in
    `n_ok`, and its CSV simply has no `Label_0` column.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good_a", 20), ("poison_b", 3), ("good_c", 20)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    n_ok = run_pipeline("byoc", infeasible_config(inputs, tmp_path,
                                                  ["good_a", "poison_b", "good_c"]),
                        csv_dir, png_dir, txt_dir)
    assert n_ok == 3

    written = {p.name: list(pd.read_csv(p).columns) for p in csv_dir.glob("*.csv")}
    assert "byoc__local__good_c.csv" in written, \
        "the dataset AFTER the failure was not written: the batch aborted"
    assert "Label_0" in written["byoc__local__good_a.csv"]
    assert "Label_0" in written["byoc__local__good_c.csv"]
    assert "Label_0" not in written["byoc__local__poison_b.csv"], \
        "the dataset whose labelling failed got a label anyway"


def test_byoc_id_mismatch_is_refused_before_any_output(tmp_path, monkeypatch, caplog):
    """`[CLM-104]`/`[CLM-105]` are decided up front for BYOC, as of 0.6.3.

    These two codes are unlike every other coded `[CLM-1xx]`: they compare the
    configuration's ids against *each dataset's own* cluster ids, and under
    `byoc` every CSV brings its own. Before 0.6.3 the mismatch surfaced midway
    through the loop and aborted the run, discarding both the datasets already
    written and the ones that would have succeeded, the scenario F3 was created
    to fix, reintroduced for one class of code.

    Now the whole batch's cluster columns are read before any work begins, so the
    run is refused with nothing written at all, and *every* offending dataset is
    named rather than only the first one reached. Reading one column of each CSV
    is what makes that affordable.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(0)
    for name, cluster_ids in [("good_a", [0, 1]), ("poison_b", [5, 6]),
                              ("poison_c", [7, 8]), ("good_d", [0, 1])]:
        pd.DataFrame({
            "f1": rng.normal(size=40), "f2": rng.normal(size=40),
            "cluster": np.repeat(cluster_ids, 20),
        }).to_csv(inputs / f"{name}.csv", index=False)

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        with pytest.raises(ValueError) as excinfo:
            run_pipeline("byoc", byoc_config(inputs, tmp_path,
                                             ["good_a", "poison_b", "poison_c", "good_d"]),
                         csv_dir, png_dir, txt_dir)

    assert "[CLM-105]" in str(excinfo.value)
    assert not list(csv_dir.glob("*.csv")), \
        "output was written despite the batch being refused"
    for offender in ("poison_b", "poison_c"):
        assert offender in caplog.text, f"{offender} was not named in the refusal"
    assert "2 of 4" in caplog.text, "the refusal did not count the offending datasets"


@pytest.mark.parametrize("name", [
    "../escape", "..\\escape", "sub/dir", "sub\\dir", "C:evil", "..", ".",
], ids=["posix-parent", "windows-parent", "posix-sep", "windows-sep",
        "drive-relative", "dotdot", "dot"])
def test_path_shaped_dataset_names_are_refused(tmp_path, monkeypatch, caplog, name):
    """A dataset name is a file stem, and is used to build paths in both directions.

    byoc resolves `input_dir/<dataset>.csv` to READ and
    `csv/<source>__<battery>__<dataset>.csv` to WRITE, so a separator or a `..`
    reaches outside both configured folders. The registry sources filter names
    against a known list and cannot carry one of these; byoc trusts the config
    verbatim, which is the only route in.

    This is not hypothetical hygiene: configs are shared artifacts here, so
    reproducing published results means running a YAML you did not write.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good", 20)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline("byoc", infeasible_config(inputs, tmp_path, [name, "good"]),
                            csv_dir, png_dir, txt_dir)

    assert n_ok == 1, "the well-named dataset should still be processed"
    assert "is not a plain name" in caplog.text
    written = {p.name for p in csv_dir.rglob("*.csv")}
    assert written == {"byoc__local__good.csv"}, f"unexpected output: {written}"
    # Nothing escaped the run folders, which is the property that matters.
    assert not list(tmp_path.parent.glob("*escape*"))
    assert not list(tmp_path.glob("*escape*"))


def test_non_byoc_id_mismatch_skips_only_that_dataset(tmp_path, monkeypatch):
    """The other sources cannot be pre-checked, so they degrade per dataset.

    Cluster ids there are only knowable by fetching or generating the dataset,
    and doing that twice for every run to gain an early refusal is not a trade
    worth making. So `[CLM-104]`/`[CLM-105]` are reported per dataset and the
    batch continues, `baseline_2class` has ids {0, 1} and cannot satisfy a rule
    naming cluster 3, while `baseline_4class` has ids 0..3 and can.

    Every *other* coded `[CLM-1xx]` still aborts the run; only these two are
    per-dataset, because only these two are statements about the data.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    config = {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": {"batteries": ["fabricated"],
                                  "datasets": ["baseline_2class", "baseline_4class"],
                                  "seed": 42},
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": {"num_classes": 2, "balance": "balanced", "matching_mode": "custom",
                          "assignment_matrix": [{"label": 0, "clusters": [3],
                                                 "recall_target": 0.5}],
                          "split_rule": "equal",
                          "spillover_rule": "proportional_to_marginal"}},
    }

    n_ok = run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir)
    assert n_ok == 1, "the batch did not continue past the mismatched dataset"
    assert {p.name for p in csv_dir.glob("*.csv")} == \
        {"fabricated_data__fabricated__baseline_4class.csv"}


def test_datasets_written_without_a_label_are_reported_separately(tmp_path, monkeypatch, caplog):
    """`n_ok` counts a dataset whose labeling failed, and used to say nothing.

    The dataset IS written, so counting it as processed is right, but a batch
    of ten could report ten successes while several CSVs lacked the `Label_0`
    column that was the entire point of the run. It is logged at ERROR, so it was
    never invisible; the summary simply contradicted it, and the summary is what
    a script reads.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good_a", 20), ("poison_b", 3)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline("byoc", infeasible_config(inputs, tmp_path,
                                                      ["good_a", "poison_b"]),
                            csv_dir, png_dir, txt_dir)

    assert n_ok == 2, "both datasets are written, so both are still processed"
    assert "1 of the 2 processed dataset(s) were written WITHOUT" in caplog.text, \
        f"the unlabelled dataset was not reported. Captured:\n{caplog.text}"


# ---------------------------------------------------------------------------
# Plot failure is reported, but is not a dataset failure
# ---------------------------------------------------------------------------

def test_plot_failure_returns_false_and_leaves_no_file(tmp_path, caplog):
    """Finding N3: a rendering failure must be distinguishable from success.

    A column of Python lists is not plottable, so seaborn raises inside the
    function. It previously caught that, logged one line, and returned `None` --
    the same value it returned on success, so no caller could tell the
    difference. It now returns a bool and tags the line `[PLOT-FAIL]`.
    """
    target = tmp_path / "should_not_exist.png"
    frame = pd.DataFrame({"x": [[1, 2]] * 10, "y": list(range(10)), "hue": [0] * 5 + [1] * 5})

    with caplog.at_level(logging.DEBUG):
        result = plot_feature_scatter(frame, "x", "y", hue_col="hue",
                                      output_path=str(target), title="forced failure")

    assert result is False, "plot failure did not report itself"
    assert not target.exists(), "a partial PNG was left behind"
    assert "[PLOT-FAIL]" in caplog.text


def test_plot_failure_does_not_reduce_the_processed_count(tmp_path, monkeypatch):
    """The project's decision, asserted: the CSV is the deliverable.

    Plotting is best-effort on top of it, so a plot that fails must be logged
    but must not turn a dataset that produced correct labels into a failure.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: False)

    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40),
                  "cluster": np.repeat([0, 1], 20)}).to_csv(inputs / "solo.csv", index=False)

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    n_ok = run_pipeline("byoc", byoc_config(inputs, tmp_path, ["solo"]),
                        csv_dir, png_dir, txt_dir)

    assert n_ok == 1, "a failed plot was counted as a failed dataset"
    assert (csv_dir / "byoc__local__solo.csv").is_file()


def test_figure_creation_failure_escapes_the_functions_own_handler(tmp_path):
    """Finding N4, characterised rather than fixed.

    `fig, ax = plt.subplots(...)` executes BEFORE the function's try/except, so
    a failure there is the one plotting error that does not get the friendly
    "Failed to generate scatter plot" treatment every other failure gets, it
    propagates raw. Not a leak (no figure exists yet to leak), but an
    inconsistent boundary: this path relies entirely on `run_pipeline`'s
    per-dataset guard rather than on the function's own handling.

    Pinned as it behaves NOW. Moving `plt.subplots` inside the `try` makes this
    fail, which is the prompt to replace it with an assertion that the call
    returns False like every other failure.
    """
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "hue": [0, 1, 0]})
    with mock.patch("clmsynth.visualization.plt.subplots",
                    side_effect=RuntimeError("simulated figure-creation failure")),          pytest.raises(RuntimeError):
        plot_feature_scatter(frame, "x", "y", hue_col="hue",
                             output_path=str(tmp_path / "x.png"), title="t")


# ---------------------------------------------------------------------------
# Exit codes: the three outcomes must stay distinguishable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [None, "", "just a string, not a mapping"],
                         ids=["missing-file", "empty-file", "not-a-mapping"])
def test_unloadable_config_exits_one(tmp_path, content):
    """Exit 1 is "ran but produced nothing", distinct from exit 2 for a coded
    config error and 0 for success.

    An empty YAML file parses to `None` and a scalar document parses to a
    string; both would surface much later as an unhelpful `AttributeError` on
    `config.get(...)` without this check.
    """
    path = tmp_path / "cfg.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        load_config(str(path))
    assert excinfo.value.code == 1


def test_unknown_data_source_processes_nothing_without_raising(tmp_path):
    """An unrecognized source is reported and yields zero, not an exception."""
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()
    assert run_pipeline("no_such_source", {}, csv_dir, png_dir, txt_dir) == 0


@pytest.mark.parametrize("clm_label,expected_exit", [
    ({"num_classes": 2, "matching_mode": "perfect"}, 0),
    ({"num_classes": 2, "matching_mode": "bogus_mode"}, 2),
    ({"num_classes": 2, "matching_mode": "custom",
      "assignment_matrix": [{"label": 0, "clusters": [0], "recall_target": 1.0}],
      "split_rule": "equal", "spillover_rule": "proportional_to_marginal"}, 1),
], ids=["success-0", "coded-config-error-2", "nothing-processed-1"])
def test_the_three_exit_codes_stay_distinguishable(tmp_path, clm_label, expected_exit, monkeypatch):
    """0, 1 and 2 mean three different things and must not collapse into each other.

    0.5.0 introduced the 1/2 split deliberately: exit 2 is "the config is wrong,
    nothing was attempted", exit 1 is "ran but produced nothing". A caller
    scripting the CLI distinguishes a mistake it should fix from a run that
    legitimately found no data. Only exit 1 was asserted before; the exit-2 path
    that release created was never verified.

    The third case reaches exit 1 through a dataset whose only cluster cannot
    hold label 0's budget, so labelling is skipped for every dataset and none
    is written.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("solo", 3 if expected_exit == 1 else 20)])

    config = {
        "global_settings": {"data_source": "byoc", "output_dir": str(tmp_path / "out")},
        "byoc_suite": {
            "batteries": ["local"], "input_dir": str(inputs),
            "datasets": ["solo"] if expected_exit != 1 else ["no_such_dataset"],
            "cluster_column": "cluster", "standardize": False, "seed": 42,
        },
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": clm_label,
        },
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth", str(config_path)])

    if expected_exit == 0:
        clmsynth.main.main()
        return

    with pytest.raises(SystemExit) as excinfo:
        clmsynth.main.main()
    assert excinfo.value.code == expected_exit


# ---------------------------------------------------------------------------
# A run that produces nothing must leave nothing
#
# The run folder, its three subfolders and the config copy are all created
# before `run_pipeline` is called, so a run that writes no dataset used to leave
# the whole skeleton behind. Reported from the wizard's byoc path: a config
# naming an input folder with no CSV in it is a perfectly reasonable thing to
# write (the wizard configures, it does not synthesise), but running it produced
# a timestamped folder asserting that a run had happened.
#
# The deletion is the part worth guarding. Only a folder holding exactly the
# scaffolding may be removed, so both directions are asserted below: the barren
# folder goes, and anything with real output in it stays.
# ---------------------------------------------------------------------------

def minimal_byoc_config(input_dir, output_dir, datasets, clm_label=None):
    """A byoc config whose `clm_label` the caller chooses.

    Distinct from `byoc_config` at the top of the module, which pins a
    `scope: pair` single-match because the batch-isolation tests need one
    dataset to fail while its neighbours succeed. These tests care only about
    whether a run produced output, so they want the plainest config that runs
    and, in one case, one that is rejected outright.
    """
    return {
        "global_settings": {"data_source": "byoc", "output_dir": str(output_dir)},
        "byoc_suite": {
            "batteries": ["local"], "input_dir": str(input_dir), "datasets": datasets,
            "cluster_column": "cluster", "standardize": False, "seed": 42,
        },
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": clm_label or {"num_classes": 2, "matching_mode": "perfect"},
        },
    }


def run_main(tmp_path, monkeypatch, config):
    """Drive the CLI over `config` and return (exit code, surviving run folders)."""
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth", str(config_path)])

    out = Path(config["global_settings"]["output_dir"])
    code = 0
    try:
        clmsynth.main.main()
    except SystemExit as exit_signal:
        code = exit_signal.code
    return code, sorted(p.name for p in out.iterdir()) if out.is_dir() else []


def test_a_run_that_writes_nothing_leaves_no_run_folder(tmp_path, monkeypatch):
    """The reported case: byoc pointed at a folder holding no such CSV.

    The fetcher reports the missing file and returns None, so no dataset is
    processed and the run exits 1. What used to survive that was a timestamped
    folder containing an empty csv/, png/ and txt/ and a copy of the config --
    indistinguishable at a glance from a run that succeeded, and picked up by
    anything globbing the output directory.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()

    code, survivors = run_main(
        tmp_path, monkeypatch,
        minimal_byoc_config(inputs, tmp_path / "out", ["no_such_dataset"]))

    assert code == 1, f"expected exit 1 for a run that processed nothing, got {code}"
    assert survivors == [], f"an empty run folder survived: {survivors}"


def test_a_coded_config_error_leaves_no_run_folder(tmp_path, monkeypatch):
    """Exit 2, and the same guarantee.

    `precheck_byoc_matching_ids` documents itself as aborting "before any output
    is written". The run folder and the config copy already existed by the time
    it ran, so that claim was false on the filesystem even though it was true of
    the csv/png/txt payload.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("solo", 20)])

    # Cluster id 9 is in no dataset -> [CLM-105] out of precheck.
    code, survivors = run_main(
        tmp_path, monkeypatch,
        minimal_byoc_config(inputs, tmp_path / "out", ["solo"],
                    clm_label={"num_classes": 2, "balance": "balanced",
                               "matching_mode": "single",
                               "single_match": {"cluster": 9, "label": 0},
                               "spillover_rule": "proportional_to_marginal"}))

    assert code == 2, f"expected exit 2 for a coded config error, got {code}"
    assert survivors == [], f"an empty run folder survived a coded abort: {survivors}"


def test_a_successful_run_keeps_everything_it_wrote(tmp_path, monkeypatch):
    """The other direction, and the one that matters if the predicate is wrong.

    A cleanup that is too eager deletes real output, which is far worse than the
    litter it was written to prevent. Asserted through the same entry point, so
    it is the shipped path being checked and not a helper in isolation.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("solo", 20)])

    code, survivors = run_main(
        tmp_path, monkeypatch, minimal_byoc_config(inputs, tmp_path / "out", ["solo"]))

    assert code == 0, f"expected a clean run, got exit {code}"
    assert len(survivors) == 1, f"the successful run's folder is missing: {survivors}"
    written = sorted(p.name for p in (tmp_path / "out" / survivors[0] / "csv").iterdir())
    assert written, "the run folder survived but its csv/ is empty"


@pytest.mark.parametrize("litter", ["csv/result.csv", "png/plot.png", "stray.txt"],
                         ids=["a-csv", "a-plot", "an-unexpected-file"])
def test_a_run_folder_holding_anything_real_is_never_removed(tmp_path, litter):
    """The predicate directly: one file is enough to make a folder untouchable.

    Parametrized over an expected output and an unexpected file, because the
    rule is not "no output" but "nothing at all beyond the scaffolding". A file
    this code did not put there is someone else's, and guessing about it is how
    a cleanup becomes a data-loss bug.
    """
    run_dir = tmp_path / "090826_BYOC_120000"
    for sub in ("csv", "png", "txt"):
        (run_dir / sub).mkdir(parents=True)
    config_copy = run_dir / "cfg.yaml"
    config_copy.write_text("global_settings: {}", encoding="utf-8")

    target = run_dir / litter
    target.write_text("content", encoding="utf-8")

    assert clmsynth.main.discard_run_dir_if_barren(run_dir, config_copy) is False
    assert run_dir.is_dir(), "a run folder holding real content was removed"
    assert target.is_file(), "the content itself was removed"


def test_the_barren_predicate_accepts_exactly_the_scaffolding(tmp_path):
    """The positive half, stated once on its own.

    Three empty subfolders and the config copy is precisely what `main` creates
    before the pipeline runs; if this stops matching, the cleanup silently stops
    happening and the litter comes back with no test going red.
    """
    run_dir = tmp_path / "090826_BYOC_120000"
    for sub in ("csv", "png", "txt"):
        (run_dir / sub).mkdir(parents=True)
    config_copy = run_dir / "cfg.yaml"
    config_copy.write_text("global_settings: {}", encoding="utf-8")

    assert clmsynth.main.discard_run_dir_if_barren(run_dir, config_copy) is True
    assert not run_dir.exists()


def test_generate_config_exits_one_on_a_missing_payload(tmp_path, monkeypatch):
    """The other console script's failure path, alongside the pipeline's.

    `clmsynth-config` reads a payload file. A missing one must exit 1 with a
    message pointing at the fix, not raise `FileNotFoundError` out of the CLI --
    it is the first command in the documented quick start, so it is the most
    likely place for a user to mistype a path.
    """
    from clmsynth import generate_config

    monkeypatch.setattr(sys, "argv", ["clmsynth-config", str(tmp_path / "no_such_payload.yaml")])
    with pytest.raises(SystemExit) as excinfo:
        generate_config.main()
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Row alignment: the guarantee the whole dataset format rests on
# ---------------------------------------------------------------------------

def test_context_refuses_a_misaligned_generated_label():
    """`DatasetContext` must reject a column that is not row-aligned.

    The project's central promise is that a generated dataset carries three
    things *strictly row-aligned*: the features, the ground-truth cluster ids,
    and the labels. If a short or long series could be attached, every MCC and
    ARI computed from the written CSV would be comparing misaligned rows, and
    nothing downstream would notice, the file would look entirely well-formed.

    This guard is the only thing enforcing that, and nothing in the suite
    exercised it.
    """
    features = pd.DataFrame({"f1": range(10), "f2": range(10)})
    context = DatasetContext("src", "battery", features,
                             ground_truths={"labels0": pd.Series(range(10))})

    for bad_length in (9, 11):
        with pytest.raises(ValueError, match="misaligned"):
            context.add_generated_label("Label_0", pd.Series(range(bad_length)))


def test_context_accepts_an_aligned_label_and_ignores_its_index():
    """The other half: correct length is accepted, and a foreign index is reset.

    A series carrying its own index, which is what any filtered or reordered
    pandas operation produces, must be re-based rather than rejected or, worse,
    silently aligned by index and scrambled.
    """
    features = pd.DataFrame({"f1": range(10), "f2": range(10)})
    context = DatasetContext("src", "battery", features,
                             ground_truths={"labels0": pd.Series(range(10))})

    reindexed = pd.Series(range(10), index=range(100, 110))
    context.add_generated_label("Label_0", reindexed)

    frame = context.to_dataframe()
    assert list(frame["Label_0"]) == list(range(10))
    assert len(frame) == 10


# ---------------------------------------------------------------------------
# Network timeout
# ---------------------------------------------------------------------------

def test_clustbench_timeout_is_passed_to_urlopen():
    """Finding F4: the configured timeout must reach the call.

    The regression is a fetch with no timeout at all, which inherits the OS
    default connect timeout and looks like a
    hang. Asserted by patching `urlopen` and reading the keyword it received,
    which is deterministic and instant. Whether an actually-hung socket then
    honors it is the OS's behavior, not this package's.
    """
    captured = {}

    def fake_urlopen(url, timeout=None):
        captured["timeout"] = timeout
        raise OSError("not connecting in a test")

    with mock.patch("clmsynth.dataset_sources.urllib.request.urlopen", fake_urlopen),          pytest.raises(OSError):
        dataset_sources._loadtxt_url("http://example.invalid/data.txt")

    assert captured["timeout"] == dataset_sources.CLUSTBENCH_TIMEOUT
    assert captured["timeout"] is not None, "fetch would inherit the OS default timeout"


def test_clustbench_fetch_returns_none_when_the_source_is_unreachable():
    """A failed fetch is a skipped dataset, not an exception out of the batch.

    Patched rather than pointed at a real unreachable host, so it is instant
    and does not depend on the runner having (or not having) a network.
    """
    with mock.patch("clmsynth.dataset_sources.urllib.request.urlopen",
                    side_effect=OSError("unreachable")):
        assert dataset_sources.fetch_clustbench_data(
            dataset_group="wut", dataset_name="smile",
            base_url="http://example.invalid/benchmark") is None
