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
  error boundary       a failure creating the figure returns False like
                       every other plotting failure (finding N4, closed in
                       0.6.9)
  threads              a saved plot does not go through pyplot, so plots
                       rendered from worker threads match sequential ones
  interactive branch   omitting `output_path` shows and keeps the figure
  exit codes           0 every dataset written with all its labels, 3 only
                       some, 2 a missing or malformed config or a coded config
                       error, 1 nothing written for another reason, such as a
                       missing data CSV or an output_dir blocked by a file,
                       which used to hang on Windows; SIGTERM exits 143
  --help               `clmsynth-config` and `clmsynth-wizard` print their usage
                       and exit 0, writing nothing and asking nothing
  network timeout      `CLUSTBENCH_TIMEOUT` is passed to `urlopen`
  mdcgen presets       each preset passes only names mdcgenpy reads, with the
                       size and dimension it names; a run records which
                       features are noise

"""

import errno
import json
import logging
import os
import re
import signal
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import clmsynth.__main__
import clmsynth._extras.visualization
import clmsynth._single_run
from clmsynth import dataset_sources, load_config, run_pipeline
from clmsynth._extras.visualization import plot_feature_scatter
from clmsynth._input_sources import clustbench_source
from clmsynth._input_sources.mdcgen_source import MDCGEN_CONFIGS
from clmsynth._single_run import ConfigFileError, run_single
from clmsynth._utils.write_files import discard_failed_run, resolved_for_report
from clmsynth.labeling import DatasetContext, generate_additional_labels

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
            "batteries": ["local"],
            "input_dir": str(input_dir),
            "datasets": datasets,
            "cluster_column": "cluster",
            "standardize": False,
            "seed": 42,
        },
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": {
                "num_classes": 2,
                "matching_mode": "single",
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
        pd.DataFrame(
            {
                "f1": rng.normal(size=40),
                "f2": rng.normal(size=40),
                "cluster": [0] * n_zero + [1] * (40 - n_zero),
            }
        ).to_csv(inputs / f"{name}.csv", index=False)


def infeasible_config(input_dir, output_dir, datasets):
    """Label 0 must fit entirely inside cluster 0, which not every dataset can do."""
    return {
        "global_settings": {"data_source": "byoc", "output_dir": str(output_dir)},
        "byoc_suite": {
            "batteries": ["local"],
            "input_dir": str(input_dir),
            "datasets": datasets,
            "cluster_column": "cluster",
            "standardize": False,
            "seed": 42,
        },
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": [0], "recall_target": 1.0}],
                "split_rule": "equal",
                "spillover_rule": "proportional_to_marginal",
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
    merely having finished everything before it. only
    the *labeling* is skipped. The dataset is still written, still counted in
    `n_ok`, and its CSV simply has no `Label_0` column.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good_a", 20), ("poison_b", 3), ("good_c", 20)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    n_ok = run_pipeline(
        "byoc",
        infeasible_config(inputs, tmp_path, ["good_a", "poison_b", "good_c"]),
        csv_dir,
        png_dir,
        txt_dir,
    )
    assert n_ok == 3

    written = {p.name: list(pd.read_csv(p).columns) for p in csv_dir.glob("*.csv")}
    assert "byoc-local-good_c.csv" in written, (
        "the dataset AFTER the failure was not written: the batch aborted"
    )
    assert "Label_0" in written["byoc-local-good_a.csv"]
    assert "Label_0" in written["byoc-local-good_c.csv"]
    assert "Label_0" not in written["byoc-local-poison_b.csv"], (
        "the dataset whose labelling failed got a label anyway"
    )


def write_cluster_sizes(inputs, specs):
    """One CSV per (name, cluster sizes), ids 0..K-1 in order, two random features."""
    rng = np.random.default_rng(0)
    for name, sizes in specs:
        n = sum(sizes)
        pd.DataFrame(
            {
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
                "cluster": np.repeat(np.arange(len(sizes)), sizes),
            }
        ).to_csv(inputs / f"{name}.csv", index=False)


def crowded_cluster_label(target_metric=None):
    """Four balanced labels, rule i -> i, and two competing_noise entries that
    each claim 90% of what the rules leave unclaimed in cluster 0."""
    clm_label: dict[str, Any] = {
        "num_classes": 4,
        "balance": "balanced",
        "matching_mode": "custom",
        "assignment_matrix": [
            {"label": i, "clusters": [i], "recall_target": 0.8} for i in range(4)
        ],
        "competing_noise": [
            {"cluster": 0, "label": 1, "share": 0.9, "favors": "random"},
            {"cluster": 0, "label": 2, "share": 0.9, "favors": "random"},
        ],
    }
    if target_metric is not None:
        clm_label["target_metric"] = target_metric
    return clm_label


def test_a_target_no_level_fits_leaves_out_only_that_datasets_label(
    tmp_path, monkeypatch, caplog
):
    """[CLM-120] used to abort the whole run (exit 2), as if the config were wrong
    for every dataset. It is judged against one dataset's cluster sizes.

    `blocked` has 53 points in cluster 0, more than label 0's 50, so its rule
    never fills the cluster and every recall level is infeasible: [CLM-152] until
    label 1 needs more than cluster 1's 47 points, then [CLM-150]. `fits`, four
    clusters of 50, is feasible at alpha 1. Now [CLM-120] leaves out only
    `blocked`'s label: it is still written and counted, without `Label_0`, and
    `fits`, after it in iteration order, gets its label.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_cluster_sizes(inputs, [("blocked", [53, 47, 50, 50]), ("fits", [50, 50, 50, 50])])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    config = minimal_byoc_config(
        inputs, tmp_path, ["blocked", "fits"], crowded_cluster_label({"type": "mcc", "value": 0.5})
    )
    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline("byoc", config, csv_dir, png_dir, txt_dir)

    assert n_ok == 2, "both datasets are written, so both are processed"
    written = {p.name: list(pd.read_csv(p).columns) for p in csv_dir.glob("*.csv")}
    assert "Label_0" not in written["byoc-local-blocked.csv"], written
    assert "Label_0" in written["byoc-local-fits.csv"], written
    assert "Skipping label generation for local/blocked: [CLM-120]" in caplog.text, caplog.text
    assert "aborting run" not in caplog.text, caplog.text


def test_over_claiming_noise_without_a_target_leaves_out_only_the_label(
    tmp_path, monkeypatch, caplog
):
    """[CLM-152] without target_metric is unchanged by [CLM-120] becoming per
    dataset: it was always an InfeasibleAllocationError that leaves out the
    dataset's label.

    Four clusters of 50 and the noise above, at the configured recall 0.8: the
    rule leaves 10 of cluster 0's 50 points unclaimed and the entries claim 9
    each. The dataset is written and counted, without `Label_0`.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_cluster_sizes(inputs, [("over_claimed", [50, 50, 50, 50])])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    config = minimal_byoc_config(inputs, tmp_path, ["over_claimed"], crowded_cluster_label())
    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline("byoc", config, csv_dir, png_dir, txt_dir)

    assert n_ok == 1
    columns = list(pd.read_csv(csv_dir / "byoc-local-over_claimed.csv").columns)
    assert "Label_0" not in columns, columns
    assert "Skipping label generation for local/over_claimed: [CLM-152]" in caplog.text, (
        caplog.text
    )


def test_byoc_id_mismatch_is_refused_before_any_output(tmp_path, monkeypatch, caplog):
    """`[CLM-105]` is decided up front for BYOC, as of 0.6.3.

    Unlike every other coded `[CLM-1xx]`, it compares the configuration's ids
    against *each dataset's own* cluster ids, and under `byoc` every CSV brings
    its own.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(0)
    for name, cluster_ids in [
        ("good_a", [0, 1]),
        ("poison_b", [5, 6]),
        ("poison_c", [7, 8]),
        ("good_d", [0, 1]),
    ]:
        pd.DataFrame(
            {
                "f1": rng.normal(size=40),
                "f2": rng.normal(size=40),
                "cluster": np.repeat(cluster_ids, 20),
            }
        ).to_csv(inputs / f"{name}.csv", index=False)

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        with pytest.raises(ValueError) as excinfo:
            run_pipeline(
                "byoc",
                byoc_config(inputs, tmp_path, ["good_a", "poison_b", "poison_c", "good_d"]),
                csv_dir,
                png_dir,
                txt_dir,
            )

    assert "[CLM-105]" in str(excinfo.value)
    assert not list(csv_dir.glob("*.csv")), "output was written despite the batch being refused"
    for offender in ("poison_b", "poison_c"):
        assert offender in caplog.text, f"{offender} was not named in the refusal"
    assert "2 of 4" in caplog.text, "the refusal did not count the offending datasets"


def test_byoc_label_out_of_range_is_reported_once_not_per_dataset(tmp_path, caplog):
    """[CLM-104] is a configuration error, reported once."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(0)
    for name in ("a", "b", "c", "d"):
        pd.DataFrame(
            {
                "f1": rng.normal(size=40),
                "f2": rng.normal(size=40),
                "cluster": np.repeat([0, 1], 20),
            }
        ).to_csv(inputs / f"{name}.csv", index=False)

    config = byoc_config(inputs, tmp_path, ["a", "b", "c", "d"])
    config["label_generation"]["clm_label"]["single_match"]["label"] = 9

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.CRITICAL, logger="clmsynth"):
        with pytest.raises(ValueError) as excinfo:
            run_pipeline("byoc", config, csv_dir, png_dir, txt_dir)

    assert "[CLM-104]" in str(excinfo.value)
    assert caplog.text.count("[CLM-104]") == 1, (
        f"104 was reported per dataset, not once:\n{caplog.text}"
    )
    assert "of 4 BYOC dataset(s)" not in caplog.text, (
        "the batch summary blamed the datasets for a configuration error"
    )


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "..\\escape",
        "sub/dir",
        "sub\\dir",
        "C:evil",
        "..",
        ".",
    ],
    ids=[
        "posix-parent",
        "windows-parent",
        "posix-sep",
        "windows-sep",
        "drive-relative",
        "dotdot",
        "dot",
    ],
)
def test_path_shaped_dataset_names_are_refused(tmp_path, monkeypatch, caplog, name):
    """A dataset name is a file stem, and is used to build paths in both directions.

    byoc resolves `input_dir/<dataset>.csv` to READ and
    `csv/<source>-<battery>-<dataset>.csv` to WRITE, so a separator or a `..`
    reaches outside both configured folders. The registry sources filter names
    against a known list and cannot carry one of these; byoc trusts the config
    verbatim, which is the only route in. Such a name refuses the run before its
    first dataset.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good", 20)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline(
            "byoc", infeasible_config(inputs, tmp_path, [name, "good"]), csv_dir, png_dir, txt_dir
        )

    assert n_ok == 0, "a path-shaped name refuses the run"
    assert "not a plain name" in caplog.text
    written = {p.name for p in csv_dir.rglob("*.csv")}
    assert not written, f"unexpected output: {written}"
    # Nothing escaped the run folders, which is the property that matters.
    assert not list(tmp_path.parent.glob("*escape*"))
    assert not list(tmp_path.glob("*escape*"))


def test_non_byoc_id_mismatch_skips_only_that_dataset(tmp_path, monkeypatch):
    """The other sources cannot be pre-checked, so they degrade per dataset.

    Cluster ids there are only knowable by fetching or generating the dataset,
    and doing that twice for every run to gain an early refusal is not a trade
    worth making. So `[CLM-105]` is reported per dataset and the
    batch continues, `baseline_2class` has ids {0, 1} and cannot satisfy a rule
    naming cluster 3, while `baseline_4class` has ids 0..3 and can.

    Every *other* coded `[CLM-1xx]` aborts the run unless it too is a statement
    about the data: `[CLM-102]`, `[CLM-119]`, `[CLM-125]`, `[CLM-127]` and
    `[CLM-133]` judge the dataset's K, ids or features and are per-dataset for
    the same reason, and `[CLM-120]`, judged against its cluster sizes, leaves
    out only that dataset's label. `[CLM-104]` is not, its bound is num_classes,
    which no dataset can change.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    config: dict[str, Any] = {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": {
            "batteries": ["fabricated"],
            "datasets": ["baseline_2class", "baseline_4class"],
            "seed": 42,
        },
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": [3], "recall_target": 0.5}],
                "split_rule": "equal",
                "spillover_rule": "proportional_to_marginal",
            },
        },
    }

    n_ok = run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir)
    assert n_ok == 1, "the batch did not continue past the mismatched dataset"
    assert {p.name for p in csv_dir.glob("*.csv")} == {
        "fabricated_data-fabricated-baseline_4class.csv"
    }


def test_datasets_written_without_a_label_are_reported_separately(tmp_path, monkeypatch, caplog):
    """`n_ok` counts a dataset whose labeling failed, and used to say nothing.

    The dataset IS written, so counting it as processed is right, but a batch
    of ten could report ten successes while several CSVs lacked the `Label_0`
    column that was the entire point of the run. It is logged at ERROR, so it was
    never invisible; the summary simply contradicted it, and the summary is what
    a script reads.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good_a", 20), ("poison_b", 3)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline(
            "byoc",
            infeasible_config(inputs, tmp_path, ["good_a", "poison_b"]),
            csv_dir,
            png_dir,
            txt_dir,
        )

    assert n_ok == 2, "both datasets are written, so both are still processed"
    assert ("Of the 2 resolved dataset(s): 1 written with all labels, 1 written with not "
            "all labels, 0 skipped") in caplog.text, (
        f"the unlabelled dataset was not reported. Captured:\n{caplog.text}"
    )


def test_a_dataset_short_of_its_later_labels_is_not_reported_as_unlabelled(tmp_path, caplog):
    """Each label has its own seed, so a later label can fail where the first fitted.

    Label 0 must sit entirely in cluster 0 (20 of 40 points), and its size is a
    Dirichlet draw per label seed: seeds 2, 3 and 4 give it 15, 9 and 36 points.
    `Label_0` and `Label_1` are written and `Label_2` raises [CLM-150]. The
    summary used to say the CSV held features and ground-truth clusters only.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("three_labels", 20)])
    config = infeasible_config(inputs, tmp_path, ["three_labels"])
    config["label_generation"].update(n_labels=3, seed=2)
    config["label_generation"]["clm_label"].update(
        balance="unbalanced", skew_rule="dirichlet", skew_params={"alpha": 1.0})

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline("byoc", config, csv_dir, png_dir, txt_dir, False)

    assert n_ok == 1, "the dataset is written, so it is still processed"
    columns = list(pd.read_csv(csv_dir / "byoc-local-three_labels.csv").columns)
    assert columns[-2:] == ["Label_0", "Label_1"] and "Label_2" not in columns, columns
    assert ("Skipping label generation for local/three_labels after 2 of 3 labels: "
            "[CLM-150]") in caplog.text, caplog.text
    assert ("Of the 1 resolved dataset(s): 0 written with all labels, 1 written with not "
            "all labels, 0 skipped") in caplog.text, caplog.text


def test_an_unexpected_error_names_its_type_and_the_skip_is_counted(tmp_path, monkeypatch, caplog):
    """An exception nobody anticipated skips its dataset alone, and says what it was.

    `str(KeyError('cluster'))` is just `'cluster'`: without the type the log line
    names a word, not a failure. The end-of-run summary counts the skip, because
    `n_ok` alone cannot tell "1 of 1" from "1 of 2".
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)
    real_build_context = clmsynth._single_run.build_context

    def build_context(source, battery, dataset, df):
        if dataset == "broken":
            raise KeyError("cluster")
        return real_build_context(source, battery, dataset, df)

    monkeypatch.setattr(clmsynth._single_run, "build_context", build_context)

    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("good", 20), ("broken", 20)])

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        n_ok = run_pipeline(
            "byoc", byoc_config(inputs, tmp_path, ["good", "broken"]), csv_dir, png_dir, txt_dir
        )

    assert n_ok == 1, "the healthy dataset must still be processed"
    assert "Skipping local/broken: unexpected KeyError: 'cluster'" in caplog.text, caplog.text
    assert ("Of the 2 resolved dataset(s): 1 written with all labels, 0 written with not "
            "all labels, 1 skipped") in caplog.text, caplog.text


# ---------------------------------------------------------------------------
# Plot failure is reported, but is not a dataset failure
# ---------------------------------------------------------------------------


def test_plot_failure_returns_false_and_leaves_no_file(tmp_path, caplog):
    """Finding N3: a rendering failure must be distinguishable from success.

    A column of Python lists is not plottable, so matplotlib raises inside the
    function. It previously caught that, logged one line, and returned `None` --
    the same value it returned on success, so no caller could tell the
    difference. It now returns a bool and tags the line `[PLOT-FAIL]`.
    """
    target = tmp_path / "should_not_exist.png"
    frame = pd.DataFrame({"x": [[1, 2]] * 10, "y": list(range(10)), "hue": [0] * 5 + [1] * 5})

    with caplog.at_level(logging.DEBUG):
        result = plot_feature_scatter(
            frame, "x", "y", hue_col="hue", output_path=str(target), title="forced failure"
        )

    assert result is False, "plot failure did not report itself"
    assert not target.exists(), "a partial PNG was left behind"
    assert "[PLOT-FAIL]" in caplog.text


def test_plot_failure_does_not_reduce_the_processed_count(tmp_path, monkeypatch):
    """The project's decision, asserted: the CSV is the deliverable.

    Plotting is best-effort on top of it, so a plot that fails must be logged
    but must not turn a dataset that produced correct labels into a failure.
    """
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: False)

    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {"f1": rng.normal(size=40), "f2": rng.normal(size=40), "cluster": np.repeat([0, 1], 20)}
    ).to_csv(inputs / "solo.csv", index=False)

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    n_ok = run_pipeline("byoc", byoc_config(inputs, tmp_path, ["solo"]), csv_dir, png_dir, txt_dir)

    assert n_ok == 1, "a failed plot was counted as a failed dataset"
    assert (csv_dir / "byoc-local-solo.csv").is_file()


def test_figure_creation_failure_returns_false_like_every_other_failure(tmp_path):
    """Finding N4, closed in 0.6.9. Regression pin."""
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "hue": [0, 1, 0]})
    with mock.patch(
        "clmsynth._extras.visualization.Figure",
        side_effect=RuntimeError("simulated figure-creation failure"),
    ):
        assert (
            plot_feature_scatter(
                frame, "x", "y", hue_col="hue", output_path=str(tmp_path / "x.png"), title="t"
            )
            is False
        )


def test_a_dollar_in_a_name_is_text_not_a_formula(tmp_path):
    """Names reached matplotlib as mathtext: `x$_$y`, a column name byoc accepts,
    was a parse error, and the dataset lost its plots."""
    frame = pd.DataFrame({"x$_$y": [0.0, 1.0, 2.0], "f2": [0.0, 1.0, 0.5],
                          "Cluster_0": [0, 1, 1]})
    target = tmp_path / "plot.png"
    assert plot_feature_scatter(frame, "x$_$y", "f2", hue_col="Cluster_0",
                                output_path=str(target))
    assert target.is_file()


def test_plot_run_folder_draws_and_counts_what_the_run_would(tmp_path, monkeypatch):
    """One feature and a tag: the run skips the plots. Read back from its folder,
    the tag counted as a second feature and was drawn against the first, and the
    dataset was counted as plotted. The summary says how many columns are
    features."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    pd.DataFrame({"f1": np.arange(20) / 10, "site": ["north", "south"] * 10,
                  "cluster": np.repeat([0, 1], 10)}).to_csv(inputs / "data.csv", index=False)
    config = minimal_byoc_config(inputs, tmp_path / "out", ["data"])
    config["byoc_suite"]["tag_columns"] = ["site"]
    code, survivors = run_main(tmp_path, monkeypatch, config, "--no-viz")
    assert code == 0

    drawn = []

    def record(*args, **kwargs):
        drawn.append(kwargs)
        return True

    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", record)
    run_dir = tmp_path / "out" / survivors[0]
    assert clmsynth._extras.visualization.plot_run_folder(run_dir) == 0
    assert not drawn


# ---------------------------------------------------------------------------
# The interactive branch: a shown figure belongs to the viewer
# ---------------------------------------------------------------------------


def test_interactive_plot_shows_and_keeps_the_figure_open(monkeypatch):
    """Omitting output_path shows and keeps the figure open."""
    show = mock.MagicMock()
    monkeypatch.setattr("clmsynth._extras.visualization.plt.show", show)
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 2.0, 3.0], "hue": [0, 1, 0, 1]}
    )

    before = set(plt.get_fignums())
    result = plot_feature_scatter(frame, "x", "y", hue_col="hue", title="interactive")
    opened = set(plt.get_fignums()) - before

    try:
        assert result is True, "the interactive branch did not report success"
        assert show.call_count == 1, "a pathless render did not call plt.show()"
        assert opened, "the shown figure was closed under the viewer"
    finally:
        for num in opened:
            plt.close(num)


# ---------------------------------------------------------------------------
# Saved plots and threads
# ---------------------------------------------------------------------------


def test_saved_plot_does_not_go_through_pyplot(monkeypatch, tmp_path):
    """pyplot keeps one figure list per process and picks the backend, often a
    GUI one that only works from the main thread; a saved plot uses neither."""
    monkeypatch.setattr(clmsynth._extras.visualization, "plt", None)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 3.0, 2.0], "hue": [0, 1, 0]})
    saved = plot_feature_scatter(frame, "x", "y", hue_col="hue", info_text="info",
                                 output_path=str(tmp_path / "x.png"))
    assert saved, "a saved plot reached pyplot"


def test_plots_from_worker_threads_match_sequential_ones(tmp_path):
    """Four plots drawn at once from worker threads are pixel-identical to the same
    four drawn one after another."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200),
                          "hue": rng.integers(0, 5, size=200)})

    def draw(folder: str, i: int) -> bool:
        return plot_feature_scatter(frame, "x", "y", hue_col="hue", title=f"plot {i}",
                                    info_text=f"info {i}",
                                    output_path=str(tmp_path / folder / f"{i}.png"))

    sequential = [draw("sequential", i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(pool.map(lambda i: draw("threaded", i), range(4)))

    assert all(sequential) and all(threaded), "a plot failed"
    for i in range(4):
        same = np.array_equal(mpimg.imread(tmp_path / "sequential" / f"{i}.png"),
                              mpimg.imread(tmp_path / "threaded" / f"{i}.png"))
        assert same, f"plot {i} drawn in a thread differs from the sequential one"


# ---------------------------------------------------------------------------
# Exit codes: the outcomes must stay distinguishable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,content",
    [("cfg.yaml", None), ("cfg.yaml", ""), ("cfg.yaml", "just a string, not a mapping"),
     ("cfg.yaml", "global_settings: [unclosed"),
     ("cfg.json", ""), ("cfg.json", '"just a string, not a mapping"'),
     ("cfg.json", '{"global_settings": [unclosed')],
    ids=["missing-file", "empty-file", "not-a-mapping", "malformed-yaml",
         "empty-json", "json-not-a-mapping", "malformed-json"],
)
def test_unloadable_config_exits_two(tmp_path, request, name, content):
    """Exit 2: the config given on the command line is wrong, as for a coded
    config error.

    An empty file parses to `None` and a scalar document parses to a string;
    both would surface much later as an unhelpful `AttributeError` on
    `config.get(...)` without this check. A syntax error used to escape as a
    `yaml.YAMLError` traceback.
    """
    if name.endswith(".yaml") and content is not None:
        request.getfixturevalue("yaml")   # without PyYAML a .yaml exits 1 before it is read
    path = tmp_path / name
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigFileError):
        load_config(str(path))
    assert run_single(str(path)) == 2


@pytest.mark.parametrize("name,encoding", [("cfg.json", "utf-16"), ("cfg.yaml", "latin-1")],
                         ids=["utf-16-json", "latin-1-yaml"])
def test_a_config_that_is_not_utf8_exits_two(tmp_path, request, name, encoding):
    """PowerShell 5.1's `>` writes UTF-16 and an editor may save Latin-1: the
    decoding error is a config file error, not a traceback."""
    if name.endswith(".yaml"):
        request.getfixturevalue("yaml")
    path = tmp_path / name
    path.write_bytes('{"note": "café"}'.encode(encoding))

    with pytest.raises(ConfigFileError, match="not UTF-8"):
        load_config(str(path))
    assert run_single(str(path)) == 2


@pytest.mark.parametrize("value", [[1, 2], "byoc"], ids=["list", "string"])
def test_a_global_settings_that_is_not_a_mapping_exits_two(value, tmp_path, monkeypatch, caplog):
    """It crashed on its first .get() with a traceback. It exits 2 like a config
    that cannot be loaded, naming the section, before anything is created."""
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"global_settings": value}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth", str(config_path)])
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.CRITICAL, logger="clmsynth"), pytest.raises(SystemExit) as excinfo:
        clmsynth.__main__.main()
    assert excinfo.value.code == 2
    assert "'global_settings' must be a mapping" in caplog.text, caplog.text
    assert [p.name for p in tmp_path.iterdir()] == ["cfg.json"], "the run created something"


def test_unknown_data_source_processes_nothing_without_raising(tmp_path):
    """An unrecognized source is reported and yields zero, not an exception."""
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()
    assert run_pipeline("no_such_source", {}, csv_dir, png_dir, txt_dir) == 0


@pytest.mark.parametrize(
    "suite",
    [
        {"datasets": "all", "seed": 42},
        {"batteries": None, "datasets": "all", "seed": 42},
        {"batteries": [], "datasets": "all", "seed": 42},
    ],
    ids=["key-absent", "key-null", "key-empty-list"],
)
def test_unset_batteries_stops_the_run_and_names_the_key(suite, tmp_path, caplog):
    """Unset batteries stops the run. Deliberately uncoded."""
    config = {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": suite,
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": {"num_classes": 4, "matching_mode": "perfect"},
        },
    }
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir) == 0

    assert "fabricated_data_suite.batteries" in caplog.text, caplog.text
    assert not list(csv_dir.glob("*.csv")), "the run did work despite an unset key"


def _fabricated_config(tmp_path, suite, label_seed: int | None = 42):
    """A one-dataset fabricated run; returns (config, csv_dir, png_dir, txt_dir)."""
    config = {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": suite,
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "seed": label_seed,
            "clm_label": {"num_classes": 4, "matching_mode": "perfect"},
        },
    }
    dirs = [tmp_path / "csv", tmp_path / "png", tmp_path / "txt"]
    for d in dirs:
        d.mkdir()
    return (config, *dirs)


@pytest.mark.parametrize(
    "key,suite",
    [
        ("batteries", {"batteries": "fabricated", "datasets": ["baseline_4class"]}),
        ("datasets", {"batteries": ["fabricated"], "datasets": "baseline_4class"}),
    ],
    ids=["batteries", "datasets"],
)
def test_a_string_batteries_or_datasets_stops_the_run_and_names_the_key(
    key, suite, tmp_path, caplog
):
    """A string other than "all" used to be iterated letter by letter.
    datasets "mydata" became m, y, d, a, t, a; batteries "fabricated" matched
    nothing. Refused like an unset batteries, naming the key."""
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, dict(suite, seed=42))
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 0
    assert f"fabricated_data_suite.{key}" in caplog.text, caplog.text
    assert not list(csv_dir.glob("*.csv"))


_ONE_DATASET = {"batteries": ["fabricated"], "datasets": ["baseline_4class"], "seed": 42}


@pytest.mark.parametrize(
    "change,said",
    [
        ({"batteries": 5}, "'fabricated_data_suite.batteries' is 5"),
        ({"datasets": 5}, "'fabricated_data_suite.datasets' is 5"),
        ({"datasets": False}, "'fabricated_data_suite.datasets' is False"),
        ({"datasets": None}, "'fabricated_data_suite.datasets' not set"),
        ({"datasets": []}, "'fabricated_data_suite.datasets' not set"),
        ({"datasets": "absent"}, "'fabricated_data_suite.datasets' not set"),
        ({"datasets": ["baseline_4class", "baselin_2class"]},
         "'baselin_2class' (did you mean 'baseline_2class'?)"),
        ({"batteries": ["fabricated", "fabricatd"]},
         "'fabricatd' (did you mean 'fabricated'?)"),
    ],
    ids=["batteries-a-number", "datasets-a-number", "datasets-false", "datasets-valueless",
         "datasets-empty", "datasets-absent", "dataset-misspelt", "battery-misspelt"],
)
def test_a_selection_the_source_does_not_have_refuses_the_run(change, said, tmp_path, caplog):
    """`batteries` and `datasets` are "all" or a list of names the source has. A
    misspelt name beside a good one used to be dropped and the run exited 0; a
    number exited 2 without a word; an unset or false `datasets` meant all."""
    suite = {**_ONE_DATASET, **change}
    if change.get("datasets") == "absent":
        del suite["datasets"]
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, suite)
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 0
    assert said in caplog.text, caplog.text
    assert not list(csv_dir.glob("*.csv"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("fabricated_data_suite", ["fabricated"]),
        ("fabricated_data_suite", "fabricated"),
        ("label_generation", [1, 2]),
        ("label_generation", 5),
    ],
    ids=["suite-list", "suite-string", "label-generation-list", "label-generation-int"],
)
def test_a_section_that_is_not_a_mapping_stops_the_run_and_names_it(key, value, tmp_path, caplog):
    """A list or a string crashed on its first .get() with a traceback, which left
    the run folder behind; an integer label_generation exited 2 without a word.
    Refused like an unset batteries, naming the section."""
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, dict(_ONE_DATASET))
    config[key] = value
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 0
    assert f"'{key}' must be a mapping" in caplog.text, caplog.text
    assert not list(csv_dir.glob("*.csv"))


@pytest.mark.parametrize("value", [[], [1, 2], "x"], ids=["empty-list", "list", "string"])
def test_an_unused_suite_of_any_shape_is_not_read(value, tmp_path):
    """Only the sections a run reads are checked."""
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, dict(_ONE_DATASET))
    config["mdcgen_suite"] = value
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 1


@pytest.mark.parametrize(
    "clm_label,shown,hidden",
    [
        (
            {
                "num_classes": 4,
                "matching_mode": "perfect",
                "balance": "unbalanced",
                "proportions": ["a", "b", "c", "d"],
            },
            "sizes: from clusters",
            "proportions",
        ),
        (
            {
                "num_classes": 4,
                "matching_mode": "random",
                "balance": "balanced",
                "proportions": [0.7, 0.1, 0.1, 0.1],
                "centroid_dependence": {"enabled": True, "favors": "boundary"},
            },
            "centrality: off",
            "0.7",
        ),
        (
            {
                "num_classes": 4,
                "matching_mode": "single",
                "single_match": {"cluster": 0, "label": 0},
            },
            "balance: balanced",
            "balance: -",
        ),
    ],
    ids=["perfect-unread-proportions", "random-balanced", "balance-left-to-its-default"],
)
def test_the_plot_box_shows_only_what_the_label_was_made_with(
    clm_label, shown, hidden, tmp_path, monkeypatch
):
    """The info box beside a label plot listed proportions 'perfect' never reads
    (and crashed converting them, after the CSV was written, so the dataset was
    counted as skipped), proportions 'balanced' ignores, and a centrality
    'random' never applies. A balance left to its default read `balance: -`."""
    boxes = []

    def record(*args, **kwargs):
        boxes.append(kwargs.get("info_text"))
        return True

    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", record)
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, dict(_ONE_DATASET))
    config["label_generation"]["clm_label"] = clm_label
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir) == 1
    label_box = next(box for box in boxes if box)
    assert shown in label_box and hidden not in label_box, label_box


@pytest.mark.parametrize("where", ["suite", "label_generation"])
def test_a_valueless_seed_draws_one_and_reports_it(where, tmp_path, caplog):
    """A valueless `seed:` asks for randomness. The label seed crashed on
    None + 0, and the suite seed reached the generator as None, so the log said
    'Seed: None' and the data could not be reproduced. One integer is now drawn
    and logged, so rerunning with it reproduces the run."""
    suite = {"batteries": ["fabricated"], "datasets": ["baseline_4class"], "seed": 42}
    if where == "suite":
        suite["seed"] = None
    config, csv_dir, png_dir, txt_dir = _fabricated_config(
        tmp_path, suite, label_seed=None if where == "label_generation" else 42
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 1
    name = "fabricated_data_suite.seed" if where == "suite" else "label_generation.seed"
    match = re.search(
        rf"'{re.escape(name)}' has no value: using the randomly drawn seed (\d+)", caplog.text
    )
    assert match, caplog.text
    # The log can be lost; the summary txt keeps the drawn value with the run.
    summary = next(txt_dir.glob("*.txt")).read_text(encoding="utf-8")
    assert f"{name} = {match.group(1)} (drawn at random; the key has no value)" in summary


@pytest.mark.parametrize(
    "label_seed,expected",
    [
        (7, "label_generation.seed = 7 (set in the config)"),
        ("absent", "label_generation.seed = 42 (default; the key is absent)"),
    ],
    ids=["set", "absent"],
)
def test_the_summary_txt_records_every_seed(label_seed, expected, tmp_path):
    """Both seeds, however each was decided, and each label's own seed (label
    i uses seed + i), so the summary alone reproduces the run."""
    suite = {"batteries": ["fabricated"], "datasets": ["baseline_4class"], "seed": 3}
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, suite, label_seed)
    config["label_generation"]["n_labels"] = 2
    if label_seed == "absent":
        del config["label_generation"]["seed"]
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 1

    summary = next(txt_dir.glob("*.txt")).read_text(encoding="utf-8")
    assert "fabricated_data_suite.seed = 3 (set in the config)" in summary, summary
    assert expected in summary, summary
    base = 7 if label_seed == 7 else 42
    assert "Label_0:" in summary and f"    seed = {base}\n" in summary, summary
    assert f"    seed = {base + 1}\n" in summary, summary


def test_a_source_that_ignores_the_suite_seed_never_mentions_it(tmp_path, caplog):
    """byoc reads a file and clustbench downloads fixed data, so their suite seed
    is never read: no seed is drawn for a valueless key, and the summary lists
    only the label seed."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {"f1": rng.normal(size=40), "f2": rng.normal(size=40), "cluster": np.repeat([0, 1], 20)}
    ).to_csv(inputs / "solo.csv", index=False)
    config = byoc_config(inputs, tmp_path, ["solo"])
    config["byoc_suite"]["seed"] = None
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        assert run_pipeline("byoc", config, csv_dir, png_dir, txt_dir, False) == 1
    assert "byoc_suite.seed" not in caplog.text, caplog.text
    summary = next(txt_dir.glob("*.txt")).read_text(encoding="utf-8")
    assert "byoc_suite.seed" not in summary, summary
    assert "label_generation.seed = 42 (set in the config)" in summary, summary


def test_resolve_seed_keeps_the_documented_default():
    """An absent key is still 42; only a valueless one draws."""
    from clmsynth._single_run import _resolve_seed

    assert _resolve_seed({}, "x.seed") == (42, "default; the key is absent")
    assert _resolve_seed({"seed": 7}, "x.seed") == (7, "set in the config")
    drawn, how = _resolve_seed({"seed": None}, "x.seed")
    assert isinstance(drawn, int) and 0 <= drawn < 2**31
    assert how == "drawn at random; the key has no value"


@pytest.mark.parametrize(
    "clm_label,n_zero,datasets,expected_exit",
    [
        ({"num_classes": 2, "matching_mode": "perfect"}, 20, ["solo"], 0),
        (
            {
                "num_classes": 2,
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": [0], "recall_target": 1.0}],
                "split_rule": "equal",
                "spillover_rule": "proportional_to_marginal",
            },
            3, ["solo"], 3,
        ),
        ({"num_classes": 2, "matching_mode": "bogus_mode"}, 20, ["solo"], 2),
        ({"num_classes": 2}, 20, ["solo"], 2),
        ({"num_classes": 3, "matching_mode": "perfect"}, 20, ["solo"], 2),
        ({"num_classes": 2, "matching_mode": "perfect"}, 20, ["no_such_dataset"], 2),
        ({"num_classes": 2, "matching_mode": "perfect"}, 20, ["solo", "no_such_dataset"], 2),
        ({"num_classes": 2, "matching_mode": "perfect"}, 20, "all", 2),
        ({"num_classes": 2, "matching_mode": "perfect"}, 20, ["header_only"], 1),
    ],
    ids=["success-0", "label-left-out-3", "coded-config-error-2", "missing-key-2",
         "every-dataset-refused-2", "csv-missing-2", "csv-missing-beside-a-good-one-2",
         "byoc-datasets-all-2", "unusable-csv-1"],
)
def test_the_exit_codes_stay_distinguishable(tmp_path, monkeypatch, clm_label, n_zero,
                                             datasets, expected_exit):
    """0, 1, 2 and 3 mean different things and must not collapse into each other.

    A caller scripting the CLI, or reading a scheduler's job record, tells a
    config to fix (2) from a failure beyond it (1) and from a run that wrote
    part of what it was asked for (3). Label 0's budget does not fit in a
    3-point cluster 0 ([CLM-150]), so the dataset is written without its label.
    A missing matching_mode ([CLM-202]) has no default, so no dataset can be
    labelled. Perfect matching of 3 labels to 2 clusters ([CLM-102]) refuses the
    only dataset. A name with no CSV is a selection the source does not have, so
    the config is refused, also beside a good one; a CSV that is there and cannot
    be imported is beyond the config. The run folder stays exactly when something
    was written.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("solo", n_zero)])
    (inputs / "header_only.csv").write_text("f1,f2,cluster\n", encoding="utf-8")

    code, survivors = run_main(
        tmp_path, monkeypatch, minimal_byoc_config(inputs, tmp_path / "out", datasets, clm_label)
    )

    assert code == expected_exit, f"expected exit {expected_exit}, got {code}"
    assert bool(survivors) == (expected_exit in (0, 3)), survivors


# ---------------------------------------------------------------------------
# A run that produces nothing must leave nothing
#
# The run folder and its three subfolders are created before `run_pipeline` is
# called, and the resolved config before its first dataset, so a run that writes
# no dataset used to leave the whole skeleton behind. Reported from the wizard's byoc path: a config
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
    dataset to fail while its neighbors succeed. These tests care only about
    whether a run produced output, so they want the plainest config that runs
    and, in one case, one that is rejected outright.
    """
    return {
        "global_settings": {"data_source": "byoc", "output_dir": str(output_dir)},
        "byoc_suite": {
            "batteries": ["local"],
            "input_dir": str(input_dir),
            "datasets": datasets,
            "cluster_column": "cluster",
            "standardize": False,
            "seed": 42,
        },
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": clm_label or {"num_classes": 2, "matching_mode": "perfect"},
        },
    }


def run_main(tmp_path, monkeypatch, config, *flags):
    """Drive the CLI over `config` and return (exit code, surviving run folders)."""
    monkeypatch.setattr(clmsynth._extras.visualization, "plot_feature_scatter", lambda *a, **k: True)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth", str(config_path), *flags])

    out = Path(config["global_settings"]["output_dir"])
    code: str | int | None = 0
    try:
        clmsynth.__main__.main()
    except SystemExit as exit_signal:
        code = exit_signal.code
    return code, sorted(p.name for p in out.iterdir()) if out.is_dir() else []


def test_a_run_that_writes_nothing_leaves_no_run_folder(tmp_path, monkeypatch):
    """The reported case: byoc pointed at a folder holding no such CSV.

    The run names the missing file and refuses the selection, so no dataset is
    processed and it exits 2. What used to survive that was a timestamped
    folder containing an empty csv/, png/ and txt/ and a copy of the config --
    indistinguishable at a glance from a run that succeeded, and picked up by
    anything globbing the output directory. --keep-failed keeps it on request.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()

    code, survivors = run_main(
        tmp_path, monkeypatch, minimal_byoc_config(inputs, tmp_path / "out", ["no_such_dataset"])
    )

    assert code == 2, f"expected exit 2 for a dataset that is not there, got {code}"
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
        tmp_path,
        monkeypatch,
        minimal_byoc_config(
            inputs,
            tmp_path / "out",
            ["solo"],
            clm_label={
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "single",
                "single_match": {"cluster": 9, "label": 0},
                "spillover_rule": "proportional_to_marginal",
            },
        ),
    )

    assert code == 2, f"expected exit 2 for a coded config error, got {code}"
    assert survivors == [], f"an empty run folder survived a coded abort: {survivors}"


def test_keep_failed_keeps_the_config_and_the_log_of_why(tmp_path, monkeypatch):
    """The same coded abort under --keep-failed: only info/ holds anything.

    The byoc precheck runs after the resolved config is written, so the config
    is kept too, not only the log.
    """
    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("solo", 20)])
    config = minimal_byoc_config(
        inputs, tmp_path / "out", ["solo"],
        clm_label={"num_classes": 2, "matching_mode": "single",
                   "single_match": {"cluster": 9, "label": 0}},
    )

    code, survivors = run_main(tmp_path, monkeypatch, config, "--keep-failed")

    assert code == 2, f"expected exit 2 for a coded config error, got {code}"
    assert len(survivors) == 1, f"the failed run's folder was not kept: {survivors}"
    run_dir = tmp_path / "out" / survivors[0]
    kept = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())
    assert kept == [f"info/{run_dir.name}.json", f"info/{run_dir.name}.log"], kept
    log_text = (run_dir / "info" / f"{run_dir.name}.log").read_text(encoding="utf-8")
    assert "[CLM-105]" in log_text, log_text


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
        tmp_path, monkeypatch, minimal_byoc_config(inputs, tmp_path / "out", ["solo"])
    )

    assert code == 0, f"expected a clean run, got exit {code}"
    assert len(survivors) == 1, f"the successful run's folder is missing: {survivors}"
    written = sorted(p.name for p in (tmp_path / "out" / survivors[0] / "csv").iterdir())
    assert written, "the run folder survived but its csv/ is empty"


def test_a_run_folder_is_named_by_run_and_its_files_by_dataset(tmp_path, monkeypatch, caplog):
    """YYYYMMDD_HHMMSS_<Source>/: per-dataset files in csv/ and info/, and the
    run's own config and log in info/, named after the folder. The log names the
    config it ran and opens each dataset with a header."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    write_byoc_batch(inputs, [("solo", 20)])
    caplog.set_level(logging.INFO)

    code, survivors = run_main(
        tmp_path, monkeypatch, minimal_byoc_config(inputs, tmp_path / "out", ["solo"])
    )

    assert code == 0, f"expected a clean run, got exit {code}"
    (name,) = survivors
    assert re.fullmatch(r"\d{8}_\d{6}_BYOC", name), name
    run_dir = tmp_path / "out" / name
    files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())
    assert files == ["csv/byoc-local-solo.csv", f"info/{name}.json", f"info/{name}.log",
                     "info/byoc-local-solo.txt"], files
    log_text = (run_dir / "info" / f"{name}.log").read_text(encoding="utf-8")
    assert f"Run {name} from config '{(tmp_path / 'cfg.json').resolve()}'." in log_text, log_text
    assert "Dataset 1 of 1: local/solo" in log_text, log_text


def test_sigterm_stops_the_run_like_ctrl_c(tmp_path, monkeypatch):
    """A scheduler cancels a job, or ends it at its time limit, with SIGTERM.
    The run logs why it stopped and exits 143, and main() puts back the handler
    it found."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    before = signal.getsignal(signal.SIGTERM)

    def terminated(*args, **kwargs):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"the run installed no SIGTERM handler: {handler!r}"
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(clmsynth._single_run, "run_pipeline", terminated)
    code, survivors = run_main(
        tmp_path, monkeypatch, minimal_byoc_config(inputs, tmp_path / "out", ["solo"]),
        "--keep-failed")

    assert code == 143, f"expected exit 143 for SIGTERM, got {code}"
    log_text = (tmp_path / "out" / survivors[0] / "info" / f"{survivors[0]}.log").read_text(
        encoding="utf-8")
    assert "Terminated; stopping." in log_text, log_text
    assert signal.getsignal(signal.SIGTERM) is before


def _failed_run_scaffold(tmp_path):
    """What a run holds before its first dataset: empty csv/ and png/, and
    info/ with its config and log."""
    run_dir = tmp_path / "20260809_120000_BYOC"
    for sub in ("csv", "png", "info"):
        (run_dir / sub).mkdir(parents=True)
    for ext in (".yaml", ".log"):
        (run_dir / "info" / f"{run_dir.name}{ext}").write_text("x", encoding="utf-8")
    return run_dir


@pytest.mark.parametrize(
    "litter",
    ["csv/result.csv", "png/plot.png", "info/byoc-local-solo.txt", "stray.txt"],
    ids=["a-csv", "a-plot", "a-summary", "an-unexpected-file"],
)
def test_a_run_folder_holding_anything_real_is_never_removed(tmp_path, litter):
    """The predicate directly: one file is enough to make a folder untouchable.

    Parametrized over an expected output and an unexpected file, because the
    rule is not "no output" but "nothing at all beyond the scaffolding". A file
    this code did not put there is someone else's, and guessing about it is how
    a cleanup becomes a data-loss bug.
    """
    run_dir = _failed_run_scaffold(tmp_path)
    target = run_dir / litter
    target.write_text("content", encoding="utf-8")

    assert discard_failed_run(run_dir, keep_failed=False) is False
    assert run_dir.is_dir(), "a run folder holding real content was removed"
    assert target.is_file(), "the content itself was removed"


def test_the_failed_run_predicate_accepts_exactly_the_scaffolding(tmp_path):
    """The positive half, stated once on its own.

    Two empty subfolders and info/ with the run's config and log is precisely
    what a run leaves before its first dataset is written; if this stops
    matching, the cleanup silently stops happening and the litter comes back
    with no test going red.
    """
    run_dir = _failed_run_scaffold(tmp_path)

    assert discard_failed_run(run_dir, keep_failed=False) is True
    assert not run_dir.exists()


@pytest.mark.parametrize("below", ["", "sub/deeper"], ids=["is-the-file", "under-the-file"])
def test_an_output_dir_blocked_by_a_file_exits_one_instead_of_hanging(
    tmp_path, monkeypatch, caplog, below
):
    """A FILE where the output folder, or a folder above it, should be.

    On Windows `mkdir` under a file raises FileExistsError, which `build_run_dir`
    read as "this run name is taken" and retried under the next suffix forever.
    (On Linux it was a NotADirectoryError traceback.) The run now stops before
    anything is created, names the file, and exits 1 without a `[CLM-###]` code.

    Bounded: `Path.mkdir` fails the test after 50 calls, so a regression is a
    red test, not a hung suite. The retry itself stays guarded by
    `test_build_run_dir_never_hands_out_the_same_name_twice` in `03_isolation`.
    """
    blocker = tmp_path / "OUTPUT"
    blocker.write_text("not a folder", encoding="utf-8")

    real_mkdir = Path.mkdir
    calls = []

    def capped_mkdir(self, *args, **kwargs):
        calls.append(self)
        if len(calls) > 50:
            raise AssertionError(f"build_run_dir is looping: {len(calls)} mkdir calls, last {self}")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", capped_mkdir)
    with caplog.at_level(logging.CRITICAL, logger="clmsynth"):
        code, _ = run_main(
            tmp_path, monkeypatch, minimal_byoc_config(tmp_path, blocker / below, ["solo"])
        )

    assert code == 1, f"expected exit 1, got {code}"
    assert blocker.read_text(encoding="utf-8") == "not a folder", "the blocking file was changed"
    assert f"'{blocker.resolve()}' is a file, not a folder" in caplog.text, caplog.text
    assert "[CLM-" not in caplog.text, "a file layout problem was reported as a CLM error"


def test_reporting_a_path_never_raises():
    """A line describing where a run will write must not be why it fails.

    `main` and `run_pipeline` state their resolved output and input directories
    before doing any work. `resolve()` can fail on a malformed path -- an
    embedded null, an over-long Windows path, a dead network mount -- and the
    fallback is the raw value.

    Here rather than in `05_config_safety`, which owns *what* the report is for:
    this asserts the pipeline degrades rather than aborts, which is this
    module's subject. A diagnostic that can kill the run it describes is the
    failure mode, not the feature.
    """
    for malformed in ("\0embedded-null", "x" * 5000, ""):
        assert isinstance(resolved_for_report(malformed), str)


@pytest.mark.usefixtures("yaml")
def test_generate_config_exits_two_on_a_missing_payload(tmp_path, monkeypatch):
    """The other console script's failure path, alongside the pipeline's.

    `clmsynth-config` reads a payload file. A missing one must exit 2 with a
    message pointing at the fix, not raise `FileNotFoundError` out of the CLI,
    it is the first command in the documented quick start, so it is the most
    likely place for a user to mistype a path.
    """
    from clmsynth._config_gen import generate_config

    monkeypatch.setattr(sys, "argv", ["clmsynth-config", str(tmp_path / "no_such_payload.yaml")])
    with pytest.raises(SystemExit) as excinfo:
        generate_config.main()
    assert excinfo.value.code == 2


@pytest.mark.usefixtures("yaml")
@pytest.mark.parametrize("text", ["clm_label: [unclosed", "target_metric: [mcc, 0.6]",
                                  "matching_mode: bogus"],
                         ids=["yaml-syntax-error", "block-of-the-wrong-shape",
                              "refused-by-the-engine"])
def test_generate_config_exits_two_on_a_malformed_payload(text, tmp_path, monkeypatch):
    """A YAML syntax error in the payload, a block of the wrong shape, or a
    clm_label block the engine refuses exits 2 like a missing file, not a
    traceback, and writes nothing."""
    from clmsynth._config_gen import generate_config

    payload = tmp_path / "payload.yaml"
    payload.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth-config", str(payload), str(tmp_path / "out.yaml")])
    with pytest.raises(SystemExit) as excinfo:
        generate_config.main()
    assert excinfo.value.code == 2
    assert not (tmp_path / "out.yaml").exists()


@pytest.mark.parametrize("failure,exit_code", [
    (OSError(errno.ENOSPC, "No space left on device"), 1),
    (KeyboardInterrupt(), 130),
], ids=["disk-full", "interrupt"])
def test_generate_config_keeps_the_old_config_when_the_write_fails(
        failure, exit_code, tmp_path, monkeypatch, yaml):
    """A write that does not finish must leave the previous config whole.

    Written in place, the file was truncated before the new text arrived, so a
    full disk or a Ctrl-C left a partial config that could still parse, and the
    next run used it. The interrupt message says "nothing written"; this keeps
    it true.
    """
    from clmsynth._config_gen import generate_config

    payload = tmp_path / "payload.yaml"
    payload.write_text(yaml.safe_dump({"num_classes": 3}), encoding="utf-8")
    out = tmp_path / "out.yaml"
    out.write_text("previous: config\n", encoding="utf-8")

    def fail(fd):
        raise failure

    monkeypatch.setattr(generate_config.os, "fsync", fail)
    monkeypatch.setattr(sys, "argv", ["clmsynth-config", str(payload), str(out)])
    with pytest.raises(SystemExit) as excinfo:
        generate_config.main()
    assert excinfo.value.code == exit_code
    assert out.read_text(encoding="utf-8") == "previous: config\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.yaml", "payload.yaml"], \
        "the temporary file was left behind"


def test_generate_config_refuses_to_overwrite_its_payload(tmp_path, monkeypatch, yaml):
    """`clmsynth-config p.yaml p.yaml` must not replace the payload it just read."""
    from clmsynth._config_gen import generate_config

    payload = tmp_path / "payload.yaml"
    text = yaml.safe_dump({"num_classes": 3})
    payload.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth-config", str(payload), str(payload)])
    with pytest.raises(SystemExit) as excinfo:
        generate_config.main()
    assert excinfo.value.code == 2
    assert payload.read_text(encoding="utf-8") == text


@pytest.mark.usefixtures("yaml")
@pytest.mark.parametrize("kind", ["a-folder", "utf-16"])
def test_generate_config_exits_two_on_a_payload_it_cannot_read(kind, tmp_path, monkeypatch):
    """A folder was a PermissionError traceback on Windows, and text in another
    encoding a UnicodeDecodeError: neither is one of the two errors the read was
    guarded for."""
    from clmsynth._config_gen import generate_config

    payload = tmp_path / "payload.yaml"
    if kind == "a-folder":
        payload.mkdir()
    else:
        payload.write_bytes("num_classes: 3\n".encode("utf-16"))
    monkeypatch.setattr(sys, "argv", ["clmsynth-config", str(payload), str(tmp_path / "out.yaml")])
    with pytest.raises(SystemExit) as excinfo:
        generate_config.main()
    assert excinfo.value.code == 2
    assert not (tmp_path / "out.yaml").exists()


# ---------------------------------------------------------------------------
# Settings a run cannot use, and what a selection names twice
# ---------------------------------------------------------------------------


def run_status(tmp_path, config):
    """run_single on `config`, written to a file: its exit status."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return run_single(str(path), render_plots=False)


def one_fabricated(tmp_path, **suite):
    """A one-dataset fabricated config that writes under tmp_path/OUT."""
    return {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path / "OUT")},
        "fabricated_data_suite": {**_ONE_DATASET, **suite},
        "label_generation": {"n_labels": 1, "seed": 42,
                             "clm_label": {"num_classes": 4, "matching_mode": "perfect"}},
    }


@pytest.mark.parametrize(("value", "statuses"), [(5, {2}), (None, {0, 2})],
                         ids=["a-number", "valueless"])
def test_an_output_dir_that_is_not_text_is_no_traceback(value, statuses, tmp_path, monkeypatch,
                                                        caplog):
    """`Path(5)` was an uncaught TypeError before anything was logged: exit 2,
    naming the key. A valueless `output_dir:` runs or is refused, never a traceback."""
    monkeypatch.chdir(tmp_path)
    config = one_fabricated(tmp_path)
    config["global_settings"]["output_dir"] = value
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        status = run_status(tmp_path, config)
    assert status in statuses
    assert status != 2 or "output_dir" in caplog.text


def test_an_output_dir_with_a_null_character_is_reported(tmp_path, caplog):
    """mkdir raises ValueError for it, which the OSError handler around the run
    folder missed: exit 1, as for any folder that cannot be created."""
    config = one_fabricated(tmp_path)
    config["global_settings"]["output_dir"] = str(tmp_path / "bad\0name")
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_status(tmp_path, config) == 1
    assert "Cannot create the output folder" in caplog.text


@pytest.mark.parametrize("name", ["no?such", "no/such"], ids=["a-question-mark", "a-slash"])
def test_an_unknown_data_source_is_refused_whatever_its_characters(name, tmp_path, caplog):
    """The run folder is named after the source before the source is checked. `?`
    made Windows refuse the folder, and the run exited 1 about the output folder;
    `/` made two levels of run folder, of which the failed run removed the inner."""
    config = one_fabricated(tmp_path)
    config["global_settings"]["data_source"] = name
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_status(tmp_path, config) == 2
    assert "Valid options" in caplog.text
    out = tmp_path / "OUT"
    assert not out.exists() or not list(out.iterdir())


def test_n_labels_0_does_not_read_clm_label(tmp_path):
    """clm_label is not read when n_labels is 0, and the saved config leaves it out,
    but byoc's precheck of the cluster ids read it and stopped the run."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    pd.DataFrame({"f1": np.arange(20) / 10, "f2": np.arange(20) / 5,
                  "cluster": np.repeat([8, 9], 10)}).to_csv(inputs / "data.csv", index=False)
    clm = {"num_classes": 2, "matching_mode": "single", "single_match": {"cluster": 0, "label": 0}}
    config = minimal_byoc_config(inputs, tmp_path / "OUT", ["data"], clm)
    config["label_generation"]["n_labels"] = 0
    assert run_status(tmp_path, config) == 0
    written = next((tmp_path / "OUT").rglob("csv/*.csv"))
    assert "Label_0" not in written.read_text(encoding="utf-8").splitlines()[0]


def test_an_empty_n_samples_takes_the_preset_s_size(tmp_path):
    """`n_samples:` with no value is the preset's size, as absent: `int(None)`
    skipped every dataset with a raw TypeError."""
    config, csv_dir, png_dir, txt_dir = _fabricated_config(
        tmp_path, {**_ONE_DATASET, "n_samples": None})
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 1
    assert len(pd.read_csv(next(csv_dir.glob("*.csv")))) == 800


@pytest.mark.parametrize("n_samples", [0, "100", 1], ids=["zero", "text", "too-few"])
def test_an_n_samples_no_preset_can_use_is_named(n_samples, tmp_path, caplog):
    """0 skipped the dataset with "division by zero" and 1 with pandas' "Bin edges
    must be unique": neither named the setting."""
    config, csv_dir, png_dir, txt_dir = _fabricated_config(
        tmp_path, {**_ONE_DATASET, "n_samples": n_samples})
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False) == 0
    assert "fabricated_data_suite.n_samples" in caplog.text, caplog.text


def test_a_dataset_named_twice_runs_once_with_a_warning(tmp_path, caplog):
    """As a batch does with a config named twice. It ran twice and counted 2, while
    the saved config holds it once."""
    config, csv_dir, png_dir, txt_dir = _fabricated_config(
        tmp_path, {**_ONE_DATASET, "datasets": ["baseline_4class", "baseline_4class"]})
    report: dict[str, int] = {}
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False,
                            report=report) == 1
    assert report["resolved"] == 1
    assert "'fabricated/baseline_4class' is named more than once" in caplog.text


def test_fetch_refuses_a_key_the_source_does_not_take(caplog):
    """run_pipeline refuses a misspelt suite key; `fetch`, which takes the same keys,
    ran it on the default without a word. The fetcher contract holds: None, and
    the key in the log."""
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        assert dataset_sources.fetch("fabricated_data", "fabricated", "baseline_2class",
                                     n_sample=10) is None
    assert "'n_sample'" in caplog.text


def test_run_pipeline_makes_its_csv_folder(tmp_path):
    """The txt and png folders are made when first written to; a csv folder that is
    not there skipped every dataset with an OSError."""
    config, *_ = _fabricated_config(tmp_path, dict(_ONE_DATASET))
    out = tmp_path / "new"
    assert run_pipeline("fabricated_data", config, out / "csv", out / "png", out / "txt",
                        False) == 1


def test_the_summary_says_a_dataset_is_short_of_its_labels(tmp_path, monkeypatch):
    """Three of five labels fit: the summary listed three and never said two were
    missing, so it read as a run of three. Only the log had it."""
    def three_of_them(context, generator):
        generate_additional_labels(context, n_labels=3, clm_config=generator.clm_config, seed=1)

    monkeypatch.setattr(clmsynth._single_run, "_label_dataset", three_of_them)
    config, csv_dir, png_dir, txt_dir = _fabricated_config(tmp_path, dict(_ONE_DATASET))
    config["label_generation"]["n_labels"] = 5
    report: dict[str, int] = {}
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir, False,
                        report=report) == 1
    assert report["short"] == 1
    assert "3 of 5 labels written" in next(txt_dir.glob("*.txt")).read_text(encoding="utf-8")


def test_a_batch_passes_a_config_whose_name_starts_with_a_dash(tmp_path, monkeypatch):
    """`clmsynth -- -a.json b.json` reached each child as a bare argument, which
    its argparse read as an option: exit 2 for a config that runs."""
    from clmsynth._extras import batch_run

    (tmp_path / "-a.json").write_text(json.dumps(one_fabricated(tmp_path)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    root = str(Path(clmsynth.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": root}
    code, output = batch_run._run_config("-a.json", ["--no-viz"], env)
    assert code == 0, output[-300:]


# ---------------------------------------------------------------------------
# --help for the two guides
# ---------------------------------------------------------------------------

GUIDES = {"clmsynth-config": "clmsynth._config_gen.generate_config",
          "clmsynth-wizard": "clmsynth._config_gen.config_wizard"}


def run_guide(command, flag, folder):
    """`<command> <flag>` in a child working in `folder`, with nothing on stdin.

    -I and the path insert are as in `03_isolation`: the child imports the clmsynth
    this session did, whatever PYTHONPATH holds.
    """
    root = str(Path(clmsynth.__file__).resolve().parents[1])
    code = (f"import sys\nsys.path.insert(0, {root!r})\nsys.argv = [{command!r}, {flag!r}]\n"
            f"from {GUIDES[command]} import main\nmain()\n")
    return subprocess.run([sys.executable, "-I", "-c", code], cwd=folder,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=60, check=False)


@pytest.mark.parametrize("flag", ["--help", "-h"])
@pytest.mark.parametrize("command", GUIDES)
def test_help_prints_usage_and_exits_0(command, flag, tmp_path):
    done = run_guide(command, flag, tmp_path)
    assert done.returncode == 0, done.stderr
    assert "usage" in done.stdout.lower()
    assert command in done.stdout


@pytest.mark.parametrize("command", GUIDES)
def test_help_writes_nothing_and_asks_nothing(command, tmp_path):
    done = run_guide(command, "--help", tmp_path)
    assert done.returncode == 0, done.stderr
    assert not list(tmp_path.iterdir())


def test_config_help_names_its_arguments(tmp_path):
    text = run_guide("clmsynth-config", "--help", tmp_path).stdout.lower()
    assert "payload" in text and "output" in text


def test_a_mistyped_option_is_a_usage_error_not_a_payload(tmp_path):
    done = run_guide("clmsynth-config", "--helpp", tmp_path)
    assert done.returncode == 2
    assert "not found" not in done.stderr and "usage" in done.stderr.lower()


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
    """
    features = pd.DataFrame({"f1": range(10), "f2": range(10)})
    context = DatasetContext(
        "src", "battery", features, ground_truths={"labels0": pd.Series(range(10))}
    )

    for bad_length in (9, 11):
        with pytest.raises(ValueError, match="misaligned"):
            context.add_generated_label("Label_0", pd.Series(range(bad_length)))


@pytest.mark.parametrize("clm_config", [None, {}], ids=["none", "empty"])
def test_a_missing_clm_label_is_refused_not_silently_filled(clm_config):
    """A config without clm_label is refused, as [CLM-207]: a missing key, so the
    run stops. It was an uncoded KeyError."""
    features = pd.DataFrame({"f1": range(10), "f2": range(10)})
    context = DatasetContext(
        "src", "battery", features, ground_truths={"labels0": pd.Series([0] * 5 + [1] * 5)}
    )

    with pytest.raises(KeyError) as excinfo:
        generate_additional_labels(context, n_labels=1, clm_config=clm_config)

    assert getattr(excinfo.value, "code", None) == 207
    assert "'clm_label' is required" in str(excinfo.value)
    assert not context.generated_labels, "a label was attached despite the refusal"


def test_context_accepts_an_aligned_label_and_ignores_its_index():
    """The other half: correct length is accepted, and a foreign index is reset.

    A series carrying its own index, which is what any filtered or reordered
    pandas operation produces, must be re-based rather than rejected or, worse,
    silently aligned by index and scrambled.
    """
    features = pd.DataFrame({"f1": range(10), "f2": range(10)})
    context = DatasetContext(
        "src", "battery", features, ground_truths={"labels0": pd.Series(range(10))}
    )

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

    with (
        mock.patch("clmsynth._input_sources.clustbench_source.urllib.request.urlopen",
                   fake_urlopen),
        pytest.raises(OSError),
    ):
        clustbench_source._loadtxt_url("http://example.invalid/data.txt")

    assert captured["timeout"] == clustbench_source.CLUSTBENCH_TIMEOUT
    assert captured["timeout"] is not None, "fetch would inherit the OS default timeout"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "data:text/plain,x", "/tmp/local.gz"],
    ids=["file", "data", "no-scheme"],
)
def test_clustbench_refuses_a_non_http_base_url(url):
    """`base_url` comes from a shareable config, so only http(s) is opened."""
    with pytest.raises(ValueError) as excinfo:
        clustbench_source._loadtxt_url(url)
    assert "only http(s) is fetched" in str(excinfo.value)


def test_clustbench_fetch_returns_none_when_the_source_is_unreachable():
    """A failed fetch is a skipped dataset, not an exception out of the batch.

    Patched rather than pointed at a real unreachable host, so it is instant
    and does not depend on the runner having (or not having) a network.
    """
    with mock.patch(
        "clmsynth._input_sources.clustbench_source.urllib.request.urlopen",
        side_effect=OSError("unreachable"),
    ):
        assert (
            clustbench_source.fetch_clustbench_data(
                dataset_group="wut",
                dataset_name="smile",
                base_url="http://example.invalid/benchmark",
            )
            is None
        )


# ---------------------------------------------------------------------------
# mdcgen presets
#
# mdcgenpy is not on PyPI and is stubbed: the stub records what a preset asked
# for, which is what these check.
# ---------------------------------------------------------------------------

MDCGEN_PRESETS = [(group, name) for group, presets in MDCGEN_CONFIGS.items() for name in presets]
IGNORED_BY_MDCGENPY = {"m", "n", "cp", "out"}
MDCGEN_SIZES = [("basic", "blobs_2d_5c", 1000, 2), ("basic", "blobs_3d_10c", 2000, 3),
                ("high_dim", "hd_50d_5c", 1000, 50), ("high_dim", "hd_100d_10c", 2000, 100),
                ("challenging", "overlapping_2d", 1000, 2), ("challenging", "noisy_3d", 1500, 3)]


@pytest.fixture
def mdcgenpy(monkeypatch):
    """A stand-in for mdcgenpy.clusters; yields the keyword arguments of each call."""
    calls = []

    class ClusterGenerator:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.kwargs = kwargs

        def generate_data(self):
            k = self.kwargs.get("k", 3)
            n = int(self.kwargs.get("n_samples", 100))
            d = int(self.kwargs.get("n_feats", 2))
            parts = len(k) if isinstance(k, (list, tuple)) else int(k)
            return np.random.default_rng(0).normal(size=(n, d)), np.arange(n) % parts

    package, clusters = types.ModuleType("mdcgenpy"), types.ModuleType("mdcgenpy.clusters")
    vars(clusters)["ClusterGenerator"] = ClusterGenerator
    vars(package)["clusters"] = clusters
    monkeypatch.setitem(sys.modules, "mdcgenpy", package)
    monkeypatch.setitem(sys.modules, "mdcgenpy.clusters", clusters)
    return calls


@pytest.mark.parametrize(("group", "name"), MDCGEN_PRESETS)
def test_a_preset_passes_only_names_mdcgenpy_reads(mdcgenpy, group, name):
    dataset_sources.fetch("mdcgen", group, name, seed=3)
    assert not IGNORED_BY_MDCGENPY & set(mdcgenpy[0])


@pytest.mark.parametrize(("group", "name", "n", "d"), MDCGEN_SIZES)
def test_a_preset_gets_the_size_and_dimension_it_names(mdcgenpy, group, name, n, d):
    frame = dataset_sources.fetch("mdcgen", group, name, seed=3)
    assert (mdcgenpy[0].get("n_samples"), mdcgenpy[0].get("n_feats")) == (n, d)
    assert frame is not None
    assert frame.filter(like="Feature_").shape == (n, d)


def test_the_overlapping_preset_sets_compactness(mdcgenpy):
    dataset_sources.fetch("mdcgen", "challenging", "overlapping_2d", seed=3)
    assert mdcgenpy[0].get("compactness_factor") == 0.1


def test_the_noisy_preset_sets_a_count_of_outliers(mdcgenpy):
    """mdcgenpy's outliers is a count; the preset's `out: 0.05` is not one."""
    dataset_sources.fetch("mdcgen", "challenging", "noisy_3d", seed=3)
    outliers = mdcgenpy[0].get("outliers", 0)
    assert isinstance(outliers, int) and outliers > 0


@pytest.mark.parametrize("key", ["n_noise", "min_samples", "ki_coeff", "corr", "rotate",
                                 "distributions"])
def test_a_preset_reaches_the_setting(key):
    """Noise dimensions, unequal sizes, distributions, correlation and rotation."""
    assert any(key in preset for presets in MDCGEN_CONFIGS.values()
               for preset in presets.values())


def test_a_preset_has_unequal_cluster_sizes():
    """Through mdcgenpy's k, min_samples or ki_coeff."""
    assert any(isinstance(preset.get("k"), (list, tuple)) or {"min_samples", "ki_coeff"} & set(preset)
               for presets in MDCGEN_CONFIGS.values() for preset in presets.values())


def test_an_old_suite_name_is_translated_or_refused(mdcgenpy):
    """A suite key in its old name, such as `cp`, is translated to the name mdcgenpy
    reads or the fetch is refused: mdcgenpy would ignore it without a word."""
    frame = dataset_sources.fetch("mdcgen", "basic", "blobs_2d_5c", seed=3, cp=0.2)
    assert frame is None or (mdcgenpy and mdcgenpy[0].get("compactness_factor") == 0.2
                             and "cp" not in mdcgenpy[0])


def test_mdcgen_records_which_features_are_noise(tmp_path, mdcgenpy):
    """Only an mdcgen dataset has this record."""
    noisy = [(group, name) for group, presets in MDCGEN_CONFIGS.items()
             for name, preset in presets.items() if "n_noise" in preset]
    assert noisy, "no preset sets n_noise"
    group, name = noisy[0]
    clm = {"num_classes": 2, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [k], "label": k, "recall_target": 0.8}
                                 for k in range(2)]}
    config: dict[str, Any] = {
        "global_settings": {"data_source": "mdcgen"},
        "mdcgen_suite": {"batteries": [group], "datasets": [name], "seed": 3},
        "label_generation": {"n_labels": 1, "seed": 1, "clm_label": clm}}
    csv_dir, png_dir, txt_dir = (tmp_path / sub for sub in ("csv", "png", "txt"))
    for folder in (csv_dir, png_dir, txt_dir):
        folder.mkdir()
    run_pipeline("mdcgen", config, csv_dir, png_dir, txt_dir, render_plots=False)
    summary = next(txt_dir.glob("*.txt")).read_text(encoding="utf-8")
    assert re.search(r"(?is)cluster structure.*noise", summary)
