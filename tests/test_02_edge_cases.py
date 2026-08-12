"""Category 2: Edge cases and boundary conditions.

Where `01_logic` probes named suspicions in the engine's logic, this probes the
edges of its inputs and duplicates.

Runs against the engine and the BYOC loader in-process. No network, no optional
dependencies; the only filesystem use is CSV fixtures written to pytest's
`tmp_path`.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from clmsynth.byoc_source import fetch_byoc_data
from clmsynth.clm_label_engine import generate_clm_labels, resolve_label_counts
from clmsynth.main import build_run_dir, run_pipeline

N = 1000
CLUSTERS = np.concatenate([np.full(400, 0), np.full(300, 1), np.full(200, 2), np.full(100, 3)])
COORDS = np.random.default_rng(0).normal(size=(N, 2))


def write_byoc_csv(directory, name, frame):
    """Write a BYOC fixture and load it back through the real source adapter."""
    frame.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8")
    return fetch_byoc_data(dataset_name=name, input_dir=str(directory), cluster_column="cluster")


# ---------------------------------------------------------------------------
# Null, empty, zero
# ---------------------------------------------------------------------------

def test_empty_dataset_under_random_mode_returns_an_empty_series():
    """N=0 is a legitimate input: nothing to label, so nothing comes back."""
    out = generate_clm_labels(np.array([], dtype=int), np.empty((0, 2)), {
        "num_classes": 2, "balance": "balanced", "matching_mode": "random",
    }, seed=1)
    assert len(out) == 0


def test_empty_dataset_under_custom_mode_rejects_the_referenced_cluster():
    """With no points there are no cluster ids, so any rule names an unknown one."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(np.array([], dtype=int), np.empty((0, 2)), {
            "num_classes": 2, "balance": "balanced", "matching_mode": "custom",
            "assignment_matrix": [{"label": 0, "clusters": [0], "recall_target": 0.5}],
        }, seed=1)
    assert "[CLM-105]" in str(excinfo.value)


def test_empty_proportions_list_falls_through_to_the_skew_rule():
    """`proportions: []` is falsy, so it means "not given", not "all zero".

    Treating an empty list as an explicit setting would hand the allocator a
    zero-length distribution instead of consulting `skew_rule`.
    """
    counts = resolve_label_counts({
        "num_classes": 3, "balance": "unbalanced", "proportions": [],
        "skew_rule": "geometric", "skew_params": {"ratio": 0.5},
    }, N, np.random.default_rng(0))
    assert len(counts) == 3
    assert sum(counts) == N


def test_proportions_length_mismatch_is_rejected():
    """[CLM-121]: one entry per label, no more and no fewer.

    Before 0.6.0 a short or long list silently resized the label space, six
    proportions under `num_classes: 4` wrote labels 4 and 5 into the dataset,
    and only `proportional_to_marginal` noticed, with an uncoded numpy
    broadcasting error. This is the guard for that.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 4, "balance": "unbalanced", "proportions": [0.5, 0.5],
            "matching_mode": "random",
        }, seed=1)
    assert "[CLM-121]" in str(excinfo.value)


def test_empty_assignment_matrix_leaves_spillover_to_cover_everything():
    """Zero rules is not an error: nothing is claimed, so spillover places all N."""
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 3, "balance": "balanced", "matching_mode": "custom",
        "assignment_matrix": [], "spillover_rule": "proportional_to_marginal",
    }, seed=1).to_numpy()
    assert len(out) == N
    assert out.min() >= 0


def test_labels_only_dataset_has_no_features_and_still_labels(tmp_path):
    """Zero feature columns is a legitimate dataset, not a broken one.

    The engine has always accepted a labels-only run, cluster ids with no
    feature space, because recall, proportions, allocation and spillover are
    pure counting. Until the `labels_only` fabricator preset existed, no config
    could produce one, so the capability was reachable only from Python.

    The complementary half, that requesting spatial placement over this dataset
    raises [CLM-125], is the catalog's `CLM-125` fixture.
    """
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()

    n_ok = run_pipeline("fabricated_data", {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": {"batteries": ["fabricated"],
                                  "datasets": ["labels_only_4class"], "seed": 42},
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": {"num_classes": 4, "balance": "balanced", "matching_mode": "custom",
                          "assignment_matrix": [{"label": i, "clusters": [i],
                                                 "recall_target": 0.5} for i in range(4)],
                          "split_rule": "equal",
                          "spillover_rule": "proportional_to_marginal"}},
    }, csv_dir, png_dir, txt_dir)

    assert n_ok == 1
    written = pd.read_csv(csv_dir / "fabricated_data__fabricated__labels_only_4class.csv")
    assert list(written.columns) == ["Cluster_0", "Label_0"], \
        "a labels-only dataset should carry clusters and labels and nothing else"
    assert len(written) == 800
    assert not list(png_dir.glob("*.png")), "fewer than 2 features: plots must be skipped"


def test_single_point_dataset():
    """N=1, M=K=1: the smallest well-formed input the engine can be given."""
    out = generate_clm_labels(np.array([0]), np.array([[0.0, 0.0]]),
                              {"num_classes": 1, "matching_mode": "perfect"}, seed=1)
    assert list(out.to_numpy()) == [0]


# ---------------------------------------------------------------------------
# Extreme ranges
# ---------------------------------------------------------------------------

def test_large_dataset_preserves_the_invariant():
    """N=80,000 with centroid weighting.

    Stresses weighted sampling without replacement and largest-remainder
    rounding at a scale where an O(N^2) mistake or a rounding drift would show.
    Runs in well under a second, so it is not a slow test despite the size.
    """
    big_n = 80_000
    sizes = [big_n // 5] * 4 + [big_n - 4 * (big_n // 5)]
    labels = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    coords = np.random.default_rng(7).normal(size=(big_n, 2))

    out = generate_clm_labels(labels, coords, {
        "num_classes": 5, "balance": "balanced", "matching_mode": "custom",
        "assignment_matrix": [{"label": i, "clusters": [i], "recall_target": 0.7} for i in range(5)],
        "split_rule": "proportional_to_size", "spillover_rule": "proportional_to_marginal",
        "centroid_dependence": {"enabled": True, "profile": "exponential",
                                "favors": "core", "steepness": 4.0},
    }, seed=1).to_numpy()

    assert len(out) == big_n
    assert out.min() >= 0 and out.max() < 5


def test_float_rounding_in_proportions_is_tolerated():
    """[1/3, 1/3, 1/3] sums to 0.9999999999999999 in binary float.

    A naive `sum(p) != 1.0` check rejects the most natural way to write an even
    three-way split, so the sum test has to carry a tolerance.
    """
    counts = resolve_label_counts({
        "num_classes": 3, "balance": "unbalanced", "proportions": [1 / 3, 1 / 3, 1 / 3],
    }, N, np.random.default_rng(0))
    assert sum(counts) == N


def test_proportions_that_are_clearly_wrong_are_still_rejected():
    """The other side of the tolerance above: 1.5 is not a rounding artifact."""
    with pytest.raises(ValueError) as excinfo:
        resolve_label_counts({
            "num_classes": 3, "balance": "unbalanced", "proportions": [0.5, 0.5, 0.5],
        }, N, np.random.default_rng(0))
    assert "[CLM-106]" in str(excinfo.value)


@pytest.mark.parametrize("skew_rule,params,note", [
    ("geometric", {"ratio": 0.0}, "0**0 == 1 by convention, so all mass lands on label 0"),
    ("dominant_minority", {"dominant_index": 0, "dominant_share": 1.0}, "every other label gets exactly 0"),
    ("dirichlet", {"alpha": 1e-4}, "near-degenerate draw, almost all mass on one label"),
], ids=["geometric-ratio-0", "dominant-share-1", "dirichlet-tiny-alpha"])
def test_degenerate_skew_parameters_still_partition_n(skew_rule, params, note):
    """Extreme skew settings must still produce counts summing to exactly N.

    Each of these drives the distribution to a corner where one label takes
    everything; the largest-remainder split still has to account for every row.
    """
    counts = resolve_label_counts({
        "num_classes": 4 if skew_rule != "dirichlet" else 5,
        "balance": "unbalanced", "skew_rule": skew_rule, "skew_params": params,
    }, N, np.random.default_rng(0))
    assert sum(counts) == N, note


def _skew_cfg(skew_rule, params, num_classes=4):
    """A config whose ONLY interesting property is its skew parameters.

    `random` mode reaches `resolve_label_counts` and returns immediately, so an
    allocation that happens to be infeasible cannot be mistaken for the rejection
    under test.
    """
    return {"num_classes": num_classes, "balance": "unbalanced",
            "skew_rule": skew_rule, "skew_params": params, "matching_mode": "random"}


@pytest.mark.parametrize("num_classes,skew_rule,params,was", [
    (4, "geometric", {"ratio": -0.5}, "returned [1600, -800, 400, -200]"),
    (4, "dominant_minority", {"dominant_index": 0, "dominant_share": 1.5},
     "returned [1500, -166, -167, -167]"),
    (4, "dominant_minority", {"dominant_index": 0, "dominant_share": -1},
     "returned [-1000, 667, 667, 666]"),
    (4, "dominant_minority", {"dominant_index": 99, "dominant_share": 0.5}, "IndexError"),
    (1, "dominant_minority", {"dominant_index": 0, "dominant_share": 0.9}, "ZeroDivisionError"),
    (4, "dirichlet", {"alpha": 0.0}, "ZeroDivisionError"),
    (4, "dirichlet", {"alpha": -1}, "numpy's bare ValueError: alpha < 0"),
    (4, "geometric", [0.5], "AttributeError, .get() on a list"),
], ids=["ratio-negative", "share-above-1", "share-below-0", "index-past-end",
        "one-class", "alpha-zero", "alpha-negative", "params-not-a-mapping"])
def test_out_of_range_skew_params_are_coded(num_classes, skew_rule, params, was):
    """[CLM-131]: every way `skew_params` could go wrong now says so.

    The first three are the ones that matter. They did not crash, they
    **returned**. The counts still summed to N, so largest-remainder rounding was
    satisfied and a negative label count flowed into allocation with nothing
    downstream objecting. The rest crashed, but uncoded.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, _skew_cfg(skew_rule, params, num_classes), seed=1)
    assert getattr(excinfo.value, "code", None) == 131, \
        f"expected [CLM-131] where the engine previously gave: {was}"


@pytest.mark.parametrize("skew_rule,params", [
    ("geometric", {"ratio": 0.0}),
    ("dominant_minority", {"dominant_index": 0, "dominant_share": 1.0}),
    ("dominant_minority", {"dominant_index": 3, "dominant_share": 0.0}),
    ("dirichlet", {"alpha": 1e-300}),
], ids=["ratio-0", "share-1", "share-0", "alpha-denormal"])
def test_legal_extremes_are_not_swept_up_by_the_guard(skew_rule, params):
    """The boundaries themselves stay valid.

    `[CLM-131]` rejects `ratio < 0`, not `<= 0`, and `alpha <= 0`, not `< 1`.
    A guard that over-reaches here would break the degenerate-but-legal
    configurations asserted directly above.
    """
    counts = resolve_label_counts(_skew_cfg(skew_rule, params), N, np.random.default_rng(0))
    assert sum(counts) == N
    assert min(counts) >= 0, f"negative count survived: {list(counts)}"


def test_bare_null_skew_params_takes_the_documented_defaults():
    """`skew_params:` with nothing after it parses to None, not to {}.

    `.get("skew_params", {})` returned that None and handed it on as the
    parameter mapping, so the next `.get()` raised AttributeError.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 4, "balance": "unbalanced", "skew_rule": "geometric",
        "skew_params": None, "matching_mode": "random"}, seed=1)
    assert len(out) == N


def test_bare_null_centroid_dependence_is_treated_as_disabled():
    """Same YAML shape, different key: `centroid_dependence:` alone.

    The engine already handled this deliberately in `_ensure_coords` and for
    `target_metric`; the allocation pipeline was the one place still passing the
    raw None on to a `.get()` call.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 4, "balance": "unbalanced", "proportions": [0.4, 0.3, 0.2, 0.1],
        "matching_mode": "custom",
        "assignment_matrix": [{"label": i, "clusters": [i], "recall_target": 0.2}
                              for i in range(4)],
        "split_rule": "equal", "spillover_rule": "proportional_to_marginal",
        "centroid_dependence": None}, seed=1)
    assert len(out) == N


def test_unreachable_tolerance_reports_non_convergence(caplog):
    """`tolerance: 0.0` cannot be met, so the solver must give up and say so.

    The failure mode being guarded is a solver that either loops forever or
    quietly reports success. It does neither: it exhausts `max_iter` and emits
    [CLM-306] with its best effort. Previously asserted as `elapsed < 10`,
    which passed with a hundredfold margin and told you nothing.
    """
    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 4, "balance": "unbalanced", "proportions": [0.4, 0.3, 0.2, 0.1],
            "matching_mode": "custom",
            "assignment_matrix": [{"label": i, "clusters": [i], "recall_target": 0.5} for i in range(4)],
            "split_rule": "proportional_to_size", "spillover_rule": "proportional_to_marginal",
            "target_metric": {"type": "mcc", "value": 0.55555, "tolerance": 0.0, "max_iter": 15},
        }, seed=1)
    assert "[CLM-306]" in caplog.text


@pytest.mark.parametrize("max_iter", [0, -5], ids=["zero", "negative"])
def test_degenerate_iteration_bounds_do_not_crash(max_iter):
    """A non-positive `max_iter` means the bisection body never runs.

    The loop bound has to degrade to "no refinement" rather than to a negative
    range or an unbounded loop.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 4, "balance": "unbalanced", "proportions": [0.4, 0.3, 0.2, 0.1],
        "matching_mode": "custom",
        "assignment_matrix": [{"label": i, "clusters": [i], "recall_target": 0.5} for i in range(4)],
        "split_rule": "proportional_to_size", "spillover_rule": "proportional_to_marginal",
        "target_metric": {"type": "mcc", "value": 0.9, "tolerance": 0.001, "max_iter": max_iter},
    }, seed=1).to_numpy()
    assert len(out) == N


# ---------------------------------------------------------------------------
# Duplicates and collisions
# ---------------------------------------------------------------------------

def test_run_dir_collision_appends_a_numeric_suffix(tmp_path):
    """Two runs in the same wall-clock second must not share a folder.

    `build_run_dir` names folders `DDMMYY_Source_HHMMSS`, so a second run
    starting inside the same second would collide. Each existing name must push
    the suffix along rather than being reused.

    Since 0.6.3 `build_run_dir` creates each folder as it hands it out, so the
    caller no longer creates them here, doing so would now raise
    FileExistsError against the folder the previous call just made. The suffix
    behavior under test is unchanged; only who does the `mkdir` moved.
    """
    first = build_run_dir(tmp_path, "TestSource")
    second = build_run_dir(tmp_path, "TestSource")
    third = build_run_dir(tmp_path, "TestSource")

    assert len({first, second, third}) == 3, "a name was handed out twice"
    assert all(p.exists() for p in (first, second, third)), \
        "build_run_dir returned a path it did not create"


def test_byoc_duplicate_rows_are_not_silently_deduplicated(tmp_path):
    """Identical rows are more points, not a data error to be cleaned up.

    Deduplicating would change the cluster sizes the CLM maths is built on.
    """
    result = write_byoc_csv(tmp_path, "dup_rows", pd.DataFrame({
        "f1": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "f2": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "cluster": ["A", "A", "A", "B", "B", "B"],
    }))
    assert result is not None
    assert len(result) == 6


def test_byoc_duplicate_column_names_are_rejected(tmp_path, caplog):
    """pandas renames the second `f1` to `f1.1`, so the column you meant is ambiguous.

    Previously this only had to load without tripping. Since the import
    requirements landed it is a rejection: silently working with `f1.1` means
    the geometry is built from a column the user never named, and which of the
    two originals it holds depends on their order in the file.
    """
    (tmp_path / "dup_cols.csv").write_text(
        "f1,f1,cluster\n1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,A\n"
        "7.0,8.0,A\n9.0,1.0,B\n2.0,3.0,B\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        result = fetch_byoc_data(dataset_name="dup_cols", input_dir=str(tmp_path),
                                 cluster_column="cluster")
    assert result is None
    assert "duplicate column name(s)" in caplog.text


# ---------------------------------------------------------------------------
# Encoding and formatting
# ---------------------------------------------------------------------------

def test_unicode_cluster_ids_survive_load_and_labelling(tmp_path):
    """Accented, Cyrillic and emoji cluster ids, end to end.

    Cluster ids are dictionary keys throughout the engine, so they only have to
    be hashable, but they also reach the CSV and the plot legend, which is
    where an encoding assumption would surface.
    """
    result = write_byoc_csv(tmp_path, "unicode_clusters", pd.DataFrame({
        "f1": np.linspace(0, 1, 12), "f2": np.linspace(1, 0, 12),
        # RUF001 flags the Cyrillic letters as confusable with ASCII. Here that
        # is the point: the ids are non-ASCII on purpose, and a "did you mean
        # Latin?" autofix would delete the only thing this case tests.
        "cluster": ["café"] * 4 + ["\U0001F600"] * 4 + ["Зебра"] * 4,  # noqa: RUF001
    }))
    assert result is not None
    assert result["GroundTruth_labels0"].nunique() == 3

    out = generate_clm_labels(result["GroundTruth_labels0"].to_numpy(),
                              result[["f1", "f2"]].to_numpy(),
                              {"num_classes": 3, "matching_mode": "perfect"}, seed=1)
    assert len(out) == 12


@pytest.mark.parametrize("frame,expect", [
    (pd.DataFrame({"f1": [1.0] * 6, "cluster": ["A"] * 3 + ["B"] * 3,
                   "Cohort_Class": [0] * 6}), "reserved"),
    (pd.DataFrame({"f1": [1.0] * 6, "cluster": ["A"] * 3 + ["B"] * 3,
                   "GroundTruth_labels0": [0] * 6}), "reserved"),
    (pd.DataFrame({"f1": [1.0] * 6, "cluster": ["A"] * 3 + ["B"] * 3,
                   "Label_0": [0] * 6}), "reserved"),
    (pd.DataFrame({"f1": [1.0] * 6, "cluster": ["A"] * 5 + ["B"]}), "fewer than 3"),
    (pd.DataFrame({"f1": [1.0] * 6, "cluster": ["A"] * 6}), "distinct value"),
    (pd.DataFrame({"f1": [1.0] * 6, "cluster": ["A", "A", "A", "B", "B", None]}),
     "missing value"),
    (pd.DataFrame({"f1": [1.0, 1.0, 1.0, 2.0, 2.0, None],
                   "cluster": ["A"] * 3 + ["B"] * 3}), "missing value"),
    (pd.DataFrame({"f1": ["a", "b", "c", "d", "e", "f"],
                   "cluster": ["A"] * 3 + ["B"] * 3}), "non-numeric"),
], ids=["reserved-cohort", "reserved-groundtruth", "reserved-label",
        "undersized-cluster", "single-cluster", "nan-cluster", "nan-feature",
        "non-numeric-feature"])
def test_byoc_import_requirements_are_enforced(tmp_path, caplog, frame, expect):
    """BYOC is an import path, so a file that is not a usable clustering is refused.

    Deliberately not `[CLM-###]` diagnostics: those describe the cluster-label
    matching model, while these describe whether the file is a clustering at
    all. The list is expected to grow, and the manual carries the same one.
    """
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        result = write_byoc_csv(tmp_path, "bad_import", frame)
    assert result is None, f"expected rejection for: {expect}"
    assert expect in caplog.text, f"rejection did not mention {expect!r}:\n{caplog.text}"


def test_byoc_import_reports_every_problem_at_once(tmp_path, caplog):
    """One pass names everything wrong, rather than one problem per attempt."""
    with caplog.at_level(logging.ERROR, logger="clmsynth"):
        result = write_byoc_csv(tmp_path, "many_problems", pd.DataFrame({
            "f1": ["a", "b", "c", "d"], "cluster": ["A", "A", "A", "B"],
            "Cohort_Class": [0, 1, 2, 3],
        }))
    assert result is None
    for expect in ("reserved", "fewer than 3", "non-numeric"):
        assert expect in caplog.text, f"missing {expect!r} from:\n{caplog.text}"


def test_byoc_accepts_a_well_formed_import(tmp_path):
    """The complement: nothing above fires on a file that meets every requirement."""
    result = write_byoc_csv(tmp_path, "clean", pd.DataFrame({
        "f1": np.linspace(0, 1, 9), "f2": np.linspace(1, 0, 9),
        "cluster": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
    }))
    assert result is not None
    assert len(result) == 9


def test_emoji_feature_column_name_loads(tmp_path):
    """Feature names are carried through to the output CSV and plot axes verbatim."""
    result = write_byoc_csv(tmp_path, "emoji_columns", pd.DataFrame({
        "\U0001F4C8_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cluster": ["A", "A", "A", "B", "B", "B"],
    }))
    assert result is not None
    assert "\U0001F4C8_feature" in result.columns


def test_config_value_with_trailing_space_is_rejected():
    """`matching_mode: "custom "` is a typo, not a synonym.

    Config values are matched exactly. Trimming them would be leniency that
    hides the mistake, and the same leniency would have to be decided for
    every other enumerated value in the schema.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 4, "matching_mode": "custom ",
        }, seed=1)
    assert "[CLM-101]" in str(excinfo.value)


def test_num_classes_zero_is_still_uncoded_below_the_entry_point():
    """`resolve_label_counts` computes `[1.0 / M] * M`, and `1.0 / 0` is a scalar.

    The `[CLM-126]` cardinality guard lives at `generate_clm_labels`, matching
    the `_ensure_coords` design of guarding once at the entry point. Calling
    this helper directly bypasses it.
    """
    with pytest.raises(ZeroDivisionError):
        resolve_label_counts({"num_classes": 0, "balance": "balanced"}, N,
                             np.random.default_rng(0))


def test_dominant_minority_with_one_class_is_uncoded_below_the_entry_point():
    """`(1 - dominant_share) / (M - 1)` divides by zero when M is 1.

    Coded as `[CLM-131]` at `generate_clm_labels` since 0.6.3 (asserted above),
    but `resolve_label_counts` is a helper *below* that boundary and stays
    uncapped, exactly as `[CLM-126]` and `_ensure_coords` do. Pinned so the
    boundary is deliberate rather than incidental; see the entry-point pair in
    `01_logic`.
    """
    with pytest.raises(ZeroDivisionError):
        resolve_label_counts({
            "num_classes": 1, "balance": "unbalanced", "skew_rule": "dominant_minority",
            "skew_params": {"dominant_index": 0, "dominant_share": 0.9},
        }, N, np.random.default_rng(0))


def test_dirichlet_alpha_zero_divides_by_zero_below_the_entry_point():
    """A THIRD ZeroDivisionError site, and not where it was thought to be.

    This was previously recorded as numpy rejecting `alpha <= 0` itself, so it
    was treated as out of the engine's hands. On numpy 2.x that is not what
    happens: `Generator.dirichlet([0, 0, 0])` returns `[0., 0., 0.]` without
    complaint, and the failure is the engine's own normalization one line
    later, `_skewed_proportions` computes `s = sum(raw)` and then
    `[x / s for x in raw]` with `s == 0`.

    Coded as `[CLM-131]` at the entry point since 0.6.3; uncapped here for the
    same reason as the case above.
    """
    with pytest.raises(ZeroDivisionError):
        resolve_label_counts({
            "num_classes": 3, "balance": "unbalanced", "skew_rule": "dirichlet",
            "skew_params": {"alpha": 0.0},
        }, N, np.random.default_rng(0))


def test_byoc_cluster_ids_are_not_whitespace_normalised(tmp_path):
    """Finding N6: 'A' and 'A ' become two clusters, silently.

    A real trap for hand-edited or exported CSVs, half a cluster acquires a
    trailing space and the dataset quietly gains a cluster, changing every
    downstream size the CLM maths depends on. Nothing warns.

    Asserted as it behaves NOW: if `byoc_source` starts stripping the column,
    this fails and should be replaced with an assertion of one cluster.
    """
    result = write_byoc_csv(tmp_path, "whitespace_clusters", pd.DataFrame({
        "f1": list(range(10)), "f2": list(range(10)),
        "cluster": ["A"] * 5 + ["A "] * 5,
    }))
    assert result is not None
    assert result["GroundTruth_labels0"].nunique() == 2
