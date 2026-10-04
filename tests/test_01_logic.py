"""Category 1: Logic -- the selective half.
When `00_contract` fails you know what is wrong.
This module is a selective test: manually curated cases aimed at named suspicions and
at specific bugs that have already shipped once. It can grow in the future to catch bugs
that have been identified after each release.

Everything here exercises `clm_label_engine` in-process. No subprocess, no
network, no filesystem.
Several codes asserted here also appear in `06_diagnostics`. That is not
duplication: `06` drives whole configs through the pipeline, these call the
engine directly, and a fault in config plumbing would show in one and not the
other.
"""

import logging

import numpy as np
import pytest

from clmsynth.byoc_source import fetch_byoc_data
from clmsynth.clm_label_engine import (
    InfeasibleAllocationError,
    build_rules,
    generate_clm_labels,
    resolve_label_counts,
)

# Fixture geometry: K=4, deliberately unequal (400/300/200/100) so that
# capacity boundaries differ per cluster and a split rule has something to
# divide unevenly.
N = 1000
CLUSTERS = np.concatenate([np.full(400, 0), np.full(300, 1), np.full(200, 2), np.full(100, 3)])
CLUSTER_SIZES = {0: 400, 1: 300, 2: 200, 3: 100}

# Each cluster nudged around a distinct centroid, so centroid-distance maths is
# non-degenerate rather than operating on one indistinct blob.
_rng = np.random.default_rng(0)
COORDS = _rng.normal(size=(N, 2))
for _k, (_dx, _dy) in {0: (0, 0), 1: (10, 0), 2: (0, 10), 3: (10, 10)}.items():
    COORDS[CLUSTERS == _k] += (_dx, _dy)


def base_custom_cfg(**overrides):
    """One rule per cluster at high recall; feasible against the sizes above."""
    cfg = {
        "num_classes": 4,
        "balance": "unbalanced",
        "proportions": [0.4, 0.3, 0.2, 0.1],
        "matching_mode": "custom",
        "assignment_matrix": [
            {"label": i, "clusters": [i], "recall_target": 0.9} for i in range(4)
        ],
        "split_rule": "proportional_to_size",
        "spillover_rule": "proportional_to_marginal",
    }
    cfg.update(overrides)
    return cfg


def assert_contingency_invariant(out, M):
    """Every point labeled, exactly N labels, all in range.

    `-1` is the engine's internal unassigned sentinel; one surviving into the
    output means a point fell through allocation and placement both.
    """
    arr = np.asarray(out)
    assert arr.min() >= 0, f"unlabelled (-1) points survived: min={arr.min()}"
    assert arr.max() < M, f"label outside [0,{M}): max={arr.max()}"
    assert len(arr) == N, f"length changed: {len(arr)} != {N}"


# ---------------------------------------------------------------------------
# Regression pins
# ---------------------------------------------------------------------------


def test_f1_null_target_metric_under_random_is_a_noop():
    """`target_metric:` left empty in YAML parses to None, meaning "unset".

    Verifies whether a target is requested. Testing
    the key's presence instead made an empty block a [CLM-114] rejection under
    `random`, which is a config that asks for nothing.
    """
    out = generate_clm_labels(
        CLUSTERS,
        COORDS,
        {
            "num_classes": 4,
            "balance": "balanced",
            "matching_mode": "random",
            "target_metric": None,
        },
        seed=1,
    )
    assert_contingency_invariant(out.to_numpy(), 4)


@pytest.mark.parametrize(
    "single_match,tag",
    [
        ({"cluster": 999, "label": 0}, "[CLM-105]"),
        ({"cluster": 0, "label": 99}, "[CLM-104]"),
    ],
    ids=["unknown-cluster", "out-of-range-label"],
)
def test_f2_pair_scope_validates_single_match_before_indexing(single_match, tag):
    """`scope: pair` must check the ids before `_pair_label_counts` uses them.

    That function indexes `cluster_sizes[cluster]` and `m_counts[label]`
    directly, so without an up-front check a bad id surfaces as a raw KeyError
    or IndexError from inside the engine rather than as a config error.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            COORDS,
            {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "single",
                "single_match": single_match,
                "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair"},
            },
            seed=1,
        )
    assert tag in str(excinfo.value)


def test_f6_pair_scope_with_the_only_other_label_at_zero():
    """The `o_sum == 0` rescue path in the pair-scope resize.

    `scope: pair` resizes the target label and rescales the others to keep the
    counts summing to N. With M=2 and the other label's proportion at 0 there
    is nothing to rescale -- the branch that has to notice and fall back to a
    uniform split. Getting it wrong leaves `-1` sentinels in the output and a
    total that is not N, which is why this is asserted on the invariant rather
    than on an error.
    """
    out = generate_clm_labels(
        CLUSTERS,
        COORDS,
        {
            "num_classes": 2,
            "balance": "unbalanced",
            "proportions": [1.0, 0.0],
            "matching_mode": "single",
            "single_match": {"cluster": 0, "label": 0},
            "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair"},
        },
        seed=1,
    )
    assert_contingency_invariant(out.to_numpy(), 2)


# ---------------------------------------------------------------------------
# Order of execution: the guard must precede the code it protects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coords", [None, np.array([])], ids=["none", "empty"])
def test_centroid_placement_without_coords_is_coded(coords):
    """Placement needs geometry; asking for it without coords is a config error.

    The point is the *type*: reaching the placement stage without coords would
    raise AttributeError on a None, which tells the user nothing.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            coords,
            base_custom_cfg(
                centroid_dependence={"enabled": True, "profile": "linear", "favors": "core"},
            ),
            seed=1,
        )
    assert "[CLM-125]" in str(excinfo.value)


def test_labels_only_config_tolerates_missing_coords():
    """The other half of the guard: coords are required only when used.

    Without this, the [CLM-125] check above could be made to pass by demanding
    coords unconditionally, which would break every labels-only run.
    """
    out = generate_clm_labels(CLUSTERS, None, base_custom_cfg(), seed=1)
    assert_contingency_invariant(out.to_numpy(), 4)


def test_perfect_mode_cardinality_is_refused_at_the_entry_point():
    """M != K under `perfect` is refused early."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            CLUSTERS, COORDS, {"num_classes": 10, "matching_mode": "perfect"}, seed=1
        )
    assert "[CLM-102]" in str(excinfo.value)


def test_perfect_mode_cardinality_guard_precedes_indexing():
    """`perfect` pairs cluster i with label i, so M must equal K.

    The guard has to fire before `cluster_ids[i]` runs off the end of a K-long
    list, or M>K is an IndexError instead of an explained mismatch.
    Driven through build_rules; the entry point masks it.
    """
    with pytest.raises(ValueError) as excinfo:
        build_rules({"num_classes": 10, "matching_mode": "perfect"}, [0, 1, 2, 3])
    assert "[CLM-102]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Branching: mutually exclusive combinations
# ---------------------------------------------------------------------------


def test_skew_guard_lives_at_the_entry_point_not_in_the_helper():
    """[CLM-131] guards `generate_clm_labels`, and deliberately not below it.

    Both halves are asserted together because the boundary is the design
    decision, not an accident of where the code was written. The engine guards
    once at the entry point -- the same choice `[CLM-126]` and `_ensure_coords`
    make -- and `engine_internals.md` states that lower-level helpers like
    `resolve_label_counts` stay uncapped when called directly.

    If a future change pushes the guard down into the helper, the first half
    fails and this docstring is the argument to weigh before "fixing" it.
    """
    cfg = {
        "num_classes": 3,
        "balance": "unbalanced",
        "skew_rule": "dirichlet",
        "skew_params": {"alpha": 0.0},
        "matching_mode": "random",
    }

    with pytest.raises(ZeroDivisionError):
        resolve_label_counts(cfg, N, np.random.default_rng(0))

    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 131


def test_skew_guard_precedes_the_counts_it_protects():
    """Ordering, not merely presence.

    `resolve_label_counts` is what consumes `skew_params`, and it runs early --
    before the mode branches and well before the other three validators. A guard
    placed alongside those would have run *after* the counts it exists to
    protect were already computed, which for the three silent cases means after
    the damage.
    """
    cfg = {
        "num_classes": 4,
        "balance": "unbalanced",
        "skew_rule": "geometric",
        "skew_params": {"ratio": -0.5},
        "matching_mode": "random",
    }
    # 'random' delivers whatever counts resolve_label_counts returns, with no
    # allocation step to object: here [1600, -800, 400, -200]. Only a guard that runs
    # first can catch them. ('perfect' no longer reads skew_params at all.)
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 131


@pytest.mark.parametrize(
    "overrides,why",
    [
        ({"proportions": [0.4, 0.3, 0.2, 0.1]}, "explicit proportions supersede the skew rule"),
        ({"balance": "balanced"}, "a balanced split never consults the skew rule"),
    ],
    ids=["proportions-given", "balanced"],
)
def test_skew_params_are_unvalidated_when_they_are_never_read(overrides, why):
    """The guard uses the same predicate `resolve_label_counts` branches on.

    A config carrying a stale or nonsensical `skew_params` block it never
    consults must not be failed for it -- otherwise adding the guard would
    reject configurations that were always correct.
    """
    # Recall 0.4, not the helper's 0.9: under `balanced` every label's budget is
    # N/4 = 250, which at 0.9 over-claims the 200- and 100-point clusters for
    # reasons that have nothing to do with the skew parameters under test.
    cfg = base_custom_cfg(
        skew_rule="dirichlet",
        skew_params={"alpha": -99},
        assignment_matrix=[{"label": i, "clusters": [i], "recall_target": 0.4} for i in range(4)],
        **overrides,
    )
    out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert_contingency_invariant(out, 4)


@pytest.mark.parametrize(
    "tolerance,expect_warning", [(0.0, True), (0.5, False)], ids=["tight", "loose"]
)
def test_pair_scope_honours_the_requested_tolerance(tolerance, expect_warning, caplog):
    """`scope: pair` used to hardcode 0.01 for its delivered-value check.

    So a requested 0.001 was silently widened and a requested 0.05 silently
    narrowed. The label is sized to an integer number of points, so the achieved
    coefficient essentially never equals the request exactly: at tolerance 0.0
    the miss must be reported, and at 0.5 it must not.
    """
    cfg = {
        "num_classes": 2,
        "balance": "balanced",
        "matching_mode": "single",
        "single_match": {"cluster": 1, "label": 0},
        "spillover_rule": "proportional_to_marginal",
        "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair", "tolerance": tolerance},
    }
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert ("[CLM-310]" in caplog.text) is expect_warning, (
        f"tolerance={tolerance} should {'' if expect_warning else 'not '}have reported a miss"
    )


def test_balanced_ignores_explicit_proportions():
    """`balance: balanced` enforces uniform 1/M and ignores proportions.

    Asserted on the resulting counts, not on the [CLM-301] warning: the warning
    firing while the proportions were honored anyway is the failure this is
    here to catch.
    """
    counts = resolve_label_counts(
        {"num_classes": 4, "balance": "balanced", "proportions": [0.7, 0.1, 0.1, 0.1]},
        N,
        np.random.default_rng(0),
    )
    assert list(counts) == [250, 250, 250, 250], list(counts)


@pytest.mark.parametrize(
    "value, mode",
    [
        (None, "random"),
        ("", "single"),
        ("Balanced", "custom"),
        ("unbalance", "random"),
        (1, "single"),
        (["unbalanced"], "custom"),
    ],
    ids=["bare", "empty", "case", "typo", "int", "list"],
)
def test_balance_outside_its_two_values_is_refused(value, mode):
    """[CLM-132]: balance used to be compared only against 'balanced'.

    Every other value was read as 'unbalanced', so with proportions the typo
    'Balanced' delivered the split it was meant to ignore. An absent key is
    still the default, 'balanced'.

    The check runs before the mode is consulted, so each value is tried once,
    and the three modes that read balance take two values each.
    """
    cfg = base_custom_cfg(
        matching_mode=mode, single_match={"cluster": 0, "label": 0}, balance=value
    )
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 132


def test_balance_typo_without_proportions_is_not_blamed_on_skew_rule():
    """Without proportions, a valueless `balance:` used to read as 'unbalanced'
    and report [CLM-203], skew_rule missing: a per-dataset skip that named the
    wrong key."""
    cfg = {"num_classes": 4, "matching_mode": "random", "balance": None}
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 132


_NAN, _INF = float("nan"), float("inf")
_TM = {"type": "mcc", "value": 0.5}
_ROW0 = {"label": 0, "clusters": [0], "recall_target": 0.9}


def _rows(**row0):
    """base_custom_cfg's matrix with row 0 patched."""
    return [{**_ROW0, **row0}] + [
        {"label": i, "clusters": [i], "recall_target": 0.9} for i in range(1, 4)
    ]


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"centroid_dependence": {"enabled": "false"}}, 134),
        ({"centroid_dependence": {"enabled": "no"}}, 134),
        ({"centroid_dependence": {"enabled": None}}, 134),
        ({"matching_mode": "single", "single_match": {"cluster": True, "label": 0}}, 134),
        ({"assignment_matrix": _rows(clusters=[True])}, 134),
        ({"assignment_matrix": _rows(clusters="01")}, 134),
        ({"assignment_matrix": _rows(clusters=1)}, 134),
        ({"assignment_matrix": _rows(recall_target=True)}, 134),
        ({"assignment_matrix": _rows(recall_target=_NAN)}, 134),
        ({"assignment_matrix": _rows(recall_target=_INF)}, 134),
        ({"assignment_matrix": _rows(recall_target="0.9")}, 134),
        ({"competing_noise": [{"cluster": 0, "label": 1, "share": True}]}, 117),
        ({"competing_noise": [{"cluster": 0, "label": 1, "share": "0.5"}]}, 117),
        ({"competing_noise": [{"cluster": True, "label": 1, "favors": "random"}]}, 134),
        ({"target_metric": {**_TM, "value": True}}, 113),
        ({"target_metric": {**_TM, "value": None}}, 113),
        ({"target_metric": {**_TM, "tolerance": _NAN}}, 134),
        ({"target_metric": {**_TM, "tolerance": _INF}}, 134),
        ({"target_metric": {**_TM, "tolerance": True}}, 134),
        ({"target_metric": {**_TM, "max_iter": True}}, 134),
        ({"target_metric": {**_TM, "max_iter": 20.0}}, 134),
        ({"target_metric": {**_TM, "max_iter": -5}}, 134),
        (
            {"centroid_dependence": {"enabled": True, "profile": "exponential",
                                     "steepness": _NAN}},
            134,
        ),
        (
            {"centroid_dependence": {"enabled": True, "profile": "exponential",
                                     "steepness": True}},
            134,
        ),
        ({"proportions": [0.4, _NAN, 0.2, 0.1]}, 134),
        ({"proportions": [0.4, "0.3", 0.2, 0.1]}, 134),
        ({"proportions": [True, 0.0, 0.0, 0.0]}, 134),
        ({"proportions": None, "skew_rule": "geometric", "skew_params": {"ratio": _NAN}}, 131),
        ({"proportions": None, "skew_rule": "geometric", "skew_params": {"ratio": _INF}}, 131),
        ({"proportions": None, "skew_rule": "dirichlet", "skew_params": {"alpha": _INF}}, 131),
        ({"spillover_rule": "concentrated", "concentrated_labels": [_INF]}, 128),
    ],
    ids=[
        "enabled-quoted-false",
        "enabled-quoted-no",
        "enabled-bare",
        "single-cluster-true",
        "clusters-true",
        "clusters-string",
        "clusters-int",
        "recall-true",
        "recall-nan",
        "recall-inf",
        "recall-string",
        "share-true",
        "share-string",
        "noise-cluster-true",
        "value-true",
        "value-bare",
        "tolerance-nan",
        "tolerance-inf",
        "tolerance-true",
        "max-iter-true",
        "max-iter-float",
        "max-iter-negative",
        "steepness-nan",
        "steepness-true",
        "proportions-nan",
        "proportions-string",
        "proportions-true",
        "ratio-nan",
        "ratio-inf",
        "alpha-inf",
        "concentrated-inf",
    ],
)
def test_a_value_of_the_wrong_type_is_refused(overrides, code):
    """Each value here used to run as something else, or crash uncoded.

    true passed as 1 (and matched cluster 1), a quoted "false" switched placement
    on, NaN passed every range check, and a negative max_iter skipped the search
    refinement. Each is now refused where the key is read.
    """
    cfg = base_custom_cfg(**overrides)
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == code


def test_proportions_given_as_a_mapping_are_refused():
    """A mapping is iterated by its keys: {0: 0.8, 1: 0.2} summed 0 + 1 = 1, passed
    [CLM-106], and every point got label 1. Replaces the 0.7.1 pin that a mapping
    with string keys crashed uncoded."""
    for proportions in ({0: 0.8, 1: 0.2}, {"a": 0.5, "b": 0.5}):
        cfg = {
            "num_classes": 2,
            "matching_mode": "random",
            "balance": "unbalanced",
            "proportions": proportions,
        }
        with pytest.raises(ValueError) as excinfo:
            generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
        assert getattr(excinfo.value, "code", None) == 134


def test_a_competing_noise_entry_without_a_cluster_names_the_key():
    """A bare KeyError used to log only 'cluster'. It stays a missing key (a
    per-dataset skip), now coded and naming the entry."""
    cfg = base_custom_cfg(competing_noise=[{"label": 1, "share": 0.5, "favors": "random"}])
    with pytest.raises(KeyError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 207
    assert "competing_noise entry 0: 'cluster'" in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"centroid_dependence": {"enabled": True, "profile": "bogus"}, "assignment_matrix": []},
        {
            "centroid_dependence": {"enabled": False, "profile": "bogus"},
            "competing_noise": [{"cluster": 0, "label": 1, "share": 0.0, "favors": "boundary"}],
        },
    ],
    ids=["enabled-nothing-placed", "noise-places-nothing"],
)
def test_a_profile_typo_is_refused_even_when_nothing_is_placed(overrides):
    """[CLM-110] used to fire only when a label was actually placed by profile."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(**overrides), seed=1)
    assert getattr(excinfo.value, "code", None) == 110


@pytest.mark.parametrize(
    "overrides",
    [
        {"centroid_dependence": {"enabled": np.True_, "favors": "core"}},
        {"centroid_dependence": {"enabled": False, "profile": "bogus", "steepness": True}},
        {
            "centroid_dependence": {"profile": "bogus"},
            "competing_noise": [{"cluster": 0, "label": 1, "share": 0.5, "favors": "random"}],
        },
        {"centroid_dependence": {"enabled": True, "profile": "linear", "steepness": "x"}},
        {"target_metric": {**_TM, "max_iter": np.int64(0), "tolerance": 0}},
        {"target_metric": {**_TM, "value": np.float64(0.5)}},
        {"assignment_matrix": _rows(clusters=(0,), recall_target=1)},
    ],
    ids=[
        "numpy-bool",
        "unread-profile-and-steepness",
        "random-noise-reads-no-profile",
        "linear-reads-no-steepness",
        "numpy-int-max-iter",
        "numpy-float-value",
        "tuple-clusters-int-recall",
    ],
)
def test_values_of_the_right_type_or_unread_keys_still_run(overrides):
    """The [CLM-134] checks refuse only what is read, and accept numpy scalars."""
    out = generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(**overrides), seed=1)
    assert len(out) == N


@pytest.mark.parametrize(
    "sizing",
    [
        {"balance": "unbalanced", "proportions": [0.5, 0.5]},
        {"balance": "unbalanced", "proportions": [0.3, 0.3, 0.2, 0.1]},
        {"balance": "unbalanced", "proportions": [0.5, 0.6, -0.1, 0.0]},
        {"balance": "unbalanced"},
        {"balance": "unbalanced", "skew_rule": "bogus"},
        {"balance": "unbalanced", "skew_rule": "geometric", "skew_params": {"ratio": -0.5}},
        {"balance": "Balanced"},
        {"skew_params": {"alpha": 0}},
    ],
    ids=[
        "length-121",
        "sum-106",
        "negative",
        "no-skew-rule-203",
        "unknown-skew-107",
        "skew-params-131",
        "balance-typo-132",
        "skew-params-only",
    ],
)
def test_perfect_never_reads_the_sizing_keys(sizing, caplog):
    """'perfect' pairs label i with cluster i, so its counts are the cluster sizes.

    It used to resolve and validate balance/proportions/skew first and then
    discard the result, so keys [CLM-302] calls ignored could still abort the
    run, or leave the dataset unlabelled ([CLM-203]).
    """
    cfg = {"num_classes": 4, "matching_mode": "perfect", **sizing}
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert np.array_equal(np.asarray(out), CLUSTERS)
    assert "[CLM-302]" in caplog.text


@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({"num_classes": 4, "balance": "unbalanced"}, 202),
        (
            {
                "num_classes": 4,
                "matching_mode": "bogus",
                "balance": "unbalanced",
                "proportions": [0.5, 0.5],
            },
            101,
        ),
        (
            {
                "num_classes": 4,
                "matching_mode": "Single",
                "target_metric": {"type": "mcc", "value": 0.5},
            },
            101,
        ),
        (
            {
                "num_classes": 3,
                "matching_mode": "perfect",
                "balance": "unbalanced",
                "proportions": [0.5, 0.5],
            },
            102,
        ),
    ],
    ids=["missing-mode", "unknown-mode", "unknown-mode-with-target", "perfect-M-ne-K"],
)
def test_mode_errors_are_reported_before_sizing_errors(cfg, expected):
    """The mode decides which keys are read at all, so it is checked first.

    A sizing error used to stand in for the real one: [CLM-203] for a missing
    mode, [CLM-121] for an unknown one or for 'perfect' with M != K, and
    [CLM-111] for a misspelled mode that carries a target_metric.
    """
    with pytest.raises((ValueError, KeyError)) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == expected


@pytest.mark.parametrize(
    "rule,deviates",
    [
        ("proportional_to_marginal", False),
        ("uniform", True),
        ("concentrated", True),
    ],
    ids=["marginal", "uniform", "concentrated"],
)
def test_spillover_reports_a_broken_marginal(rule, deviates, caplog):
    """[CLM-304] fires exactly when the delivered counts left their target.

    The catalog holds one fixture per code, so the spillover arm of 304 (the
    competing_noise arm owns the fixture) can only be covered here.
    """
    cfg = base_custom_cfg(spillover_rule=rule)
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    counts = list(np.bincount(np.asarray(out), minlength=4))
    assert (counts != [400, 300, 200, 100]) is deviates, counts
    assert ("[CLM-304]" in caplog.text) is deviates, caplog.text


@pytest.mark.parametrize(
    "label1_recall,deviates", [(0.9, False), (1.0, True)], ids=["room-left", "past-target"]
)
def test_competing_noise_reports_only_counts_it_moved(label1_recall, deviates, caplog):
    """[CLM-304]'s competing_noise arm checks the delivered counts too.

    It used to fire whenever noise was placed. Under proportional_to_marginal the
    spillover pool is each label's target minus what is already placed, so noise
    on a label with room left only displaces spillover. The counts move only
    when the noise takes a label past its target: here 20 points of label 1 in
    cluster 0, on top of 270 (room left) or 300 (full) in cluster 1.
    """
    matrix = [{"label": i, "clusters": [i], "recall_target": 0.9} for i in range(4)]
    matrix[1]["recall_target"] = label1_recall
    cfg = base_custom_cfg(
        assignment_matrix=matrix,
        competing_noise=[{"cluster": 0, "label": 1, "share": 0.5, "favors": "random"}],
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    counts = list(np.bincount(np.asarray(out), minlength=4))
    assert (counts != [400, 300, 200, 100]) is deviates, counts
    assert ("[CLM-304]" in caplog.text) is deviates, caplog.text


@pytest.mark.parametrize(
    "cfg,tag",
    [
        (
            {
                "num_classes": 4,
                "matching_mode": "perfect",
                "target_metric": {"type": "mcc", "value": 0.5},
            },
            "[CLM-111]",
        ),
        (
            {
                "num_classes": 4,
                "balance": "balanced",
                "matching_mode": "random",
                "target_metric": {"type": "mcc", "value": 0.5},
            },
            "[CLM-114]",
        ),
        (
            {
                "num_classes": 4,
                "balance": "balanced",
                "matching_mode": "random",
                "competing_noise": [{"cluster": 0, "label": 1, "share": 0.5}],
            },
            "[CLM-115]",
        ),
        (
            {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "single",
                "single_match": {"cluster": 0, "label": 0},
                "target_metric": {"type": "ari", "value": 0.5, "scope": "pair"},
            },
            "[CLM-123]",
        ),
    ],
    ids=["perfect+target", "random+target", "random+competing", "pair+ari"],
)
def test_incompatible_mode_combinations_are_rejected(cfg, tag):
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert tag in str(excinfo.value)


def test_pair_scope_outside_single_mode_is_rejected():
    """The closed form is defined for one cluster/label pair, so `custom` has no pair."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            COORDS,
            base_custom_cfg(target_metric={"type": "mcc", "value": 0.5, "scope": "pair"}),
            seed=1,
        )
    assert "[CLM-124]" in str(excinfo.value)


def test_global_scope_with_ari_and_custom_mode_is_accepted():
    """A negative test: the pair-scope guards must not over-reject.

    `ari` and `custom` are each rejected under `scope: pair` ([CLM-123],
    [CLM-124]); together under the default global scope they are a valid
    request, and a guard written too broadly would refuse it.
    """
    out = generate_clm_labels(
        CLUSTERS,
        COORDS,
        base_custom_cfg(
            target_metric={"type": "ari", "value": 0.3, "tolerance": 0.05, "max_iter": 10}
        ),
        seed=1,
    )
    assert_contingency_invariant(out.to_numpy(), 4)


@pytest.mark.parametrize(
    "cfg,tag",
    [
        ({"num_classes": 4, "matching_mode": "bogus"}, "[CLM-101]"),
        (
            {
                "num_classes": 3,
                "matching_mode": "random",
                "balance": "unbalanced",
                "skew_rule": "bogus",
            },
            "[CLM-107]",
        ),
        ({"num_classes": 3, "matching_mode": "random", "balance": "Balanced"}, "[CLM-132]"),
        (
            {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": [0, 1], "recall_target": 0.5}],
                "split_rule": "bogus",
                "spillover_rule": "proportional_to_marginal",
            },
            "[CLM-108]",
        ),
        (base_custom_cfg(spillover_rule="bogus"), "[CLM-109]"),
        (base_custom_cfg(centroid_dependence={"enabled": True, "profile": "bogus"}), "[CLM-110]"),
    ],
    ids=["matching_mode", "skew_rule", "balance", "split_rule", "spillover_rule", "profile"],
)
def test_unknown_rule_names_are_rejected(cfg, tag):
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert tag in str(excinfo.value)


def test_split_rule_is_validated_wherever_custom_reads_it():
    """[CLM-108] fires under 'custom' even when every rule names one cluster.

    `_split_row_allocation` short-circuits a single-cluster rule without
    consulting `split_rule`, so a typo in that common shape used to pass
    silently, and a user correcting a `split_rule` they believed was active saw
    no change. It is now checked up front under 'custom', the one mode that
    reads it. Under the other modes it is still not read, so still not checked.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(split_rule="bogus"), seed=1)
    assert "[CLM-108]" in str(excinfo.value)

    single = {
        "num_classes": 4,
        "balance": "balanced",
        "matching_mode": "single",
        "single_match": {"cluster": 0, "label": 0},
        "split_rule": "bogus",
    }
    out = generate_clm_labels(CLUSTERS, COORDS, single, seed=1)
    assert_contingency_invariant(out.to_numpy(), 4)


# ---------------------------------------------------------------------------
# Exact boundaries
# ---------------------------------------------------------------------------


def test_single_label_space():
    """M=1: every point takes label 0. Degenerate, and must stay non-crashing."""
    out = generate_clm_labels(
        CLUSTERS,
        COORDS,
        {
            "num_classes": 1,
            "balance": "balanced",
            "matching_mode": "random",
        },
        seed=1,
    )
    assert_contingency_invariant(out.to_numpy(), 1)


def test_allocation_exactly_saturating_a_cluster_succeeds():
    """tp == capacity is feasible; the failure starts one point later.

    Cluster 3 holds exactly 100 points and label 0's budget is exactly 100 at
    recall 1.0, so this is the tightest allocation that can still succeed. A
    `>=` where `>` belongs would reject it.
    """
    out = generate_clm_labels(
        CLUSTERS,
        COORDS,
        {
            "num_classes": 2,
            "balance": "unbalanced",
            "proportions": [0.1, 0.9],
            "matching_mode": "custom",
            "assignment_matrix": [{"label": 0, "clusters": [3], "recall_target": 1.0}],
            "split_rule": "equal",
            "spillover_rule": "proportional_to_marginal",
        },
        seed=1,
    )
    assert_contingency_invariant(out.to_numpy(), 2)


def test_allocation_exceeding_capacity_by_one_point_is_infeasible():
    """The very next integer: 101 points demanded of a 100-point cluster."""
    with pytest.raises(InfeasibleAllocationError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            COORDS,
            {
                "num_classes": 2,
                "balance": "unbalanced",
                "proportions": [0.101, 0.899],
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": [3], "recall_target": 1.0}],
                "split_rule": "equal",
                "spillover_rule": "proportional_to_marginal",
            },
            seed=1,
        )
    assert "[CLM-150]" in str(excinfo.value)
    assert "proportions[0] = 0.101" in str(excinfo.value)


def test_infeasible_single_blames_the_label_size_not_recall_target():
    """'single' has no recall_target key: it places the whole label in its cluster.

    [CLM-150] used to advise a 'max feasible recall_target', a key the user
    cannot set in this mode. Only a smaller label helps, so the message names
    the setting the label's size came from.
    """
    cfg = {
        "num_classes": 4,
        "balance": "unbalanced",
        "proportions": [0.1, 0.2, 0.3, 0.4],
        "matching_mode": "single",
        "single_match": {"cluster": 3, "label": 3},
        "spillover_rule": "proportional_to_marginal",
    }
    with pytest.raises(InfeasibleAllocationError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    message = str(excinfo.value)
    assert "[CLM-150]" in message
    assert "recall_target" not in message
    assert "proportions[3] = 0.4" in message


def test_infeasible_advice_is_itself_feasible():
    """The advised recall_target must work when the user applies it.

    2000 / 3000 = 0.6667 rounded to three places is 0.667, and 0.667 x 3000 =
    2001 points, one more than the cluster holds; the advice is rounded down.
    A recall_target above 1 (80 meant as 80 %) used to be told it was feasible
    up to capacity / budget, itself above 1 and refused by [CLM-153].
    """
    clusters = np.repeat([0, 1], [2000, 1000])
    cfg = {
        "num_classes": 2,
        "balance": "unbalanced",
        "proportions": [1.0, 0.0],
        "matching_mode": "custom",
        "assignment_matrix": [{"label": 0, "clusters": [0], "recall_target": 0.8}],
        "spillover_rule": "proportional_to_marginal",
    }
    with pytest.raises(InfeasibleAllocationError) as excinfo:
        generate_clm_labels(clusters, None, cfg, seed=1)
    assert "feasible up to 0.666" in str(excinfo.value), str(excinfo.value)
    cfg["assignment_matrix"][0]["recall_target"] = 0.666
    generate_clm_labels(clusters, None, cfg, seed=1)

    over_one = base_custom_cfg()
    over_one["assignment_matrix"][3]["recall_target"] = 80
    with pytest.raises(InfeasibleAllocationError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, over_one, seed=1)
    assert "must be at most 1.0" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("value", [0.0, 1.0], ids=["independence", "ceiling"])
def test_solver_grid_endpoints_are_evaluable(value):
    """The coarse scan is `linspace(0, 1, 11)`, so both endpoints are probed.

    Neither is expected to be *reached* -- 1.0 sits above the structural
    ceiling for this geometry, which is what [CLM-306] reports -- but both must
    be evaluable rather than crashing the bracket search.
    """
    out = generate_clm_labels(
        CLUSTERS,
        COORDS,
        base_custom_cfg(
            target_metric={"type": "mcc", "value": value, "tolerance": 0.05, "max_iter": 20}
        ),
        seed=1,
    )
    assert_contingency_invariant(out.to_numpy(), 4)


def test_dirichlet_alpha_controls_concentration():
    """`alpha` must actually shape the draw, not just be accepted.

    This is the defining property of the parameter: small alpha concentrates
    almost all mass on one label, large alpha approaches a uniform split. The
    gap it guards is that nothing else in the suite would notice `skew_params`
    being ignored -- `00_contract` runs one dirichlet case at alpha=1.0 and
    asserts the generic invariants, all of which hold just as well if the
    engine silently drew a fixed alpha every time.

    Compared across several seeds rather than one, so the assertion rests on
    the distribution's behavior and not on a single lucky draw.
    """

    def spread(alpha, seed):
        counts = resolve_label_counts(
            {
                "num_classes": 5,
                "balance": "unbalanced",
                "skew_rule": "dirichlet",
                "skew_params": {"alpha": alpha},
            },
            N,
            np.random.default_rng(seed),
        )
        assert sum(counts) == N, "dirichlet draw did not partition N"
        return max(counts) - min(counts)

    for seed in range(5):
        concentrated = spread(0.1, seed)
        near_uniform = spread(100.0, seed)
        assert concentrated > near_uniform, (
            "alpha=0.1 should be more lopsided than alpha=100 "
            f"(seed {seed}: spread {concentrated} vs {near_uniform})"
        )


def test_dirichlet_is_stochastic_across_seeds():
    """Reproducible per seed, different between seeds.

    `00_contract` asserts the first half for every rule. The second half is
    specific to dirichlet: it is the only skew rule that draws, so it is the
    only one where two seeds are expected to disagree. `geometric` and
    `dominant_minority` are deterministic functions of their parameters.
    """

    def counts(seed):
        return list(
            resolve_label_counts(
                {
                    "num_classes": 5,
                    "balance": "unbalanced",
                    "skew_rule": "dirichlet",
                    "skew_params": {"alpha": 0.5},
                },
                N,
                np.random.default_rng(seed),
            )
        )

    assert counts(1) == counts(1)
    assert counts(1) != counts(2)


def test_resolve_label_counts_at_zero_rows():
    """N=0: the largest-remainder split has nothing to distribute."""
    counts = resolve_label_counts(
        {"num_classes": 3, "balance": "balanced"}, 0, np.random.default_rng(0)
    )
    assert list(counts) == [0, 0, 0], list(counts)


# ---------------------------------------------------------------------------
# Data-source degradation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_name": "does_not_exist"},
        {"dataset_name": None},
    ],
    ids=["missing-file", "no-name"],
)
def test_byoc_returns_none_rather_than_raising(kwargs):
    """The source contract: unusable input is None (logged), never an exception.

    `run_pipeline` treats None as "skip this dataset"; a raised
    FileNotFoundError would instead abort the batch.
    """
    assert fetch_byoc_data(cluster_column="c", **kwargs) is None
