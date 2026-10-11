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

import itertools
import logging
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from clmsynth.byoc_source import fetch_byoc_data
from clmsynth.clm_label_engine import (
    InfeasibleAllocationError,
    build_rules,
    generate_clm_labels,
    resolve_label_counts,
)
from clmsynth.metrics import (
    MetricCeilings,
    clustering_ari,
    clustering_ari_pair,
    clustering_mcc,
    clustering_mcc_pair,
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
        ({"target_metric": [_TM]}, 134),
        ({"target_metric": 0.5}, 134),
        ({"centroid_dependence": True}, 134),
        ({"centroid_dependence": [{"enabled": True}]}, 134),
        ({"competing_noise": {"cluster": 0, "label": 1, "favors": "random"}}, 134),
        ({"competing_noise": [None]}, 134),
        ({"competing_noise": [{"cluster": [0], "label": 1, "favors": "random"}]}, 134),
        ({"assignment_matrix": _ROW0}, 134),
        ({"assignment_matrix": [None, *_rows()[1:]]}, 134),
        ({"assignment_matrix": _rows(clusters=[[0]])}, 134),
        ({"matching_mode": "single", "single_match": [{"cluster": 0, "label": 0}]}, 134),
        ({"matching_mode": "single", "single_match": {"cluster": [0], "label": 0}}, 134),
        ({"matching_mode": "random", "competing_noise": "x"}, 134),
        ({"matching_mode": "perfect", "centroid_dependence": True}, 134),
        ({"proportions": [0.5, 0.6, -0.1, 0.0]}, 134),
        ({"assignment_matrix": _rows(recall_target=-0.5)}, 134),
        ({"target_metric": {**_TM, "tolerance": -0.1}}, 134),
        (
            {"centroid_dependence": {"enabled": True, "profile": "exponential", "steepness": -2.0}},
            134,
        ),
        ({"matching_mode": "single", "single_match": {"cluster": None, "label": 0}}, 134),
        ({"matching_mode": "single", "single_match": {"cluster": _NAN, "label": 0}}, 134),
        ({"assignment_matrix": _rows(clusters=[None])}, 134),
        ({"competing_noise": [{"cluster": None, "label": 1, "favors": "random"}]}, 134),
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
        "target-list",
        "target-number",
        "centroid-true",
        "centroid-list",
        "noise-mapping",
        "noise-entry-null",
        "noise-cluster-list",
        "matrix-mapping",
        "matrix-row-null",
        "clusters-nested",
        "single-match-list",
        "single-cluster-list",
        "random-noise-string",
        "perfect-centroid-true",
        "proportions-negative",
        "recall-negative",
        "tolerance-negative",
        "steepness-negative",
        "single-cluster-null",
        "single-cluster-nan",
        "clusters-null-id",
        "noise-cluster-null",
    ],
)
def test_a_value_of_the_wrong_type_is_refused(overrides, code):
    """Each value here used to run as something else, crash uncoded, or get the
    wrong code.

    true passed as 1 (and matched cluster 1), a quoted "false" switched placement
    on, NaN passed every range check, and a negative max_iter skipped the search
    refinement. Other negatives were refused with an uncoded ValueError, a
    per-dataset "unexpected error", even in keys the run never read. A block of
    the wrong shape crashed on its first read, or, as a mapping without its YAML
    dash, was reported as a missing key ([CLM-207], [CLM-205]); a cluster id given
    as a list, or written as null, was a cluster the dataset lacks ([CLM-105],
    [CLM-119]). Both of those skip one dataset. Each is now refused where the key
    is read.
    """
    cfg = base_custom_cfg(**overrides)
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == code


@pytest.mark.parametrize("cfg", [None, [1, 2], "x", 5], ids=["none", "list", "string", "int"])
def test_a_clm_label_that_is_not_a_mapping_is_refused(cfg):
    """None or a number crashed uncoded; a list or a string was reported as a
    missing num_classes ([CLM-201]), which writes the dataset unlabelled."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 134


def test_a_read_only_mapping_is_accepted_as_a_block():
    """The shape checks ask for a mapping, not a dict: a read-only mapping ran
    before they existed, as the whole clm_label block and as a block inside it."""
    centroid = MappingProxyType({"enabled": True, "favors": "core"})
    cfg: Any = MappingProxyType(base_custom_cfg(centroid_dependence=centroid))
    out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert len(out) == N


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
        {"single_match": "x"},
        {
            "matching_mode": "single",
            "single_match": {"cluster": 0, "label": 0},
            "assignment_matrix": [None],
            "target_metric": _TM,
        },
        {"centroid_dependence": {"enabled": True, "profile": "linear", "steepness": -2.0}},
        {"assignment_matrix": _rows(recall_target=-0.5), "target_metric": _TM},
        {"matching_mode": "random", "balance": "balanced", "proportions": [0.5, 0.6, -0.1, 0.0]},
    ],
    ids=[
        "numpy-bool",
        "unread-profile-and-steepness",
        "random-noise-reads-no-profile",
        "linear-reads-no-steepness",
        "numpy-int-max-iter",
        "numpy-float-value",
        "tuple-clusters-int-recall",
        "custom-reads-no-single-match",
        "single-reads-no-matrix",
        "linear-reads-no-negative-steepness",
        "target-reads-no-negative-recall",
        "balanced-reads-no-negative-proportions",
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
        ({"num_classes": 4, "matching_mode": None, "balance": "unbalanced"}, 202),
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
    ids=["missing-mode", "valueless-mode", "unknown-mode", "unknown-mode-with-target",
         "perfect-M-ne-K"],
)
def test_mode_errors_are_reported_before_sizing_errors(cfg, expected):
    """The mode decides which keys are read at all, so it is checked first.

    A sizing error used to stand in for the real one: [CLM-203] for a missing
    mode, [CLM-121] for an unknown one or for 'perfect' with M != K, and
    [CLM-111] for a misspelled mode that carries a target_metric. A mode written
    with no value is missing ([CLM-202]), not an unknown mode named None
    ([CLM-101]).
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


# The target_metric search cases below run on four clusters of 200, the catalog's
# geometry, where the band of recall levels a claim blocks is easy to read off.
EQUAL = np.repeat([0, 1, 2, 3], 200)
EQUAL_COORDS = np.random.default_rng(0).normal(size=(800, 2))

# The line the search logs per [CLM-15x] code among its blocked recall levels,
# before [CLM-120] or [CLM-306]. It quotes the code although nothing was raised.
BLOCKED_LINE = re.compile(
    r"target_metric: (\d+) probed recall level\(s\) in \[([\d.]+), ([\d.]+)\] are "
    r"infeasible; at alpha=[\d.]+: \[CLM-(\d+)\]"
)


def equal_cfg(**overrides):
    """`base_custom_cfg` sized for EQUAL: four labels of 200."""
    return base_custom_cfg(**{"proportions": [0.25] * 4, **overrides})


def over_claiming_noise():
    """Two entries each converting 70% of cluster 0's unclaimed points."""
    return [
        {"cluster": 0, "label": 1, "share": 0.7, "favors": "random"},
        {"cluster": 0, "label": 2, "share": 0.7, "favors": "random"},
    ]


def blocked_lines(caplog):
    """(code, levels, lowest alpha, highest alpha) per blocked-level line, in log order."""
    found = [BLOCKED_LINE.match(r.getMessage()) for r in caplog.records]
    return [(int(m[4]), int(m[1]), m[2], m[3]) for m in found if m]


def reported_by_306(caplog) -> float:
    """The achieved value [CLM-306] quotes for the closest level."""
    match = re.search(r"\[CLM-306\].*achieved=(-?[\d.]+)", caplog.text)
    assert match, f"no [CLM-306] logged. Captured:\n{caplog.text}"
    return float(match[1])


@pytest.mark.parametrize(
    "overrides,blocked",
    [
        (
            {
                "assignment_matrix": [
                    {"label": i, "clusters": [i], "recall_target": 0.9} for i in (1, 2, 3)
                ]
            },
            [(152, 11, "0.000", "1.000")],
        ),
        (
            {"proportions": [0.375, 0.25, 0.25, 0.125]},
            [(152, 7, "0.000", "0.600"), (150, 4, "0.700", "1.000")],
        ),
    ],
    ids=["one-code", "two-codes"],
)
def test_a_target_with_no_feasible_level_names_what_blocked_the_levels(
    overrides, blocked, caplog
):
    """[CLM-120] used to arrive alone. The search dropped each probe's
    [CLM-15x], so the user learned that no recall level fits, not why.

    Every blocked probe now keeps its error, and before [CLM-120] the search
    logs one line per code, lowest alpha first, naming the levels it blocked.
    one-code: no rule touches cluster 0, so the two entries over-claim its 200
    unclaimed points at every level. two-codes: label 0 holds 300 points; the
    entries over-claim whatever its rule leaves of cluster 0 until the rule fills
    the cluster at alpha 2/3, which the grid does not probe, and from 0.7 up the
    rule needs more than the cluster's 200 points. Nothing is delivered, and no
    [CLM-306] is logged: there is no closest level to report.
    """
    cfg = equal_cfg(
        competing_noise=over_claiming_noise(),
        target_metric={"type": "mcc", "value": 0.5},
        **overrides,
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        with pytest.raises(ValueError) as excinfo:
            generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42)
    assert not isinstance(excinfo.value, InfeasibleAllocationError), str(excinfo.value)
    assert getattr(excinfo.value, "code", None) == 120, str(excinfo.value)
    assert blocked_lines(caplog) == blocked, caplog.text
    assert "[CLM-306]" not in caplog.text, caplog.text


def test_over_claiming_noise_without_a_target_still_raises_152():
    """Without target_metric there is one recall level, the configured one, and an
    over-claim there is what it always was: an InfeasibleAllocationError coded
    [CLM-152] (cluster 0 keeps 20 of its 200 points unclaimed at recall 0.9, and
    the entries claim 14 each). Making the search report its blocked levels
    must not have touched this path; test_04_failure_modes covers the pipeline
    half, a dataset written without the label.
    """
    cfg = equal_cfg(competing_noise=over_claiming_noise())
    with pytest.raises(InfeasibleAllocationError) as excinfo:
        generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42)
    assert getattr(excinfo.value, "code", None) == 152, str(excinfo.value)
    assert "jointly claim 28 points but only 20" in str(excinfo.value), str(excinfo.value)


def test_a_missed_target_delivers_the_closest_label_and_names_the_blocked_levels(caplog):
    """Label 0 holds 776 of 800 points, so from alpha 0.3 up its rule needs more
    than cluster 0's 200 points ([CLM-150]), and no feasible level reaches 0.5.

    The miss used to be reported twice and without its cause: [CLM-306] and
    [CLM-309] for the same value, nothing about the blocked levels. Now the
    [CLM-150] line names them, [CLM-306] follows, and the closest label is
    delivered with the value [CLM-306] quotes: on the run seed the search scores
    the labeling it delivers. No [CLM-309], nothing diverged; no [CLM-120], some
    level was feasible.
    """
    cfg = equal_cfg(
        proportions=[0.97, 0.01, 0.01, 0.01], target_metric={"type": "mcc", "value": 0.5}
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42).to_numpy()

    assert blocked_lines(caplog) == [(150, 8, "0.300", "1.000")], caplog.text
    messages = [r.getMessage() for r in caplog.records]
    first_blocked = next(i for i, m in enumerate(messages) if BLOCKED_LINE.match(m))
    missed = [i for i, m in enumerate(messages) if "[CLM-306]" in m]
    assert len(missed) == 1 and first_blocked < missed[0], caplog.text
    assert "[CLM-120]" not in caplog.text and "[CLM-309]" not in caplog.text, caplog.text
    assert list(np.bincount(out, minlength=4)) == [776, 8, 8, 8]
    assert clustering_mcc(EQUAL, out) == pytest.approx(reported_by_306(caplog), abs=5e-5)


def test_150_under_a_target_names_the_search_level_not_a_recall_target(caplog):
    """Under target_metric a [CLM-150] comes from a search probe, whose recall is
    the level being probed: recall_target is not read ([CLM-303]).

    Its advice used to say "The rule's recall_target=0.3 is feasible up to ...
    Lower it", pointing at a key the search ignores, and printed a grid level
    raw (0.7000000000000001). It now names the level, to three places, and the
    highest level that fits: 200 of label 0's 776 points is 0.2577, rounded down.
    """
    cfg = equal_cfg(
        proportions=[0.97, 0.01, 0.01, 0.01], target_metric={"type": "mcc", "value": 0.5}
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42)

    line = next(r.getMessage() for r in caplog.records if BLOCKED_LINE.match(r.getMessage()))
    assert "search's recall level alpha=0.300" in line, line
    assert "levels up to 0.257 fit here" in line, line
    assert "Lower it" not in line, line
    assert not re.search(r"\d\.\d{5,}", line), line


@pytest.mark.parametrize(
    "probe_seed,diverges",
    [(None, False), (42, False), (7, True)],
    ids=["default", "run-seed", "other-seed"],
)
def test_309_reports_only_a_probe_seed_divergence(probe_seed, diverges, caplog):
    """[CLM-309] used to fire on every delivered value outside tolerance, so each
    missed target came with [CLM-306] and [CLM-309] at once.

    It now reports one thing: the delivered labeling is not the one the search
    scored at that alpha. That takes a target_metric.probe_seed other than the
    run seed (42 here); left out, or set to the run seed, the search scores the
    labeling it delivers. The missed config above: the label is delivered every
    time, and only the divergent run delivers a value [CLM-306] did not quote.
    """
    tm: dict[str, Any] = {"type": "mcc", "value": 0.5}
    if probe_seed is not None:
        tm["probe_seed"] = probe_seed
    cfg = equal_cfg(proportions=[0.97, 0.01, 0.01, 0.01], target_metric=tm)
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42).to_numpy()

    assert list(np.bincount(out, minlength=4)) == [776, 8, 8, 8]
    assert ("[CLM-309]" in caplog.text) is diverges, caplog.text
    delivered = clustering_mcc(EQUAL, out)
    assert (abs(delivered - reported_by_306(caplog)) > 5e-5) is diverges, caplog.text


def test_305_is_not_warned_by_a_search_probe(caplog):
    """[CLM-305] says an entry had no effect: its cluster had no unclaimed point.

    Every search probe used to warn it too, so the probe at alpha 1, where rule 0
    fills cluster 0, reported an entry that the delivered alpha 0.5 applies:
    the rule leaves 100 of cluster 0's points unclaimed and the entry converts
    half of them to label 1. Probes no longer warn it; the delivered labeling
    has the entry's points and the log says nothing against them.
    """
    cfg = equal_cfg(
        competing_noise=[{"cluster": 0, "label": 1, "share": 0.5, "favors": "random"}],
        target_metric={"type": "mcc", "value": 0.5, "tolerance": 0.05},
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42).to_numpy()
    assert "[CLM-305]" not in caplog.text, caplog.text
    assert int(np.sum(out[EQUAL == 0] == 1)) >= 50, "the entry placed nothing in cluster 0"


def test_305_is_still_warned_when_the_delivered_alpha_fills_the_cluster(caplog):
    """The other half: the warning is true of the delivered labeling, so it stays.

    Four clusters of 50, two entries claiming 90% of cluster 0's unclaimed points
    each: below alpha 1 they over-claim ([CLM-152]), so the search delivers alpha
    1, where rule 0 takes the whole cluster and both entries have nothing left.
    One [CLM-305] per entry, and cluster 0 holds label 0 only.
    """
    clusters = np.repeat([0, 1, 2, 3], 50)
    cfg = equal_cfg(
        competing_noise=[
            {"cluster": 0, "label": 1, "share": 0.9, "favors": "random"},
            {"cluster": 0, "label": 2, "share": 0.9, "favors": "random"},
        ],
        target_metric={"type": "mcc", "value": 0.5},
    )
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(clusters, EQUAL_COORDS[:200], cfg, seed=42).to_numpy()
    assert re.search(r"\[CLM-306\].*alpha=1\.0000", caplog.text), caplog.text
    assert caplog.text.count("[CLM-305]") == 2, caplog.text
    assert set(out[clusters == 0].tolist()) == {0}


# ---------------------------------------------------------------------------
# Size ceiling: the highest value any labeling with the label counts can score
# ---------------------------------------------------------------------------


def three_label_cfg(**overrides):
    """Three labels of 272/264/264 on EQUAL, label i in cluster i, cluster 3 left
    to spillover. No recall_target: every case here sets a target_metric."""
    return base_custom_cfg(
        **{
            "num_classes": 3,
            "proportions": [0.34, 0.33, 0.33],
            "assignment_matrix": [{"label": i, "clusters": [i]} for i in range(3)],
            **overrides,
        }
    )


def every_labeling(sizes, M):
    """(clusters, labeling, sorted label counts) for every K x M table with row sums `sizes`."""
    clusters = np.repeat(np.arange(len(sizes)), sizes)
    rows = [[r for r in itertools.product(range(n + 1), repeat=M) if sum(r) == n] for n in sizes]
    for table in itertools.product(*rows):
        labeling = np.concatenate([np.repeat(np.arange(M), row) for row in table])
        yield clusters, labeling, tuple(sorted(np.sum(table, axis=0).tolist(), reverse=True))


@pytest.mark.parametrize(
    "sizes,M",
    [((3, 2, 1), 3), ((2, 2, 2), 2), ((2, 2, 2), 3), ((4, 2), 3), ((2, 2, 1, 1), 2),
     ((2, 2, 1, 1), 3)],
)
def test_rk_ceiling_is_the_highest_mcc_any_labeling_with_the_counts_scores(sizes, M):
    """The size ceiling the engine reports is MetricCeilings.rk_ceiling.

    Every labeling of a small geometry, scored by the engine's own
    clustering_mcc: for each set of label counts the highest value equals the
    ceiling, never more and never less. ari_bound is only a bound: no labeling
    exceeds it, and some counts never reach it.
    """
    best_mcc: dict[tuple, float] = {}
    best_ari: dict[tuple, float] = {}
    for clusters, labeling, counts in every_labeling(sizes, M):
        best_mcc[counts] = max(best_mcc.get(counts, -1.0), clustering_mcc(clusters, labeling))
        best_ari[counts] = max(best_ari.get(counts, -1.0), clustering_ari(clusters, labeling))
    for counts, top in best_mcc.items():
        assert top == pytest.approx(MetricCeilings.rk_ceiling(sizes, counts), abs=1e-12), counts
        assert best_ari[counts] <= MetricCeilings.ari_bound(sizes, counts) + 1e-12, counts


def test_the_size_ceiling_follows_the_label_counts_not_m_and_k():
    """sqrt(M(M-1)/(K(K-1))) is the ceiling of a balanced merge of equal clusters
    only. Four clusters of 100 merged 3+1 into two labels score 0.4714, above its
    0.4082. The size ceiling reads the label counts: it is 0.4082 for 200/200 and
    0.4714 for 300/100, and the 3+1 merge reaches it.
    """
    clusters = np.repeat([0, 1, 2, 3], 100)
    merged = np.where(clusters < 3, 0, 1)
    closed_form = math.sqrt(2 * 1 / (4 * 3))
    assert clustering_mcc(clusters, merged) > closed_form + 0.06
    assert MetricCeilings.rk_ceiling([100] * 4, [300, 100]) == pytest.approx(
        clustering_mcc(clusters, merged), abs=1e-12
    )
    assert MetricCeilings.rk_ceiling([100] * 4, [200, 200]) == pytest.approx(closed_form, abs=1e-12)


@pytest.mark.parametrize(
    "kind,name,ceiling",
    [("mcc", "ceiling", MetricCeilings.rk_ceiling), ("ari", "bound", MetricCeilings.ari_bound)],
    ids=["mcc", "ari"],
)
def test_a_missed_target_names_the_size_ceiling(kind, name, ceiling, caplog):
    """[CLM-306] used to name only the closest value it found, so a target no
    labeling can reach read like a search that ran out of iterations. With the
    label counts fixed it now also names their size ceiling (a bound for ari),
    and says when the target is above it. It fires exactly as before.
    """
    cfg = three_label_cfg(target_metric={"type": kind, "value": 0.99, "tolerance": 0.001})
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42)
    value = ceiling([200] * 4, [272, 264, 264])
    line = next(r.getMessage() for r in caplog.records if "[CLM-306]" in r.getMessage())
    assert line.endswith(
        f"Size {name} for these label counts: {kind}={value:.4f}; the target is above "
        "it, so no labeling with these counts reaches it."
    ), line


@pytest.mark.parametrize(
    "overrides,ceiling",
    [({}, True), ({"spillover_rule": "uniform"}, False)],
    ids=["counts-fixed", "counts-can-move"],
)
def test_a_negative_mcc_target_is_named_as_unreachable(overrides, ceiling, caplog):
    """The matching maximises R_K and R_K averages 0 over all matchings, so no
    labeling scores below 0 whatever its counts. [CLM-306] says so for a target
    below 0, with or without a size ceiling to name. It fires as before.
    """
    cfg = three_label_cfg(target_metric={"type": "mcc", "value": -0.3}, **overrides)
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        out = generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42).to_numpy()
    line = next(r.getMessage() for r in caplog.records if "[CLM-306]" in r.getMessage())
    assert line.endswith(" mcc is never below 0, so no labeling reaches the target."), line
    assert ("Size ceiling" in line) is ceiling, line
    assert clustering_mcc(EQUAL, out) >= 0


def test_the_report_names_the_request_the_delivery_the_highest_and_the_ceiling(caplog):
    """`report` receives what was only logged: the request, the delivered value,
    the highest value the search's probes scored, and the size ceiling.

    The target 0.5 is met at alpha 0.527; the probes saw 0.6594 at alpha 0.7,
    which is recorded too. The ceiling of 272/264/264 on four clusters of 200 is
    0.7071. Reported only: a run without `report` delivers the same labels.
    """
    cfg = three_label_cfg(target_metric={"type": "mcc", "value": 0.5})
    report: dict[str, Any] = {}
    with caplog.at_level(logging.INFO, logger="clmsynth"):
        out = generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42, report=report).to_numpy()

    assert report["scope"] == "global" and report["type"] == "mcc"
    assert report["requested"] == 0.5
    assert report["achieved"] == pytest.approx(clustering_mcc(EQUAL, out), abs=1e-12)
    assert abs(report["achieved"] - 0.5) <= 0.01
    assert report["highest"] == pytest.approx(0.6594, abs=1e-4)
    assert report["highest_alpha"] == pytest.approx(0.7)
    assert report["ceiling"] == pytest.approx(
        MetricCeilings.rk_ceiling([200] * 4, np.bincount(out)), abs=1e-12
    )
    assert report["achieved"] < report["highest"] < report["ceiling"]
    assert "highest mcc the search found is 0.6594 (alpha=0.7000); size ceiling 0.7071" in (
        caplog.text
    ), caplog.text
    assert np.array_equal(out, generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42).to_numpy())


@pytest.mark.parametrize(
    "overrides,cause",
    [
        ({"spillover_rule": "uniform"}, "spillover_rule 'uniform'"),
        (
            {"spillover_rule": "concentrated", "concentrated_labels": [1]},
            "spillover_rule 'concentrated'",
        ),
        (
            {"competing_noise": [{"cluster": 3, "label": 1, "share": 0.5, "favors": "random"}]},
            "competing_noise",
        ),
    ],
    ids=["uniform", "concentrated", "competing_noise"],
)
def test_no_size_ceiling_is_reported_when_the_counts_can_move(overrides, cause, caplog):
    """A size ceiling holds for one set of label counts. These settings can
    deliver other counts at other recall levels, so none is reported: the log
    names the setting, [CLM-306] names no ceiling, and `report` holds None.
    """
    cfg = three_label_cfg(target_metric={"type": "mcc", "value": 0.99}, **overrides)
    report: dict[str, Any] = {}
    with caplog.at_level(logging.INFO, logger="clmsynth"):
        generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42, report=report)
    assert report["ceiling"] is None
    assert f"no size ceiling, since {cause} can move the label counts" in caplog.text, caplog.text
    assert "[CLM-306]" in caplog.text and "Size ceiling" not in caplog.text, caplog.text


def test_moved_counts_can_score_above_the_ceiling_of_their_targets():
    """Why the counts must be fixed. Two labels of 500 on 400/300/200/100, label 0
    in cluster 0 and the rest concentrated on label 1: at alpha 0.8 label 0 fills
    cluster 0 and the delivered counts are 400/600, scoring 0.6211, above 0.5916,
    the ceiling of the 500/500 targets.
    """
    cfg = {
        "num_classes": 2,
        "balance": "balanced",
        "matching_mode": "custom",
        "assignment_matrix": [{"label": 0, "clusters": [0]}],
        "spillover_rule": "concentrated",
        "concentrated_labels": [1],
        "target_metric": {"type": "mcc", "value": 1.0},
    }
    report: dict[str, Any] = {}
    out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=42, report=report).to_numpy()
    assert list(np.bincount(out)) == [400, 600]
    assert report["achieved"] == pytest.approx(0.6211, abs=1e-4)
    assert report["achieved"] > MetricCeilings.rk_ceiling([400, 300, 200, 100], [500, 500]) + 0.02
    assert report["ceiling"] is None


def test_the_report_is_filled_only_under_a_target():
    """scope 'pair' reports its request and delivery and no ceiling: its range is
    [CLM-307]'s. Without a target_metric the report is left untouched."""
    pair: dict[str, Any] = {}
    cfg = {
        "num_classes": 4,
        "balance": "balanced",
        "matching_mode": "single",
        "single_match": {"cluster": 0, "label": 0},
        "target_metric": {"type": "mcc", "value": 0.6, "scope": "pair"},
    }
    out = generate_clm_labels(EQUAL, EQUAL_COORDS, cfg, seed=42, report=pair).to_numpy()
    assert pair == {
        "scope": "pair",
        "type": "mcc",
        "requested": 0.6,
        "achieved": clustering_mcc_pair(EQUAL, out, 0, 0),
    }

    untouched: dict[str, Any] = {}
    generate_clm_labels(EQUAL, EQUAL_COORDS, equal_cfg(), seed=42, report=untouched)
    assert untouched == {}


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
                "num_classes": 4,
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


# Pair ARI. The engine refuses it with [CLM-123]: setting ARI to a target has no
# closed-form solution. The search below stays in the package, unused, and is
# tested on its own.


def test_the_pair_ari_curve_is_sklearns_ari():
    """The pair ARI of the single-dominant construction over the label size
    m = 1..n_k (all of l* inside k*). Every point of it equals sklearn's
    adjusted_rand_score on the two binary partitions, and clustering_ari_pair,
    including the degenerate sizes: a cluster of one point, and a cluster
    holding every point.
    """
    from sklearn.metrics import adjusted_rand_score

    from clmsynth.clm_label_engine import _pair_ari_curve

    for N in (2, 3, 7, 40):
        for n_k in range(1, N + 1):
            curve = _pair_ari_curve(N, n_k)
            cluster = np.r_[np.ones(n_k), np.zeros(N - n_k)]
            for m in range(1, n_k + 1):
                label = np.r_[np.ones(m), np.zeros(N - m)]
                assert curve[m - 1] == pytest.approx(
                    adjusted_rand_score(cluster, label), abs=1e-12
                ), (N, n_k, m)
                assert curve[m - 1] == pytest.approx(
                    clustering_ari_pair(cluster, label, 1, 1), abs=1e-12
                ), (N, n_k, m)


@pytest.mark.parametrize(
    "n_k,value",
    [(250, 0.5), (600, 0.5), (600, -0.05)],
    ids=["rising", "dipping-cluster", "inside-the-dip"],
)
def test_the_pair_ari_size_is_the_closest_on_the_rising_side(n_k, value):
    """The size of l* whose pair ARI comes closest to the target, searched over
    every size of 1000 points. Past half the points the curve dips below 0
    before it rises to 1; inside the dip two sizes score alike, and the larger
    is taken, the side every target above the dip lands on.
    """
    from clmsynth.clm_label_engine import _pair_ari_curve, _pair_ari_size

    curve = _pair_ari_curve(1000, n_k)
    low = int(np.argmin(curve))
    m = _pair_ari_size(1000, n_k, value)
    assert m >= low + 1, (m, low + 1)
    assert np.abs(curve[m - 1] - value) == pytest.approx(np.abs(curve[low:] - value).min())


def test_the_pair_ari_floor_is_the_bottom_of_its_curve():
    """pair_reachable_range(kind='ari') reports the bottom of the curve, which is
    not at m = 1 when the cluster holds more than half the points: 600 of 1000
    bottom out below their value at m = 1, itself below 0.
    """
    from clmsynth.clm_label_engine import _pair_ari_curve, pair_reachable_range

    floor, top = pair_reachable_range(1000, 600, "ari")
    curve = _pair_ari_curve(1000, 600)
    assert top == 1.0 and floor == curve.min() < curve[0] < 0


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

    `custom` ([CLM-124]) and `ari` ([CLM-123]) are each rejected under
    `scope: pair`; together under the default global scope they are a valid
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
    cfg: dict[str, Any] = {
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
