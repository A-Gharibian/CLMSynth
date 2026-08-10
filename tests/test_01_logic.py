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

    Truthiness, not presence, decides whether a target is requested. Testing
    the key's presence instead made an empty block a [CLM-114] rejection under
    `random`, which is a config that asks for nothing.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 4, "balance": "balanced", "matching_mode": "random",
        "target_metric": None,
    }, seed=1)
    assert_contingency_invariant(out.to_numpy(), 4)


@pytest.mark.parametrize("single_match,tag", [
    ({"cluster": 999, "label": 0}, "[CLM-105]"),
    ({"cluster": 0, "label": 99}, "[CLM-104]"),
], ids=["unknown-cluster", "out-of-range-label"])
def test_f2_pair_scope_validates_single_match_before_indexing(single_match, tag):
    """`scope: pair` must check the ids before `_pair_label_counts` uses them.

    That function indexes `cluster_sizes[cluster]` and `m_counts[label]`
    directly, so without an up-front check a bad id surfaces as a raw KeyError
    or IndexError from inside the engine rather than as a config error.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 2, "balance": "balanced", "matching_mode": "single",
            "single_match": single_match,
            "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair"},
        }, seed=1)
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
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 2, "balance": "unbalanced", "proportions": [1.0, 0.0],
        "matching_mode": "single", "single_match": {"cluster": 0, "label": 0},
        "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair"},
    }, seed=1)
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
        generate_clm_labels(CLUSTERS, coords, base_custom_cfg(
            centroid_dependence={"enabled": True, "profile": "linear", "favors": "core"},
        ), seed=1)
    assert "[CLM-125]" in str(excinfo.value)


def test_labels_only_config_tolerates_missing_coords():
    """The other half of the guard: coords are required only when used.

    Without this, the [CLM-125] check above could be made to pass by demanding
    coords unconditionally, which would break every labels-only run.
    """
    out = generate_clm_labels(CLUSTERS, None, base_custom_cfg(), seed=1)
    assert_contingency_invariant(out.to_numpy(), 4)


def test_perfect_mode_cardinality_guard_precedes_indexing():
    """`perfect` pairs cluster i with label i, so M must equal K.

    The guard has to fire before `cluster_ids[i]` runs off the end of a K-long
    list, or M>K is an IndexError instead of an explained mismatch.
    """
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {"num_classes": 10, "matching_mode": "perfect"}, seed=1)
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
    cfg = {"num_classes": 3, "balance": "unbalanced", "skew_rule": "dirichlet",
           "skew_params": {"alpha": 0.0}, "matching_mode": "random"}

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
    cfg = {"num_classes": 4, "balance": "unbalanced", "skew_rule": "geometric",
           "skew_params": {"ratio": -0.5}, "matching_mode": "perfect"}
    # 'perfect' with M != K would raise [CLM-102] from its own guard; the skew
    # rejection has to win, which it can only do by running first.
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert getattr(excinfo.value, "code", None) == 131


@pytest.mark.parametrize("overrides,why", [
    ({"proportions": [0.4, 0.3, 0.2, 0.1]}, "explicit proportions supersede the skew rule"),
    ({"balance": "balanced"}, "a balanced split never consults the skew rule"),
], ids=["proportions-given", "balanced"])
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
        skew_rule="dirichlet", skew_params={"alpha": -99},
        assignment_matrix=[{"label": i, "clusters": [i], "recall_target": 0.4}
                           for i in range(4)],
        **overrides)
    out = generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert_contingency_invariant(out, 4)


@pytest.mark.parametrize("tolerance,expect_warning", [(0.0, True), (0.5, False)],
                         ids=["tight", "loose"])
def test_pair_scope_honours_the_requested_tolerance(tolerance, expect_warning, caplog):
    """`scope: pair` used to hardcode 0.01 for its delivered-value check.

    So a requested 0.001 was silently widened and a requested 0.05 silently
    narrowed. The label is sized to an integer number of points, so the achieved
    coefficient essentially never equals the request exactly: at tolerance 0.0
    the miss must be reported, and at 0.5 it must not.
    """
    cfg = {"num_classes": 2, "balance": "balanced", "matching_mode": "single",
           "single_match": {"cluster": 1, "label": 0},
           "spillover_rule": "proportional_to_marginal",
           "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair",
                             "tolerance": tolerance}}
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert ("[CLM-310]" in caplog.text) is expect_warning, \
        f"tolerance={tolerance} should {'' if expect_warning else 'not '}have reported a miss"


def test_balanced_ignores_explicit_proportions():
    """`balance: balanced` enforces uniform 1/M and ignores proportions.

    Asserted on the resulting counts, not on the [CLM-301] warning: the warning
    firing while the proportions were honored anyway is the failure this is
    here to catch.
    """
    counts = resolve_label_counts(
        {"num_classes": 4, "balance": "balanced", "proportions": [0.7, 0.1, 0.1, 0.1]},
        N, np.random.default_rng(0))
    assert list(counts) == [250, 250, 250, 250], list(counts)


@pytest.mark.parametrize("cfg,tag", [
    ({"num_classes": 4, "matching_mode": "perfect",
      "target_metric": {"type": "mcc", "value": 0.5}}, "[CLM-111]"),
    ({"num_classes": 4, "balance": "balanced", "matching_mode": "random",
      "target_metric": {"type": "mcc", "value": 0.5}}, "[CLM-114]"),
    ({"num_classes": 4, "balance": "balanced", "matching_mode": "random",
      "competing_noise": [{"cluster": 0, "label": 1, "share": 0.5}]}, "[CLM-115]"),
    ({"num_classes": 2, "balance": "balanced", "matching_mode": "single",
      "single_match": {"cluster": 0, "label": 0},
      "target_metric": {"type": "ari", "value": 0.5, "scope": "pair"}}, "[CLM-123]"),
], ids=["perfect+target", "random+target", "random+competing", "pair+ari"])
def test_incompatible_mode_combinations_are_rejected(cfg, tag):
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert tag in str(excinfo.value)


def test_pair_scope_outside_single_mode_is_rejected():
    """The closed form is defined for one cluster/label pair, so `custom` has no pair."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(
            target_metric={"type": "mcc", "value": 0.5, "scope": "pair"}), seed=1)
    assert "[CLM-124]" in str(excinfo.value)


def test_global_scope_with_ari_and_custom_mode_is_accepted():
    """A negative test: the pair-scope guards must not over-reject.

    `ari` and `custom` are each rejected under `scope: pair` ([CLM-123],
    [CLM-124]); together under the default global scope they are a valid
    request, and a guard written too broadly would refuse it.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(
        target_metric={"type": "ari", "value": 0.3, "tolerance": 0.05, "max_iter": 10}), seed=1)
    assert_contingency_invariant(out.to_numpy(), 4)


@pytest.mark.parametrize("cfg,tag", [
    ({"num_classes": 4, "matching_mode": "bogus"}, "[CLM-101]"),
    ({"num_classes": 3, "balance": "unbalanced", "skew_rule": "bogus"}, "[CLM-107]"),
    ({"num_classes": 2, "balance": "balanced", "matching_mode": "custom",
      "assignment_matrix": [{"label": 0, "clusters": [0, 1], "recall_target": 0.5}],
      "split_rule": "bogus", "spillover_rule": "proportional_to_marginal"}, "[CLM-108]"),
    (base_custom_cfg(spillover_rule="bogus"), "[CLM-109]"),
    (base_custom_cfg(centroid_dependence={"enabled": True, "profile": "bogus"}), "[CLM-110]"),
], ids=["matching_mode", "skew_rule", "split_rule", "spillover_rule", "profile"])
def test_unknown_rule_names_are_rejected(cfg, tag):
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, cfg, seed=1)
    assert tag in str(excinfo.value)


def test_split_rule_is_unvalidated_when_every_rule_targets_one_cluster():
    """Characterisation: [CLM-108] is unreachable for single-cluster rules.

    `_split_row_allocation` short-circuits before consulting `split_rule` when
    a rule names exactly one cluster -- there is nothing to split -- so an
    invalid name is accepted silently. The test above proves the validation
    works when a rule spans two clusters; this pins the gap between them.

    Harmless today (the value is genuinely unused on this path) but it means a
    typo goes unreported in the common one-cluster-per-rule shape, and a user
    correcting a `split_rule` they believe is active would see no change.

    Asserted as it behaves NOW: if the validation is ever moved earlier this
    fails, which is the prompt to delete this test rather than a bug report.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(split_rule="bogus"), seed=1)
    assert_contingency_invariant(out.to_numpy(), 4)


# ---------------------------------------------------------------------------
# Exact boundaries
# ---------------------------------------------------------------------------

def test_single_label_space():
    """M=1: every point takes label 0. Degenerate, and must stay non-crashing."""
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 1, "balance": "balanced", "matching_mode": "random",
    }, seed=1)
    assert_contingency_invariant(out.to_numpy(), 1)


def test_allocation_exactly_saturating_a_cluster_succeeds():
    """tp == capacity is feasible; the failure starts one point later.

    Cluster 3 holds exactly 100 points and label 0's budget is exactly 100 at
    recall 1.0, so this is the tightest allocation that can still succeed. A
    `>=` where `>` belongs would reject it.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 2, "balance": "unbalanced", "proportions": [0.1, 0.9],
        "matching_mode": "custom",
        "assignment_matrix": [{"label": 0, "clusters": [3], "recall_target": 1.0}],
        "split_rule": "equal", "spillover_rule": "proportional_to_marginal",
    }, seed=1)
    assert_contingency_invariant(out.to_numpy(), 2)


def test_allocation_exceeding_capacity_by_one_point_is_infeasible():
    """The very next integer: 101 points demanded of a 100-point cluster."""
    with pytest.raises(InfeasibleAllocationError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 2, "balance": "unbalanced", "proportions": [0.101, 0.899],
            "matching_mode": "custom",
            "assignment_matrix": [{"label": 0, "clusters": [3], "recall_target": 1.0}],
            "split_rule": "equal", "spillover_rule": "proportional_to_marginal",
        }, seed=1)
    assert "[CLM-150]" in str(excinfo.value)


@pytest.mark.parametrize("value", [0.0, 1.0], ids=["independence", "ceiling"])
def test_solver_grid_endpoints_are_evaluable(value):
    """The coarse scan is `linspace(0, 1, 11)`, so both endpoints are probed.

    Neither is expected to be *reached* -- 1.0 sits above the structural
    ceiling for this geometry, which is what [CLM-306] reports -- but both must
    be evaluable rather than crashing the bracket search.
    """
    out = generate_clm_labels(CLUSTERS, COORDS, base_custom_cfg(
        target_metric={"type": "mcc", "value": value, "tolerance": 0.05, "max_iter": 20}), seed=1)
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
            {"num_classes": 5, "balance": "unbalanced", "skew_rule": "dirichlet",
             "skew_params": {"alpha": alpha}}, N, np.random.default_rng(seed))
        assert sum(counts) == N, "dirichlet draw did not partition N"
        return max(counts) - min(counts)

    for seed in range(5):
        concentrated = spread(0.1, seed)
        near_uniform = spread(100.0, seed)
        assert concentrated > near_uniform, (
            "alpha=0.1 should be more lopsided than alpha=100 "
            f"(seed {seed}: spread {concentrated} vs {near_uniform})")


def test_dirichlet_is_stochastic_across_seeds():
    """Reproducible per seed, different between seeds.

    `00_contract` asserts the first half for every rule. The second half is
    specific to dirichlet: it is the only skew rule that draws, so it is the
    only one where two seeds are expected to disagree. `geometric` and
    `dominant_minority` are deterministic functions of their parameters.
    """
    def counts(seed):
        return list(resolve_label_counts(
            {"num_classes": 5, "balance": "unbalanced", "skew_rule": "dirichlet",
             "skew_params": {"alpha": 0.5}}, N, np.random.default_rng(seed)))

    assert counts(1) == counts(1)
    assert counts(1) != counts(2)


def test_resolve_label_counts_at_zero_rows():
    """N=0: the largest-remainder split has nothing to distribute."""
    counts = resolve_label_counts({"num_classes": 3, "balance": "balanced"}, 0,
                                  np.random.default_rng(0))
    assert list(counts) == [0, 0, 0], list(counts)


# ---------------------------------------------------------------------------
# Data-source degradation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"dataset_name": "does_not_exist"},
    {"dataset_name": None},
], ids=["missing-file", "no-name"])
def test_byoc_returns_none_rather_than_raising(kwargs):
    """The source contract: unusable input is None (logged), never an exception.

    `run_pipeline` treats None as "skip this dataset"; a raised
    FileNotFoundError would instead abort the batch.
    """
    assert fetch_byoc_data(cluster_column="c", **kwargs) is None
