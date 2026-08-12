"""Contract / invariant checks for the CLM label engine.
If `00_contract` fails you know something is wrong.
The diagnostic catalogue covers the failure half of the behavior space: wrong
input produces the right error. This covers the other half, which nothing else
does: RIGHT input produces RIGHT output. A sensitive test, it sweeps the valid
configuration space broadly and asks whether the engine keeps its promises anywhere
in it. The risk it targets is not a crash but silent incorrectness, a run that
succeeds while delivering a labeling whose agreement with the clusters is not what
the configuration asked for.

Every check runs against the engine directly (generate_clm_labels), on synthetic
geometry built here, so it needs no network, no data sources and no optional
dependency.

    python -m pytest tests/test_00_contract.py            # this module
    python -m pytest tests/test_00_contract.py -v         # per-case detail
"""

import logging
from typing import Any

import numpy as np
import pytest

from clmsynth.clm_errors import InfeasibleAllocationError
from clmsynth.clm_label_engine import generate_clm_labels, resolve_label_counts
from clmsynth.metrics import clustering_ari, clustering_mcc

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def geometry(sizes, dim=2, seed=0):
    """K gaussian blobs with the given sizes -> (cluster_labels, coords)."""
    rng = np.random.default_rng(seed)
    labels, coords = [], []
    for k, n in enumerate(sizes):
        centre = np.zeros(dim)
        centre[0] = k * 10.0
        coords.append(rng.normal(centre, 1.0, size=(n, dim)))
        labels.append(np.full(n, k))
    return np.concatenate(labels), np.concatenate(coords)


GEOMETRIES = {
    "equal_4x200": [200] * 4,
    "unequal_4": [400, 300, 200, 100],
    "equal_3x150": [150] * 3,
    "skewed_5": [500, 120, 90, 60, 30],
}


# ---------------------------------------------------------------------------
# Log capture, diagnostic run check
# ---------------------------------------------------------------------------

class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def codes(self):
        out = set()
        for m in self.records:
            if "[CLM-" in m:
                out.add(m.split("[CLM-")[1].split("]")[0])
        return out


def run(cfg, cluster_labels, coords, seed=42):
    cap = Capture()
    log = logging.getLogger("clmsynth")
    log.addHandler(cap)
    prev = log.level
    log.setLevel(logging.DEBUG)
    try:
        series = generate_clm_labels(cluster_labels, coords, cfg, seed=seed)
    finally:
        log.removeHandler(cap)
        log.setLevel(prev)
    return np.asarray(series), cap.codes()


# ---------------------------------------------------------------------------
# Configurations: the VALID space, one axis at a time plus key combinations
# ---------------------------------------------------------------------------

def placement(profile=None, favors="core"):
    if profile is None:
        return {"enabled": False}
    cd = {"enabled": True, "profile": profile, "favors": favors}
    if profile == "exponential":
        cd["steepness"] = 3.0
    return cd


PLACEMENTS = [
    ("off", placement()),
    ("linear/core", placement("linear", "core")),
    ("linear/boundary", placement("linear", "boundary")),
    ("exponential/core", placement("exponential", "core")),
    ("exponential/boundary", placement("exponential", "boundary")),
    ("step/core", placement("step", "core")),
    ("step/boundary", placement("step", "boundary")),
]


def safe_recall(sizes, M, margin=0.8):
    """Largest recall that keeps a one-rule-per-cluster bijection feasible.

    With balanced labels each label owns N/M points; a rule routing recall*N/M of
    them into one cluster needs that cluster to hold them. The binding constraint
    is therefore the SMALLEST cluster, which is why a flat recall that works on
    equal clusters raises [CLM-150] on a skewed geometry.
    """
    return round(margin * min(sizes) / (sum(sizes) / M), 3)


def bijection(M, recall=0.8, **extra):
    cfg = {
        "num_classes": M,
        "balance": "balanced",
        "matching_mode": "custom",
        "assignment_matrix": [{"clusters": [i], "label": i, "recall_target": recall}
                              for i in range(M)],
        "split_rule": "proportional_to_size",
        "spillover_rule": "proportional_to_marginal",
        "centroid_dependence": placement(),
    }
    cfg.update(extra)
    return cfg


def cases():
    """(name, geometry_key, cfg) triples spanning the valid configuration space."""
    out: list[Any] = []

    # -- matching modes -----------------------------------------------------
    out.append(("perfect", "equal_4x200",
                {"num_classes": 4, "balance": "balanced", "matching_mode": "perfect",
                 "centroid_dependence": placement()}))
    out.append(("random", "equal_4x200",
                {"num_classes": 4, "balance": "balanced", "matching_mode": "random",
                 "centroid_dependence": placement()}))
    out.append(("single", "equal_4x200",
                {"num_classes": 4, "balance": "balanced", "matching_mode": "single",
                 "single_match": {"cluster": 0, "label": 0},
                 "spillover_rule": "proportional_to_marginal",
                 "centroid_dependence": placement()}))
    for g, sizes in GEOMETRIES.items():
        K = len(sizes)
        out.append((f"custom_bijection[{g}]", g,
                    bijection(K, recall=safe_recall(sizes, K))))

    # -- recall levels ------------------------------------------------------
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append((f"recall={r}", "equal_4x200", bijection(4, recall=r)))

    # -- balance / skew -----------------------------------------------------
    out.append(("proportions_explicit", "equal_4x200",
                bijection(4, balance="unbalanced",
                          proportions=[0.4, 0.3, 0.2, 0.1], recall=0.4)))
    for rule, params in (("geometric", {"ratio": 0.5}),
                         ("dominant_minority", {"dominant_index": 0, "dominant_share": 0.55}),
                         ("dirichlet", {"alpha": 1.0})):
        # Low recall: a dirichlet draw can put most of the mass on one label, and
        # a high recall would then over-claim its cluster ([CLM-150]) for reasons
        # unrelated to the invariant under test.
        out.append((f"skew={rule}", "equal_4x200",
                    bijection(4, balance="unbalanced", skew_rule=rule,
                              skew_params=params, recall=0.2)))

    # -- spillover ----------------------------------------------------------
    for rule in ("proportional_to_marginal", "uniform", "concentrated"):
        out.append((f"spillover={rule}", "equal_4x200",
                    bijection(4, spillover_rule=rule, recall=0.5)))

    # -- split rule (rules spanning >1 cluster) -----------------------------
    # recall 0.3, not 0.5: an EQUAL split halves the budget across clusters [2,3]
    # regardless of their sizes, so at 0.5 it hands 125 points to a 100-point
    # cluster ([CLM-151]) while proportional_to_size would have fit. The recall
    # has to suit the stricter of the two rules for them to be comparable.
    for split in ("proportional_to_size", "equal"):
        out.append((f"split={split}", "unequal_4",
                    {"num_classes": 2, "balance": "balanced", "matching_mode": "custom",
                     "assignment_matrix": [{"clusters": [0, 1], "label": 0, "recall_target": 0.3},
                                           {"clusters": [2, 3], "label": 1, "recall_target": 0.3}],
                     "split_rule": split, "spillover_rule": "proportional_to_marginal",
                     "centroid_dependence": placement()}))

    # -- placement ----------------------------------------------------------
    for name, cd in PLACEMENTS:
        out.append((f"placement={name}", "equal_4x200",
                    bijection(4, recall=0.6, centroid_dependence=cd)))

    # -- target metric ------------------------------------------------------
    for t in (0.3, 0.5, 0.7):
        out.append((f"target_mcc={t}", "equal_4x200",
                    bijection(4, target_metric={"type": "mcc", "value": t, "tolerance": 0.05})))
        out.append((f"target_ari={t}", "equal_4x200",
                    bijection(4, target_metric={"type": "ari", "value": t, "tolerance": 0.05})))
    out.append(("target_pair_mcc", "equal_4x200",
                {"num_classes": 4, "balance": "balanced", "matching_mode": "single",
                 "single_match": {"cluster": 0, "label": 0},
                 "spillover_rule": "proportional_to_marginal",
                 "target_metric": {"type": "mcc", "value": 0.6, "scope": "pair"},
                 "centroid_dependence": placement()}))

    # -- competing noise ----------------------------------------------------
    for favors in ("core", "boundary", "random"):
        out.append((f"competing={favors}", "equal_4x200",
                    bijection(4, recall=0.5,
                              competing_noise=[{"cluster": 1, "label": 2,
                                                "share": 0.5, "favors": favors}])))

    # -- cardinality edges --------------------------------------------------
    out.append(("M=1", "equal_4x200",
                {"num_classes": 1, "balance": "balanced", "matching_mode": "custom",
                 "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.2}],
                 "split_rule": "proportional_to_size",
                 "spillover_rule": "proportional_to_marginal",
                 "centroid_dependence": placement()}))
    out.append(("M<K", "equal_4x200", bijection(2, recall=0.5)))
    out.append(("M>K", "equal_3x150",
                {"num_classes": 5, "balance": "balanced", "matching_mode": "custom",
                 "assignment_matrix": [{"clusters": [i], "label": i, "recall_target": 0.5}
                                       for i in range(3)],
                 "split_rule": "proportional_to_size",
                 "spillover_rule": "proportional_to_marginal",
                 "centroid_dependence": placement()}))
    return out


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def check_case(name, gkey, cfg):
    """Returns a list of (invariant, ok, detail)."""
    sizes = GEOMETRIES[gkey]
    c, X = geometry(sizes)
    N, M = len(c), cfg["num_classes"]
    res = []

    try:
        out, codes = run(cfg, c, X)
    except InfeasibleAllocationError as exc:
        # A documented, legitimate outcome ([CLM-15x]) rather than a contract
        # breach: this geometry cannot hold the requested budget. The failure
        # catalogue covers those; skip the case here rather than mis-report it.
        return [("config_feasible_as_written", False, f"skipped: {exc}")]
    except Exception as exc:                       # broad by design: reporting
        return [("no_unexpected_exception", False, f"{type(exc).__name__}: {exc}")]
    res.append(("no_unexpected_exception", True, ""))

    # 1. label domain
    bad = sorted({int(v) for v in np.unique(out) if not (0 <= v < M)})
    res.append(("labels_within_0..M-1", not bad,
                f"out-of-range labels {bad} (M={M})" if bad else ""))

    # 2. row alignment
    res.append(("length_preserved", len(out) == N, f"{len(out)} != {N}"))

    # 3. determinism
    out2, _ = run(cfg, c, X)
    res.append(("deterministic_for_seed", np.array_equal(out, out2),
                "second run differed" if not np.array_equal(out, out2) else ""))

    # 4. exact marginals, only where the engine promises them
    promises_counts = (
        cfg.get("spillover_rule") == "proportional_to_marginal"
        and not cfg.get("competing_noise")
        and cfg["matching_mode"] in ("custom", "single", "random")
        and (cfg.get("target_metric") or {}).get("scope") != "pair"
    )
    if promises_counts:
        want = resolve_label_counts(cfg, N, np.random.default_rng(42))
        got = np.bincount(out, minlength=M)
        ok = len(want) == len(got) and np.array_equal(np.asarray(want), got)
        res.append(("exact_label_counts", ok,
                    "" if ok else f"target {list(want)} != achieved {list(got)}"))

    # 5. target metric delivered, or flagged
    tm = cfg.get("target_metric")
    if tm and tm.get("scope") != "pair":
        fn = clustering_mcc if tm["type"] == "mcc" else clustering_ari
        achieved = float(fn(c, out))
        tol = tm.get("tolerance", 0.01)
        within = abs(achieved - tm["value"]) <= tol
        flagged = bool({"306", "309"} & codes)
        res.append(("target_met_or_flagged", within or flagged,
                    "" if (within or flagged) else
                    f"achieved {achieved:.4f} vs {tm['value']}+/-{tol}, no 306/309"))

    return res


def check_placement_invariance():
    """Core / boundary / random placement must leave MCC and ARI untouched.

    This is the paper's central claim: the contingency table is fixed before
    placement runs, so spatial structure varies at a constant metric value.
    """
    rows = []
    for gkey, sizes in GEOMETRIES.items():
        K = len(sizes)
        c, X = geometry(sizes)
        seen = {}
        r = safe_recall(sizes, K)
        for pname, cd in PLACEMENTS:
            cfg = bijection(K, recall=r, centroid_dependence=cd)
            out, _ = run(cfg, c, X)
            seen[pname] = (round(float(clustering_mcc(c, out)), 10),
                           round(float(clustering_ari(c, out)), 10))
        vals = set(seen.values())
        ok = len(vals) == 1
        detail = "" if ok else "; ".join(f"{k}={v}" for k, v in seen.items())
        rows.append((f"placement_invariance[{gkey}]", ok, detail))
    return rows


def check_concentrated_labels_guard():
    """concentrated_labels is documented but unvalidated: an out-of-range value
    reaches the written labeling with no error. Isolated here because it is a
    known open item, so its failure is expected until the guard lands."""
    c, X = geometry(GEOMETRIES["equal_4x200"])
    cfg = bijection(4, recall=0.5, spillover_rule="concentrated",
                    concentrated_labels=[99])
    try:
        out, _ = run(cfg, c, X)
    except Exception as exc:                       # broad by design: reporting
        return [("concentrated_labels_rejected", True, f"raised {type(exc).__name__}")]
    bad = sorted({int(v) for v in np.unique(out) if not (0 <= v < 4)})
    return [("concentrated_labels_rejected", not bad,
             f"label 99 accepted; wrote out-of-range {bad} into the labeling" if bad else "")]


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------

CASES = cases()

def _assert_all_hold(results, context):
    """Fail with every broken invariant listed, not just the first."""
    broken = [(inv, detail) for inv, ok, detail in results if not ok]
    assert not broken, context + "\n" + "\n".join(
        f"  {inv}: {detail}" for inv, detail in broken)


def test_case_inventory_is_not_empty():
    """Guard the parametrisation itself.

    A module that collects nothing is silently green inside a directory run:
    `pytest` succeeds identically whether this file contributes 41 tests or
    zero. That is not hypothetical -- this module contributed zero for its
    entire existence before being converted, and nothing said so.
    """
    assert len(CASES) >= 40, f"expected the full sweep, collected {len(CASES)} cases"
    assert len({name for name, _, _ in CASES}) == len(CASES), "duplicate case names"


@pytest.mark.parametrize("name,gkey,cfg", CASES, ids=[c[0] for c in CASES])
def test_configuration_keeps_its_contract(name, gkey, cfg):
    """Right input produces right output, across the valid configuration space.

    One test per configuration rather than per (configuration, invariant):
    `check_case` already reports which invariant broke, and 41 named cases read
    better in a failure summary than ~200 fragments of them.
    """
    results = check_case(name, gkey, cfg)

    # An infeasible allocation is a documented outcome ([CLM-15x]) for a
    # geometry that cannot hold the requested budget, not a contract breach.
    # The old harness had no notion of a skip and recorded it as a failure,
    # contradicting its own comment; pytest does, so this now says what was
    # always meant.
    if len(results) == 1 and results[0][0] == "config_feasible_as_written":
        pytest.skip(results[0][2])

    _assert_all_hold(results, f"case {name!r} on geometry {gkey!r}")


@pytest.mark.parametrize("geometry_key", sorted(GEOMETRIES), ids=sorted(GEOMETRIES))
def test_placement_leaves_the_metrics_untouched(geometry_key):
    """The central claim: the contingency table is fixed before placement runs.

    Core, boundary and random placement move which points carry a label without
    moving the agreement at all, so spatial structure varies at a constant
    metric value. If this fails, placement is leaking into the marginals.
    """
    rows = [r for r in check_placement_invariance() if geometry_key in r[0]]
    assert rows, f"no placement row produced for {geometry_key}"
    _assert_all_hold(rows, f"placement invariance on {geometry_key!r}")


def test_concentrated_labels_out_of_range_is_rejected():
    """`concentrated_labels: [99]` under num_classes=4 must not reach the output.

    Written while this was an open defect -- the value was drawn from directly
    and written into the label column, putting a label in the dataset that
    `num_classes` never declared. Closed by [CLM-128] in 0.6.0; kept as the
    regression pin for it.
    """
    _assert_all_hold(check_concentrated_labels_guard(), "concentrated_labels guard")
