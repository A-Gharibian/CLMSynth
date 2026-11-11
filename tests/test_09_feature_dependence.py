"""Category 9: Labels that follow a feature.

A `feature_dependence` entry moves a label's placed points towards one end of a
feature, or towards a value of it, inside the clusters the label is placed in. It
picks which points carry the label, never how many. What is checked, in order:

  placement      `step` takes the favored end exactly and `linear` leans towards
                 it; only placed points move, in every cluster the label is placed
                 in; entries go in the order written; the counts, the scores and a
                 target's level are those of the same config without the block.
  refusals       A malformed block is refused before any dataset and again at
                 labelling: a missing key is [CLM-207], a label [CLM-104], anything
                 else [CLM-141], as is an entry beside an enabled
                 centroid_dependence. A dataset without the column is [CLM-125].
  warnings       An entry that moves nothing is [CLM-305], for the delivered labels
                 only; under `random` the block is not read, [CLM-311].
  a run          A dataset without the column is skipped and the run goes on; a
                 malformed entry stops the run before any fetch.
  saved config   The resolved config records the entries with their defaults
                 filled in, leaves the block out under `random`, and repeats the
                 run.
  shown          The plot panel and clmsynth-config carry the entries.
  the wizard     Asks for entries under `single` and `custom` only, and not after
                 centroid_dependence was enabled.

Every test goes through a `DatasetContext` or a run, which have the column names.
No network; the clmsynth-config cases need PyYAML and skip without it.
"""

import json
import logging
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from clmsynth import (
    MissingConfigKey,
    build_context,
    clustering_ari,
    clustering_mcc,
    clustering_mcc_pair,
    generate_additional_labels,
    generate_clm_labels,
    run_pipeline,
)
from clmsynth._config_gen import config_wizard as wizard
from clmsynth._config_gen.generate_config import generate_base_config
from clmsynth._config_gen.questions import SCHEMA
from clmsynth._extras.visualization import _clm_info_text
from clmsynth._single_run import _exit_status
from clmsynth.clm_errors import CODES
from clmsynth.config_tree import SETTINGS
from clmsynth.labeling import LabelGenerator
from clmsynth.resolved_config import resolve_config

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def blobs(per_cluster=200, clusters=2, seed=0, features=("f1", "f2", "f3")):
    """Noise features and integer cluster ids, in the frame a fetcher returns."""
    rng = np.random.default_rng(seed)
    n = per_cluster * clusters
    frame = pd.DataFrame({name: rng.normal(size=n) for name in features})
    frame["GroundTruth_labels0"] = np.repeat(np.arange(clusters), per_cluster)
    return frame


def custom(M=2, recall=0.8, **extra):
    """label i on cluster i at `recall`, balanced, for M labels and M clusters."""
    return {"num_classes": M, "matching_mode": "custom", "balance": "balanced",
            "assignment_matrix": [{"clusters": [k], "label": k, "recall_target": recall}
                                  for k in range(M)], **extra}


def entry(label=0, feature="f1", favors="high", profile="step", **extra):
    """One feature_dependence entry."""
    return {"label": label, "feature": feature, "favors": favors, "profile": profile, **extra}


def label_frame(frame, clm, seed=1, n_labels=1):
    """The context of `frame` with `n_labels` columns generated under `clm`."""
    context = build_context("unit", "battery", "dataset", frame)
    generate_additional_labels(context, n_labels=n_labels, clm_config=clm, seed=seed)
    return context


def column(context, i=0):
    """Label_i as an array."""
    return context.generated_labels[f"Label_{i}"].to_numpy()


def in_cluster(frame, cluster):
    return frame["GroundTruth_labels0"].to_numpy() == cluster


def ranked(frame, cluster, feature, k, high=True):
    """Row positions of the k highest (or lowest) values of `feature` in `cluster`."""
    where = np.flatnonzero(in_cluster(frame, cluster))
    values = frame[feature].to_numpy()[where]
    order = np.argsort(-values if high else values)
    return where[order[:k]]


def closest(frame, cluster, feature, value, k):
    """Row positions of the k values of `feature` in `cluster` closest to `value`."""
    where = np.flatnonzero(in_cluster(frame, cluster))
    return where[np.argsort(np.abs(frame[feature].to_numpy()[where] - value))[:k]]


def following(frame, labels, cluster, feature, label):
    """The chance that a point of `label` sits above another point of `cluster`
    on `feature`: 1.0 always, 0.5 no relation."""
    keep = in_cluster(frame, cluster)
    values, held = frame[feature].to_numpy()[keep], labels[keep] == label
    n1, n0 = int(held.sum()), int((~held).sum())
    return float((rankdata(values)[held].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def warned(caplog, *words):
    """Warning records whose message holds every word."""
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and all(w in r.getMessage() for w in words)]


def csv_frame(frame):
    """A byoc CSV's columns: the features and a `cluster` column."""
    return frame.rename(columns={"GroundTruth_labels0": "cluster"})


def byoc_config(folder, datasets, clm, n_labels=1, seed=1, **suite):
    """A byoc run over CSVs in `folder`."""
    return {
        "global_settings": {"data_source": "byoc", "output_dir": str(folder / "OUT")},
        "byoc_suite": {"batteries": ["local"], "datasets": list(datasets),
                       "input_dir": str(folder), "cluster_column": "cluster", **suite},
        "label_generation": {"n_labels": n_labels, "seed": seed, "clm_label": clm},
    }


def fabricated_config(clm, datasets=("baseline_4class",), n_labels=1, seed=1):
    """A fabricated_data run: 800 points, four clusters of 200, Feature_1..6."""
    return {
        "global_settings": {"data_source": "fabricated_data", "output_dir": "OUT"},
        "fabricated_data_suite": {"batteries": ["fabricated"], "datasets": list(datasets),
                                  "seed": 7},
        "label_generation": {"n_labels": n_labels, "seed": seed, "clm_label": clm},
    }


def run(source, config, folder):
    """run_pipeline into `folder`: (datasets written, its report, csv dir, txt dir,
    the resolved config's path)."""
    dirs = [folder / sub for sub in ("csv", "png", "txt")]
    for d in dirs:
        d.mkdir(parents=True)
    report: dict[str, Any] = {}
    resolved = folder / "resolved.json"
    n = run_pipeline(source, config, dirs[0], dirs[1], dirs[2], render_plots=False,
                     config_out=resolved, report=report)
    return n, report, dirs[0], dirs[2], resolved


def drive(monkeypatch, answers):
    """build_clm with ask_from answering from `answers`, an iterator per key for a
    key asked more than once. Returns (clm, the keys asked, their overrides)."""
    asked: list[str] = []
    overrides: dict[str, list[dict[str, Any]]] = {}
    base = {"clm_label.num_classes": 3, "clm_label.balance": "balanced",
            "clm_label.assignment_matrix.clusters": [0],
            "clm_label.single_match.cluster": 0,
            "clm_label.competing_noise.cluster": 1,
            "clm_label.feature_dependence.feature": "f1"}

    def fake(key, **override):
        asked.append(key)
        overrides.setdefault(key, []).append(override)
        value = {**base, **answers}.get(key, SCHEMA[key].default)
        return next(value) if hasattr(value, "__next__") else value

    monkeypatch.setattr(wizard, "ask_from", fake)
    return wizard.build_clm(), asked, overrides


# ---------------------------------------------------------------------------
# Placement: where the points go
# ---------------------------------------------------------------------------

# Two clusters of 200, label i on cluster i at recall 0.8: 160 placed points each.
PLACED = 160


def test_high_step_takes_the_highest_values():
    frame = blobs()
    labels = column(label_frame(frame, custom(feature_dependence=[entry()])))
    assert (labels[ranked(frame, 0, "f1", PLACED)] == 0).all()


def test_low_step_takes_the_lowest_values():
    frame = blobs()
    labels = column(label_frame(frame, custom(feature_dependence=[entry(favors="low")])))
    assert (labels[ranked(frame, 0, "f1", PLACED, high=False)] == 0).all()


def test_an_entry_without_favors_takes_high():
    """An entry needs a label and a feature; favors defaults to high."""
    frame = blobs()
    plain = {"label": 0, "feature": "f1", "profile": "step"}
    labels = column(label_frame(frame, custom(feature_dependence=[plain])))
    assert (labels[ranked(frame, 0, "f1", PLACED)] == 0).all()


@pytest.mark.parametrize("value", [-0.5, 0.0, 0.75])
def test_value_step_takes_the_closest_values(value):
    frame = blobs()
    clm = custom(feature_dependence=[entry(favors="value", value=value)])
    labels = column(label_frame(frame, clm))
    assert (labels[closest(frame, 0, "f1", value, PLACED)] == 0).all()


def test_a_value_beyond_the_range_acts_as_high():
    frame = blobs()
    far = column(label_frame(frame, custom(
        feature_dependence=[entry(favors="value", value=1e6)])))
    assert (far[ranked(frame, 0, "f1", PLACED)] == 0).all()


def test_a_value_below_the_range_acts_as_low():
    frame = blobs()
    far = column(label_frame(frame, custom(
        feature_dependence=[entry(favors="value", value=-1e6)])))
    assert (far[ranked(frame, 0, "f1", PLACED, high=False)] == 0).all()


def test_an_absent_profile_is_linear_and_leans_towards_the_favored_end():
    """Weaker than step, so measured over seeds."""
    frame = blobs()
    plain_entry = {"label": 0, "feature": "f1", "favors": "high"}
    scores = [following(frame, column(label_frame(frame, custom(
        feature_dependence=[plain_entry]), seed=s)), 0, "f1", 0) for s in range(12)]
    assert np.mean(scores) > 0.58


def test_step_beats_linear():
    frame = blobs()
    def mean_following(profile):
        return np.mean([following(frame, column(label_frame(frame, custom(
            feature_dependence=[entry(profile=profile)]), seed=s)), 0, "f1", 0)
            for s in range(8)])
    assert mean_following("step") > mean_following("linear") + 0.05
    assert mean_following("step") > 0.8


def test_the_feature_is_found_by_name_not_by_position():
    frame = blobs()
    reordered = frame[["f3", "f1", "f2", "GroundTruth_labels0"]]
    labels = column(label_frame(reordered, custom(feature_dependence=[entry()])))
    assert (labels[ranked(reordered, 0, "f1", PLACED)] == 0).all()


def test_an_entry_applies_in_every_cluster_the_label_is_placed_in():
    frame = blobs()
    clm = {"num_classes": 2, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0, 1], "label": 0, "recall_target": 0.8}],
           "feature_dependence": [entry()]}
    labels = column(label_frame(frame, clm))
    # 200 points of label 0 at recall 0.8, split 80 + 80.
    assert (labels[ranked(frame, 0, "f1", 80)] == 0).all()
    assert (labels[ranked(frame, 1, "f1", 80)] == 0).all()


def test_only_placed_points_move_and_spillover_stays_random():
    """Label 0 in cluster 1 is spillover only, so it does not follow f1 there."""
    frame = blobs(per_cluster=500)
    clm = custom(recall=0.5, feature_dependence=[entry()])
    scores = [following(frame, column(label_frame(frame, clm, seed=s)), 1, "f1", 0)
              for s in range(6)]
    labels = column(label_frame(frame, clm))
    assert (labels[ranked(frame, 0, "f1", 250)] == 0).all()
    assert 0.42 < np.mean(scores) < 0.58


def test_an_entry_goes_before_a_bigger_label():
    """A bigger label placed first at random would take the favored points."""
    frame = blobs(per_cluster=300)
    clm = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.5},
                                 {"clusters": [1], "label": 1, "recall_target": 0.5},
                                 {"clusters": [0, 1], "label": 2, "recall_target": 0.8}],
           "feature_dependence": [entry(label=2)]}
    labels = column(label_frame(frame, clm))
    # Label 2 asks 160, 80 per cluster; label 0 asks 100 in cluster 0 and is bigger.
    assert (labels[ranked(frame, 0, "f1", 80)] == 2).all()


SHARED = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
          "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.5},
                                {"clusters": [0], "label": 2, "recall_target": 0.25}]}


def test_the_list_order_settles_who_goes_first():
    """Labels of 100 and 50 points in one cluster: the first entry takes the top."""
    frame = blobs(per_cluster=300)
    first_big = {**SHARED, "feature_dependence": [entry(label=0), entry(label=2)]}
    first_small = {**SHARED, "feature_dependence": [entry(label=2), entry(label=0)]}
    labels = column(label_frame(frame, first_big))
    assert (labels[ranked(frame, 0, "f1", 100)] == 0).all()
    assert (labels[ranked(frame, 0, "f1", 150)[100:]] == 2).all()
    labels = column(label_frame(frame, first_small))
    assert (labels[ranked(frame, 0, "f1", 50)] == 2).all()
    assert (labels[ranked(frame, 0, "f1", 150)[50:]] == 0).all()


def test_two_labels_can_follow_two_features():
    frame = blobs()
    labels = column(label_frame(frame, custom(
        feature_dependence=[entry(label=0, feature="f1"), entry(label=1, feature="f2")])))
    assert (labels[ranked(frame, 0, "f1", PLACED)] == 0).all()
    assert (labels[ranked(frame, 1, "f2", PLACED)] == 1).all()


def test_two_labels_can_take_the_two_ends_of_one_cluster():
    frame = blobs(per_cluster=300)
    clm = {**SHARED, "feature_dependence": [entry(label=0), entry(label=2, favors="low")]}
    labels = column(label_frame(frame, clm))
    assert (labels[ranked(frame, 0, "f1", 100)] == 0).all()
    assert (labels[ranked(frame, 0, "f1", 50, high=False)] == 2).all()


def test_an_integer_feature_with_ties_works():
    frame = blobs()
    frame["f1"] = np.arange(len(frame)) % 2
    labels = column(label_frame(frame, custom(recall=0.4, feature_dependence=[entry()])))
    keep = in_cluster(frame, 0)
    # 80 placed points of the 100 at value 1; spillover adds about half of the rest.
    assert (labels[keep & (frame["f1"].to_numpy() == 1)] == 0).mean() >= 0.8


def test_cluster_ids_need_not_be_contiguous():
    frame = blobs()
    frame["GroundTruth_labels0"] = frame["GroundTruth_labels0"].map({0: 5, 1: 9})
    clm = {"num_classes": 2, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [5], "label": 0, "recall_target": 0.8},
                                 {"clusters": [9], "label": 1, "recall_target": 0.8}],
           "feature_dependence": [entry()]}
    labels = column(label_frame(frame, clm))
    assert (labels[ranked(frame, 5, "f1", PLACED)] == 0).all()


def test_single_match_label_follows():
    frame = blobs()
    clm = {"num_classes": 2, "matching_mode": "single", "balance": "unbalanced",
           "proportions": [0.25, 0.75], "single_match": {"cluster": 0, "label": 0},
           "feature_dependence": [entry()]}
    labels = column(label_frame(frame, clm))
    assert (labels[ranked(frame, 0, "f1", 100)] == 0).all()


@pytest.mark.parametrize("spillover", [{"spillover_rule": "uniform"},
                                       {"spillover_rule": "concentrated",
                                        "concentrated_labels": [1]}],
                         ids=["uniform", "concentrated"])
def test_an_entry_runs_under_any_spillover_rule(spillover):
    """centroid_dependence is the only rule between settings. Spillover goes to label 1
    under `concentrated`, or cluster 0 would hold nothing but label 0."""
    frame = blobs()
    labels = column(label_frame(frame, custom(**spillover, feature_dependence=[entry()])))
    assert (labels[ranked(frame, 0, "f1", PLACED)] == 0).all()


@pytest.mark.parametrize("centroid", [False, True], ids=["plain", "centroid"])
@pytest.mark.parametrize("empty", [[], None], ids=["list", "none"])
def test_an_empty_feature_dependence_is_off(empty, centroid):
    """No entries: the labels of the same config without the block, also with
    centroid_dependence enabled, which an entry would refuse ([CLM-141])."""
    frame = blobs()
    extra = {"centroid_dependence": {"enabled": True, "favors": "core"}} if centroid else {}
    without = column(label_frame(frame, custom(**extra)))
    assert (column(label_frame(frame, custom(**extra, feature_dependence=empty))) == without).all()


def test_noise_takes_what_the_entries_leave():
    """Inside a cluster: entries, other targets, competing noise, spillover."""
    frame = blobs()
    clm = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.5}],
           "competing_noise": [{"cluster": 0, "label": 1, "share": 0.5, "favors": "core"}],
           "feature_dependence": [entry()]}
    context = label_frame(frame, clm)
    placed = context.label_reports["Label_0"]["assignment_matrix"][0]["placed"]
    assert (column(context)[ranked(frame, 0, "f1", placed)] == 0).all()


def test_a_label_keeps_its_noise_points_while_its_placed_points_move():
    """Label 2 has a row in cluster 0 and a noise entry in cluster 1: the entry moves
    the row's points and leaves the noise where competing_noise put it."""
    frame = blobs(per_cluster=300)
    clm = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 2, "recall_target": 0.5},
                                 {"clusters": [1], "label": 1, "recall_target": 0.5}],
           "competing_noise": [{"cluster": 1, "label": 2, "share": 0.3, "favors": "random"}],
           "feature_dependence": [entry(label=2)]}
    context = label_frame(frame, clm)
    labels = column(context)
    assert (labels[ranked(frame, 0, "f1", 100)] == 2).all()
    assert context.label_reports["Label_0"]["competing_noise"][0]["placed"] == 60
    assert abs(following(frame, labels, 1, "f1", 2) - 0.5) < 0.15


def test_the_counts_and_the_scores_do_not_move():
    """Placement picks which points carry a label, never how many. The move itself is
    checked too, so the two runs are not the same run."""
    frame = blobs()
    truth = frame["GroundTruth_labels0"].to_numpy()
    plain = column(label_frame(frame, custom()))
    moved = column(label_frame(frame, custom(feature_dependence=[entry()])))
    assert (moved[ranked(frame, 0, "f1", PLACED)] == 0).all()
    assert (pd.crosstab(truth, plain).to_numpy() == pd.crosstab(truth, moved).to_numpy()).all()
    assert clustering_mcc(truth, plain) == clustering_mcc(truth, moved)
    assert clustering_ari(truth, plain) == clustering_ari(truth, moved)


@pytest.mark.parametrize("kind", ["mcc", "ari"])
@pytest.mark.parametrize("value", [0.9, 0.5, 0.1])
def test_a_global_target_solves_to_the_same_level(kind, value):
    frame = blobs()
    target = {"type": kind, "value": value}
    plain = label_frame(frame, {**custom(), "target_metric": target})
    moved = label_frame(frame, {**custom(), "target_metric": target,
                                "feature_dependence": [entry()]})
    assert (plain.label_reports["Label_0"]["alpha"]
            == moved.label_reports["Label_0"]["alpha"])
    assert (plain.label_reports["Label_0"]["achieved"]
            == moved.label_reports["Label_0"]["achieved"])
    placed = moved.label_reports["Label_0"]["assignment_matrix"][0]["placed"]
    assert (column(moved)[ranked(frame, 0, "f1", placed)] == 0).all()


def test_a_pair_target_still_lands_and_its_label_follows():
    frame = blobs()
    clm = {"num_classes": 2, "matching_mode": "single", "balance": "balanced",
           "single_match": {"cluster": 0, "label": 0},
           "target_metric": {"type": "mcc", "value": 0.7, "scope": "pair"},
           "feature_dependence": [entry()]}
    labels = column(label_frame(frame, clm))
    truth = frame["GroundTruth_labels0"].to_numpy()
    assert abs(clustering_mcc_pair(truth, labels, 0, 0) - 0.7) <= 0.01
    size = int((labels == 0).sum())
    assert (labels[ranked(frame, 0, "f1", size)] == 0).all()


def test_the_same_seed_gives_the_same_labels():
    frame = blobs()
    clm = custom(feature_dependence=[entry(profile="linear")])
    plain = column(label_frame(frame, custom()))
    first, second = column(label_frame(frame, clm)), column(label_frame(frame, clm))
    assert (first == second).all()
    assert (first != plain).any()


def test_each_column_of_a_run_follows():
    frame = blobs()
    context = label_frame(frame, custom(feature_dependence=[entry()]), n_labels=3)
    for i in range(3):
        assert (column(context, i)[ranked(frame, 0, "f1", PLACED)] == 0).all()


def test_a_direct_call_works_or_refuses_but_never_ignores_the_entry():
    """A direct call follows the entry or refuses with a code: an entry is never
    dropped without a word."""
    frame = blobs()
    truth = frame["GroundTruth_labels0"].to_numpy()
    coords = frame[["f1", "f2", "f3"]].to_numpy()
    clm = custom(feature_dependence=[entry()])
    try:
        labels = generate_clm_labels(truth, coords, clm, seed=1).to_numpy()
    except (ValueError, MissingConfigKey) as e:
        assert getattr(e, "code", None) is not None
        return
    assert (labels[ranked(frame, 0, "f1", PLACED)] == 0).all()


def test_a_read_only_feature_entry_is_accepted():
    """An entry may be any mapping, not only a dict."""
    frame = blobs()
    labels = column(label_frame(frame, custom(feature_dependence=[MappingProxyType(entry())])))
    assert (labels[ranked(frame, 0, "f1", PLACED)] == 0).all()


# ---------------------------------------------------------------------------
# What is refused and what is warned about
# ---------------------------------------------------------------------------

MALFORMED = {
    "a-mapping": {"label": 0, "feature": "f1"},
    "a-string": "f1",
    "a-number": 3,
    "entry-a-string": ["f1"],
    "entry-a-number": [3],
    "no-label": [{"feature": "f1", "favors": "high"}],
    "no-feature": [{"label": 0, "favors": "high"}],
    "label-a-string": [entry(label="0")],
    "label-a-float": [entry(label=1.5)],
    "label-a-bool": [entry(label=True)],
    "label-negative": [entry(label=-1)],
    "label-past-M": [entry(label=2)],
    "feature-a-number": [entry(feature=3)],
    "feature-empty": [entry(feature="")],
    "feature-a-list": [entry(feature=["f1", "f2"])],
    "favors-unknown": [entry(favors="middle")],
    "favors-of-centroid": [entry(favors="core")],
    "favors-a-bool": [entry(favors=True)],
    "value-without-a-value": [entry(favors="value")],
    "value-nan": [entry(favors="value", value=float("nan"))],
    "value-inf": [entry(favors="value", value=float("inf"))],
    "value-a-bool": [entry(favors="value", value=True)],
    "value-a-string": [entry(favors="value", value="0.5")],
    "profile-exponential": [entry(profile="exponential")],
    "profile-unknown": [entry(profile="geometric")],
    "profile-a-number": [entry(profile=3)],
    "a-second-entry-for-a-label": [entry(feature="f1"), entry(feature="f2")],
}


# A missing key and a label are coded as in an assignment_matrix row;
# every other case is [CLM-141].
CODED = {"no-label": 207, "no-feature": 207, "label-a-string": 104, "label-a-float": 104,
         "label-a-bool": 104, "label-negative": 104, "label-past-M": 104}


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_block_is_refused_before_any_dataset(name):
    """LabelGenerator.check is what a run calls before its first fetch."""
    with pytest.raises((ValueError, MissingConfigKey)) as caught:
        LabelGenerator(custom(feature_dependence=MALFORMED[name])).check()
    assert getattr(caught.value, "code", None) == CODED.get(name, 141)


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_block_is_refused_at_labelling(name):
    with pytest.raises((ValueError, MissingConfigKey)) as caught:
        label_frame(blobs(), custom(feature_dependence=MALFORMED[name]))
    assert getattr(caught.value, "code", None) == CODED.get(name, 141)


def test_centroid_dependence_enabled_with_entries_is_141():
    clm = custom(feature_dependence=[entry()],
                 centroid_dependence={"enabled": True, "favors": "core"})
    with pytest.raises(ValueError) as caught:
        LabelGenerator(clm).check()
    assert getattr(caught.value, "code", None) == 141
    assert "feature_dependence" in str(caught.value)


def test_centroid_dependence_written_but_off_does_not_stop_an_entry():
    frame = blobs()
    clm = custom(feature_dependence=[entry()],
                 centroid_dependence={"enabled": False, "favors": "core"})
    labels = column(label_frame(frame, clm))
    assert (labels[ranked(frame, 0, "f1", 160)] == 0).all()


def test_competing_noise_favors_is_no_conflict():
    """The only rule between settings is centroid_dependence: a noise entry's
    default favors 'boundary' does not refuse an entry."""
    frame = blobs()
    clm = custom(recall=0.5, feature_dependence=[entry()],
                 competing_noise=[{"cluster": 1, "label": 0, "share": 0.2}])
    labels = column(label_frame(frame, clm))
    assert (labels[ranked(frame, 0, "f1", 100)] == 0).all()


def test_a_label_with_a_noise_entry_warns_about_that_cluster(caplog):
    """Label 2 has a row in cluster 0, so the entry moves points there, and a noise
    entry in cluster 1, where the entry is ignored."""
    caplog.set_level(logging.WARNING)
    clm = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 2, "recall_target": 0.5},
                                 {"clusters": [1], "label": 1, "recall_target": 0.5}],
           "competing_noise": [{"cluster": 1, "label": 2, "share": 0.3}],
           "feature_dependence": [entry(label=2)]}
    label_frame(blobs(per_cluster=300), clm)
    assert warned(caplog, "feature_dependence", "cluster 1")


@pytest.mark.parametrize("mode", ["perfect", "random"])
def test_a_mode_with_nothing_to_move_warns(mode, caplog):
    caplog.set_level(logging.WARNING)
    clm = {"num_classes": 2, "matching_mode": mode, "feature_dependence": [entry()]}
    label_frame(blobs(), clm)
    assert warned(caplog, "feature_dependence")


def test_random_jumps_over_a_malformed_block_with_a_warning(caplog):
    """Under `random` the block is not checked: it warns that it is ignored and
    refuses nothing."""
    caplog.set_level(logging.WARNING)
    clm = {"num_classes": 2, "matching_mode": "random", "feature_dependence": "not a list"}
    label_frame(blobs(), clm)
    assert warned(caplog, "feature_dependence")


def test_random_with_centroid_style_block_needs_no_geometry(caplog):
    """A frame with no features still runs under random, and warns."""
    caplog.set_level(logging.WARNING)
    frame = blobs()[["GroundTruth_labels0"]]
    label_frame(frame, {"num_classes": 2, "matching_mode": "random",
                        "feature_dependence": [entry()]})
    assert warned(caplog, "feature_dependence")


def test_under_single_any_label_but_the_matched_one_warns(caplog):
    caplog.set_level(logging.WARNING)
    clm = {"num_classes": 2, "matching_mode": "single", "balance": "balanced",
           "single_match": {"cluster": 0, "label": 0},
           "feature_dependence": [entry(label=1)]}
    label_frame(blobs(), clm)
    assert warned(caplog, "feature_dependence")


def test_the_warning_is_for_the_delivered_labels_only(caplog):
    """A search's probes run with warnings off: one label, one warning."""
    caplog.set_level(logging.WARNING)
    clm = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 0},
                                 {"clusters": [1], "label": 1}],
           "target_metric": {"type": "mcc", "value": 0.5},
           "feature_dependence": [entry(label=2)]}
    label_frame(blobs(per_cluster=150, clusters=3), clm)
    assert len(warned(caplog, "feature_dependence")) == 1


def test_each_warning_names_the_entry(caplog):
    caplog.set_level(logging.WARNING)
    clm = {"num_classes": 3, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.5}],
           "feature_dependence": [entry(label=2, feature="f3")]}
    label_frame(blobs(), clm)
    hits = warned(caplog, "feature_dependence", "f3")
    assert hits and all(getattr(r, "clm_code", None) == 305 for r in hits)


def test_305_opens_with_where_it_applies():
    """[CLM-305] names its block."""
    assert CODES[305].startswith("{where}:")


def test_the_random_warning_is_311(caplog):
    """[CLM-311] names its block."""
    caplog.set_level(logging.WARNING)
    label_frame(blobs(), {"num_classes": 2, "matching_mode": "random",
                          "feature_dependence": [entry()]})
    assert [getattr(r, "clm_code", None) for r in warned(caplog, "feature_dependence")] == [311]


def test_125_names_feature_dependence():
    assert "feature_dependence" in CODES[125]
    assert not CODES[125].startswith("centroid placement")


def test_a_value_under_high_is_warned_about(caplog):
    """`value` is read only under `favors: value`."""
    caplog.set_level(logging.WARNING)
    label_frame(blobs(), custom(feature_dependence=[entry(value=0.5)]))
    assert warned(caplog, "feature_dependence", "value")


def test_a_label_that_fills_clusters_warns_once_and_names_them(caplog):
    """Label 0 takes all of clusters 70 and 72, where the entry has nothing to choose
    between, and half of cluster 71, where it still moves points."""
    caplog.set_level(logging.WARNING)
    frame = blobs(per_cluster=100, clusters=5)
    frame["GroundTruth_labels0"] += 70
    clm = {"num_classes": 2, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [70], "label": 0, "recall_target": 0.4},
                                 {"clusters": [72], "label": 0, "recall_target": 0.4},
                                 {"clusters": [71], "label": 0, "recall_target": 0.2}],
           "feature_dependence": [entry()]}
    labels = column(label_frame(frame, clm))
    hits = [r.getMessage() for r in warned(caplog, "feature_dependence")]
    assert len(hits) == 1
    assert "70" in hits[0] and "72" in hits[0] and "71" not in hits[0]
    assert (labels[ranked(frame, 71, "f1", 50)] == 0).all()


def test_perfect_still_skips_a_dataset_without_the_column():
    """Only `random` jumps over the block. Under `perfect` the entry is ignored, and a
    dataset without its column is still [CLM-125], as for centroid_dependence."""
    frame = blobs().rename(columns={"f1": "other"})
    with pytest.raises(ValueError) as caught:
        label_frame(frame, {"num_classes": 2, "matching_mode": "perfect",
                            "feature_dependence": [entry()]})
    assert getattr(caught.value, "code", None) == 125


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------

def test_the_tree_holds_the_choices_and_the_default_profile():
    favors = SETTINGS["clm_label.feature_dependence.favors"]
    assert favors.choices == ("high", "low", "value")
    assert favors.default == "high"
    profile = SETTINGS["clm_label.feature_dependence.profile"]
    assert profile.choices == ("linear", "step")
    assert profile.default == "linear"


# ---------------------------------------------------------------------------
# Through a run
# ---------------------------------------------------------------------------

def test_a_run_writes_a_following_label(tmp_path):
    clm = {"num_classes": 4, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [k], "label": k, "recall_target": 0.8}
                                 for k in range(4)],
           "feature_dependence": [entry(feature="Feature_3")]}
    _, _, csv_dir, _, _ = run("fabricated_data", fabricated_config(clm), tmp_path)
    frame = pd.read_csv(next(csv_dir.glob("*.csv")))
    frame = frame.rename(columns={"Cluster_0": "GroundTruth_labels0"})
    assert (frame["Label_0"].to_numpy()[ranked(frame, 0, "Feature_3", 160)] == 0).all()


def test_value_is_in_the_units_the_run_writes(tmp_path):
    """After byoc's standardize, when that is on."""
    csv_frame(blobs()).to_csv(tmp_path / "data.csv", index=False)
    clm = custom(feature_dependence=[entry(favors="value", value=0.25)])
    config = byoc_config(tmp_path, ["data"], clm, standardize=True)
    _, _, csv_dir, _, _ = run("byoc", config, tmp_path / "run")
    frame = pd.read_csv(next(csv_dir.glob("*.csv")))
    frame = frame.rename(columns={"Cluster_0": "GroundTruth_labels0"})
    assert (frame["Label_0"].to_numpy()[closest(frame, 0, "f1", 0.25, 160)] == 0).all()


def test_a_tag_column_can_not_be_followed(tmp_path):
    frame = csv_frame(blobs())
    frame["note"] = np.arange(len(frame))
    frame.to_csv(tmp_path / "data.csv", index=False)
    config = byoc_config(tmp_path, ["data"], custom(feature_dependence=[entry(feature="note")]),
                         tag_columns=["note"])
    n, report, _, _, _ = run("byoc", config, tmp_path / "run")
    assert (n, report["refused"]) == (0, 1)


def test_a_dataset_without_the_column_is_skipped_and_the_run_goes_on(tmp_path):
    """[CLM-125] is per dataset: one is skipped, the other written, exit 3."""
    csv_frame(blobs()).to_csv(tmp_path / "with.csv", index=False)
    csv_frame(blobs().rename(columns={"f1": "other"})).to_csv(tmp_path / "without.csv",
                                                              index=False)
    config = byoc_config(tmp_path, ["with", "without"],
                         custom(feature_dependence=[entry()]))
    n, report, _, _, _ = run("byoc", config, tmp_path / "run")
    assert (n, report["refused"]) == (1, 1)
    assert _exit_status(report) == 3


def test_one_feature_data_is_skipped(tmp_path):
    """Single-feature data is out of scope for placement, entries included."""
    csv_frame(blobs(features=("f1",))).to_csv(tmp_path / "one.csv", index=False)
    config = byoc_config(tmp_path, ["one"], custom(feature_dependence=[entry()]))
    n, report, _, _, _ = run("byoc", config, tmp_path / "run")
    assert (n, report["refused"]) == (0, 1)


def test_a_preset_with_no_features_is_skipped(tmp_path):
    clm = {"num_classes": 4, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.5}],
           "feature_dependence": [entry(feature="Feature_1")]}
    config = fabricated_config(clm, datasets=("labels_only_4class",))
    n, report, _, _, _ = run("fabricated_data", config, tmp_path)
    assert (n, report["refused"]) == (0, 1)


def test_a_malformed_entry_stops_the_run_before_any_fetch(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setitem(
        __import__("clmsynth._single_run", fromlist=["FETCHERS"]).FETCHERS, "fabricated_data",
        lambda *a, **k: fetched.append(a))
    clm = custom(M=4, feature_dependence=[entry(favors="middle")])
    with pytest.raises(ValueError) as caught:
        run("fabricated_data", fabricated_config(clm), tmp_path)
    assert getattr(caught.value, "code", None) == 141
    assert not fetched


# ---------------------------------------------------------------------------
# The saved config
# ---------------------------------------------------------------------------

def _resolved(clm):
    config = fabricated_config(clm)
    return resolve_config(config, "fabricated_data", ["fabricated"], ["baseline_4class"],
                          [("fabricated", "baseline_4class")], 7, {}, 1)[
        "label_generation"]["clm_label"]


def test_the_saved_config_records_the_list():
    block = _resolved(custom(M=4, feature_dependence=[entry(profile="step")]))
    assert block["feature_dependence"] == [entry(profile="step")]


def test_the_saved_config_fills_in_the_default_profile():
    plain = {"label": 0, "feature": "f1", "favors": "high"}
    block = _resolved(custom(M=4, feature_dependence=[plain]))
    assert block["feature_dependence"] == [{**plain, "profile": "linear"}]


def test_the_saved_config_fills_in_the_default_favors():
    plain = {"label": 0, "feature": "f1"}
    block = _resolved(custom(M=4, feature_dependence=[plain]))
    assert block["feature_dependence"] == [{**plain, "favors": "high", "profile": "linear"}]


def test_the_saved_config_keeps_value_only_where_it_is_read():
    block = _resolved(custom(M=4, feature_dependence=[
        entry(favors="value", value=0.5), entry(label=1, value=0.5)]))
    assert block["feature_dependence"][0]["value"] == 0.5
    assert "value" not in block["feature_dependence"][1]


def test_the_saved_config_leaves_it_out_under_random():
    """`random` does not read the block ([CLM-311]), so the saved config does not carry it."""
    block = _resolved({"num_classes": 4, "matching_mode": "random",
                       "feature_dependence": [entry()]})
    assert block["matching_mode"] == "random"
    assert "feature_dependence" not in block


def test_running_the_saved_config_repeats_the_run(tmp_path):
    clm = {"num_classes": 4, "matching_mode": "custom", "balance": "balanced",
           "assignment_matrix": [{"clusters": [k], "label": k, "recall_target": 0.8}
                                 for k in range(4)],
           "feature_dependence": [entry(feature="Feature_3", profile="linear")]}
    _, _, first, _, resolved = run("fabricated_data", fabricated_config(clm), tmp_path / "a")
    saved = json.loads(resolved.read_text(encoding="utf-8"))
    assert saved["label_generation"]["clm_label"]["feature_dependence"]
    _, _, again, _, _ = run("fabricated_data", saved, tmp_path / "b")
    name = next(first.glob("*.csv")).name
    assert (first / name).read_bytes() == (again / name).read_bytes()


# ---------------------------------------------------------------------------
# What the plots and clmsynth-config show
# ---------------------------------------------------------------------------

def test_the_plot_panel_has_one_line_for_it():
    text = _clm_info_text(custom(feature_dependence=[entry(feature="Feature_3")]))
    assert "Feature_3" in text


@pytest.mark.usefixtures("yaml")
def test_clmsynth_config_writes_the_list(tmp_path):
    out = tmp_path / "config.json"
    generate_base_config({
        "data_source": "fabricated_data", "batteries": ["fabricated"],
        "datasets": ["baseline_4class"], "num_classes": 4, "matching_mode": "custom",
        "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.8}],
        "feature_dependence": [entry(feature="Feature_3")]}, output_path=str(out))
    block = json.loads(out.read_text(encoding="utf-8"))["label_generation"]["clm_label"]
    assert block["feature_dependence"] == [entry(feature="Feature_3")]


@pytest.mark.usefixtures("yaml")
def test_clmsynth_config_refuses_a_malformed_entry(tmp_path):
    out = tmp_path / "config.json"
    with pytest.raises(ValueError) as caught:
        generate_base_config({
            "data_source": "fabricated_data", "batteries": ["fabricated"],
            "datasets": ["baseline_4class"], "num_classes": 4, "matching_mode": "custom",
            "assignment_matrix": [{"clusters": [0], "label": 0, "recall_target": 0.8}],
            "feature_dependence": [entry(favors="middle")]}, output_path=str(out))
    assert getattr(caught.value, "code", None) == 141
    assert not out.exists()


@pytest.mark.usefixtures("yaml")
def test_clmsynth_config_shows_the_random_warning(tmp_path, caplog):
    """Under `random` the block is ignored whatever the dataset, so its warning sits
    with [CLM-311] and clmsynth-config shows it too."""
    caplog.set_level(logging.WARNING)
    generate_base_config({
        "data_source": "fabricated_data", "batteries": ["fabricated"],
        "datasets": ["baseline_4class"], "num_classes": 4, "matching_mode": "random",
        "feature_dependence": [entry(feature="Feature_3")]},
        output_path=str(tmp_path / "config.json"))
    assert warned(caplog, "feature_dependence")


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------

FEATURE = "clm_label.feature_dependence"


@pytest.mark.parametrize("mode", ["single", "custom"])
def test_the_wizard_asks_for_an_entry(monkeypatch, mode):
    clm, _, _ = drive(monkeypatch, {"clm_label.matching_mode": mode,
                                    f"{FEATURE}._enabled": True})
    assert clm["feature_dependence"] == [
        {"label": 0, "feature": "f1", "favors": "high", "profile": "linear"}]


def test_the_wizard_loops_for_a_second_entry(monkeypatch):
    clm, _, _ = drive(monkeypatch, {
        "clm_label.matching_mode": "custom", f"{FEATURE}._enabled": True,
        f"{FEATURE}.label": iter([0, 1]), f"{FEATURE}._add_another": iter([True, False])})
    assert [e["label"] for e in clm["feature_dependence"]] == [0, 1]


def test_the_wizard_caps_the_label_at_M_minus_1(monkeypatch):
    _, _, overrides = drive(monkeypatch, {"clm_label.matching_mode": "custom",
                                          f"{FEATURE}._enabled": True})
    assert overrides[f"{FEATURE}.label"][0]["maxv"] == 2


def test_the_wizard_asks_for_value_only_with_favors_value(monkeypatch):
    _, asked, _ = drive(monkeypatch, {"clm_label.matching_mode": "custom",
                                      f"{FEATURE}._enabled": True})
    assert f"{FEATURE}.favors" in asked and f"{FEATURE}.value" not in asked
    clm, asked, _ = drive(monkeypatch, {
        "clm_label.matching_mode": "custom", f"{FEATURE}._enabled": True,
        f"{FEATURE}.favors": "value", f"{FEATURE}.value": 0.5})
    assert f"{FEATURE}.value" in asked and clm["feature_dependence"][0]["value"] == 0.5


def test_the_wizard_re_asks_a_second_entry_for_a_label(monkeypatch):
    """A conflict that follows from the answers alone is re-asked at its prompt."""
    clm, _, _ = drive(monkeypatch, {
        "clm_label.matching_mode": "custom", f"{FEATURE}._enabled": True,
        f"{FEATURE}.label": iter([0, 0, 1]), f"{FEATURE}._add_another": iter([True, False])})
    assert [e["label"] for e in clm["feature_dependence"]] == [0, 1]


@pytest.mark.parametrize("mode", ["perfect", "random"])
def test_the_wizard_leaves_it_out_where_it_has_no_effect_but_asks_elsewhere(monkeypatch, mode):
    """Asked only under single and custom; custom is checked too, so the test is
    not satisfied by a wizard that never asks."""
    _, asked, _ = drive(monkeypatch, {"clm_label.matching_mode": mode})
    assert f"{FEATURE}._enabled" not in asked
    _, asked, _ = drive(monkeypatch, {"clm_label.matching_mode": "custom"})
    assert f"{FEATURE}._enabled" in asked


def test_the_wizard_leaves_it_out_after_centroid_was_enabled_but_asks_otherwise(monkeypatch):
    _, asked, _ = drive(monkeypatch, {"clm_label.matching_mode": "custom",
                                      "clm_label.centroid_dependence._enabled": True})
    assert f"{FEATURE}._enabled" not in asked
    _, asked, _ = drive(monkeypatch, {"clm_label.matching_mode": "custom"})
    assert f"{FEATURE}._enabled" in asked
