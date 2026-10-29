# SPDX-License-Identifier: MIT
# metrics.py
"""Evaluation metrics for cluster-label agreement, and their ceilings given the group sizes.
"""
from math import comb

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.api.types import infer_dtype
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, matthews_corrcoef
from sklearn.metrics.cluster import contingency_matrix

__all__ = [
    "MetricCeilings",
    "clustering_ari",
    "clustering_ari_pair",
    "clustering_mcc",
    "clustering_mcc_pair",
]


def _label_arrays(labels_true: ArrayLike, labels_pred: ArrayLike,
                  *pair: object) -> tuple[np.ndarray, np.ndarray]:
    """Both label arrays, refused when no documented use would pass them; `pair` is
    the (cluster, label) of a pair measure."""
    t, p = np.asarray(labels_true), np.asarray(labels_pred)
    if t.ndim != 1 or p.ndim != 1:
        problem = "labels_true and labels_pred must be 1-D"
    elif len(t) != len(p):
        problem = f"labels_true and labels_pred differ in length ({len(t)} and {len(p)})"
    elif len(t) == 0:
        problem = "labels_true and labels_pred are empty"
    elif any(pd.isna(a).any()
             or (a.dtype.kind in "fO" and pd.Series(a).isin([np.inf, -np.inf]).any())
             for a in (t, p)):
        problem = "labels_true or labels_pred holds missing values (NaN, None or inf)"
    elif any(a.dtype == object and infer_dtype(a) in ("mixed", "mixed-integer") for a in (t, p)):
        problem = "labels_true or labels_pred mixes id types, such as int and str"
    elif pair and not (t == pair[0]).any():
        problem = f"labels_true has no cluster {pair[0]!r}"
    elif pair and not (p == pair[1]).any():
        problem = f"labels_pred has no label {pair[1]!r}"
    elif pair and (t == t[0]).all() and (p == p[0]).all():
        problem = "labels_true and labels_pred each hold one group, where MCC is 0 and ARI 1"
    else:
        return t, p
    raise ValueError(f"{problem}. CLMSynth is meant to be run as documented, which never "
                     "passes such input to its metric functions.")


def clustering_ari(labels_true: ArrayLike, labels_pred: ArrayLike) -> float:
    """Adjusted Rand Index between two clusterings.

    Label *values* are irrelevant: ARI scores which points share a group, so it
    is invariant to renaming clusters and needs no matching step.

    Parameters
    ----------
    labels_true : array-like of int or str, shape (n_samples,)
        Reference partition, normally the ground-truth cluster ids.
    labels_pred : array-like of int, shape (n_samples,)
        Partition to score, normally a generated ``Label_n`` column.

    Returns
    -------
    float
        In ``[-1/2, 1]`` (Chacón & Rastrojo, 2023). ``1.0`` is identical
        partitions; ``0.0`` is the value expected of an independent random
        labeling.
    """
    return float(adjusted_rand_score(*_label_arrays(labels_true, labels_pred)))


def clustering_mcc_pair(labels_true: ArrayLike, labels_pred: ArrayLike,
                        cluster: int | str, label: int) -> float:
    """Single-pair (2x2) Matthews correlation: one cluster vs the rest against
    one label vs the rest.
    Parameters
    ----------
    labels_true : array-like of int or str, shape (n_samples,)
        Reference partition holding ``cluster`` among its ids.
    labels_pred : array-like of int, shape (n_samples,)
        Partition to score, holding ``label`` among its ids.
    cluster : int or str
        The one cluster id collapsed to the positive class in ``labels_true``.
    label : int
        The one label collapsed to the positive class in ``labels_pred``.

    Returns
    -------
    float
        In ``[-1, 1]``, for the 2x2 problem only. It says nothing about how the
        remaining clusters and labels agree.

    Examples
    --------
    >>> from clmsynth import clustering_mcc_pair
    >>> clustering_mcc_pair([0, 0, 1, 1], [5, 5, 9, 9], cluster=0, label=5)
    1.0
    """
    t, p = _label_arrays(labels_true, labels_pred, cluster, label)
    u = (t == cluster).astype(int)
    v = (p == label).astype(int)
    return float(matthews_corrcoef(u, v))


def clustering_ari_pair(labels_true: ArrayLike, labels_pred: ArrayLike,
                        cluster: int | str, label: int) -> float:
    """Single-pair adjusted Rand index: one cluster vs the rest against one
    label vs the rest.

    Returns
    -------
    float
        In ``[-1/2, 1]``, for the 2x2 problem only.
    """
    t, p = _label_arrays(labels_true, labels_pred, cluster, label)
    u = (t == cluster).astype(int)
    v = (p == label).astype(int)
    return float(adjusted_rand_score(u, v))


def clustering_mcc(labels_true: ArrayLike, labels_pred: ArrayLike) -> float:
    """
    Gorodkin's K-category correlation coefficient (R_K), the
    confusion-matrix generalization of MCC, adapted for comparing two
    clusterings.

    scikit-learn's `matthews_corrcoef` already implements R_K for the
    multiclass case, this wrapper adds the missing step, matching to maximise R_K.
    but it assumes label i in y_true and label i in y_pred refer to the
    same category. That holds in classification but not in clustering,
    where cluster IDs are arbitrary and the number of clusters can differ
    between the ground truth and a candidate partition.

    Parameters
    ----------
    labels_true : array-like of int or str, shape (n_samples,)
        Reference partition, normally the ground-truth cluster ids. May hold any
        number of distinct values; ids need not be contiguous or integer.
    labels_pred : array-like of int, shape (n_samples,)
        Partition to score. Its number of distinct values may differ from
        ``labels_true``; surplus predicted clusters are given fresh codes so they
        cannot spuriously agree with any true cluster.

    Returns
    -------
    float
        In ``[0, 1]``. The value is the maximum of R_K over cluster-label
        matchings. A single matching can score below 0, but the scores of all
        matchings average exactly 0, so their maximum never does. ``1.0`` is a
        perfect correspondence under some relabeling.

    Notes
    -----
    The matching maximises R_K rather than accuracy: the assignment runs on
    ``n * C - outer(t, p)`` for contingency matrix ``C`` with row sums ``t`` and
    column sums ``p``. Reading R_K off the accuracy-optimal matching instead can
    understate agreement, even below 0.
    """
    labels_true, labels_pred = _label_arrays(labels_true, labels_pred)

    # Original label values/names don't matter to MCC, only which
    # points share a cluster, so work in dense integer code space.
    _, true_codes = np.unique(labels_true, return_inverse=True)
    _, pred_codes = np.unique(labels_pred, return_inverse=True)
    n_true = int(true_codes.max()) + 1
    n_pred = int(pred_codes.max()) + 1

    C = contingency_matrix(true_codes, pred_codes).astype(np.float64)

    # Assignment that maximizes R_K, not accuracy.
    W = C.sum() * C - np.outer(C.sum(axis=1), C.sum(axis=0))
    row_ind, col_ind = linear_sum_assignment(-W)

    # Relabel predicted clusters: a matched one takes its matched true
    # cluster's code. An unmatched predicted cluster (only possible when
    # n_pred > n_true) gets a fresh code so it can't spuriously "agree"
    # with any true cluster.
    pred_relabel = np.full(n_pred, -1, dtype=np.int64)
    pred_relabel[col_ind] = row_ind
    next_free_code = n_true
    for p in range(n_pred):
        if pred_relabel[p] == -1:
            pred_relabel[p] = next_free_code
            next_free_code += 1

    recoded_pred = pred_relabel[pred_codes]
    return float(matthews_corrcoef(true_codes, recoded_pred))


class MetricCeilings:
    """Highest values the agreement measures can take, given only the group sizes.
    The engine reads it for the size ceiling it reports, which reaches users as
    ``report["ceiling"]`` of ``generate_clm_labels``.
    """

    @staticmethod
    def _sizes(cluster_sizes: ArrayLike, label_sizes: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Both size lists as int64 arrays, after checking they are 1-D lists of
        counts >= 0 with the same total."""
        try:
            t = np.asarray(cluster_sizes, dtype=np.int64)
            p = np.asarray(label_sizes, dtype=np.int64)
            counts = t.ndim == p.ndim == 1 and (t >= 0).all() and (p >= 0).all()
        except (TypeError, ValueError, OverflowError):
            counts = False
        if not counts:
            raise ValueError("cluster_sizes and label_sizes must be 1-D lists of counts >= 0")
        if t.sum() != p.sum():
            raise ValueError("cluster_sizes and label_sizes must count the same points, "
                             f"got {int(t.sum())} and {int(p.sum())}")
        return t, p

    @staticmethod
    def rk_ceiling(cluster_sizes: ArrayLike, label_sizes: ArrayLike) -> float:
        """Maximum of `clustering_mcc` (R_K) over every labeling with these sizes.
        Parameters
        ----------
        cluster_sizes : array-like of int
            Point count of each cluster (``t_k``).
        label_sizes : array-like of int
            Point count of each label (``p_j``), with the same total.

        Returns
        -------
        float
            In ``[0, 1]``. ``1.0`` only when the two lists hold the same non-zero
            sizes in some order. ``0.0`` when either side has a single non-empty
            group, where R_K is 0 for every labeling.
        """
        t, p = MetricCeilings._sizes(cluster_sizes, label_sizes)
        t = t.astype(np.float64)
        p = p.astype(np.float64)
        n = t.sum()

        denom_sq = (n * n - (p * p).sum()) * (n * n - (t * t).sum())
        if denom_sq == 0:
            return 0.0

        # Largest numerator term each cluster/label pair can contribute if matched.
        gain = n * np.minimum.outer(t, p) - np.outer(t, p)
        row_ind, col_ind = linear_sum_assignment(-gain)
        return float(gain[row_ind, col_ind].sum() / np.sqrt(denom_sq))

    @staticmethod
    def ari_bound(cluster_sizes: ArrayLike, label_sizes: ArrayLike) -> float:
        """Upper bound of `clustering_ari` over every labeling with these sizes.

        ARI counts pairs of points. With ``A`` pairs sharing a cluster, ``B``
        sharing a label and ``P = C(N, 2)`` pairs in all, it rises with the pairs
        that share both, which cannot exceed ``min(A, B)``. Putting that count into
        the Hubert-Arabie formula gives the bound.

        Unlike `rk_ceiling`, the bound is not always attained. It is reached only
        when one partition can sit entirely inside the other, which needs the
        smaller groups to add up exactly to the larger ones: for sizes ``[2, 2]``
        against ``[3, 1]`` it is 2/3, yet every labeling scores 0. The exact
        maximum is an integer optimization with no general closed form.

        Parameters
        ----------
        cluster_sizes : array-like of int
            Point count of each cluster.
        label_sizes : array-like of int
            Point count of each label, with the same total.

        Returns
        -------
        float
            At most ``1.0``. No labeling with these sizes scores above it.
        """
        t, p = MetricCeilings._sizes(cluster_sizes, label_sizes)
        a = sum(comb(int(x), 2) for x in t)
        b = sum(comb(int(x), 2) for x in p)
        pairs = comb(int(t.sum()), 2)

        # Hubert-Arabie with min(A, B) shared pairs, multiplied through by P so the
        # arithmetic stays in exact integers until the final division.
        numerator = 2 * (pairs * min(a, b) - a * b)
        denominator = a * (pairs - b) + b * (pairs - a)
        if denominator == 0:
            # Both sides split every pair, or both join every pair: the one labeling
            # scores 1. One side splits every pair and the other joins them all: 0.
            return 1.0 if a == b else 0.0
        return numerator / denominator
