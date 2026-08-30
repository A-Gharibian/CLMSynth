# metrics.py
"""Evaluation metrics for cluster-label agreement.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, matthews_corrcoef
from sklearn.metrics.cluster import contingency_matrix


def clustering_ari(labels_true, labels_pred) -> float:
    """Adjusted Rand Index between two clusterings.

    Label *values* are irrelevant: ARI scores which points share a group, so it
    is invariant to renaming clusters and needs no matching step.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Reference partition, normally the ground-truth cluster ids.
    labels_pred : array-like of shape (n_samples,)
        Partition to score, normally a generated ``Label_n`` column.

    Returns
    -------
    float
        In ``[-1, 1]``. ``1.0`` is identical partitions; ``0.0`` is the value
        expected of an independent random labelling.

    See Also
    --------
    clustering_mcc : Gorodkin's R_K, which does need a matching step.

    Examples
    --------
    >>> from clmsynth import clustering_ari
    >>> clustering_ari([0, 0, 1, 1], [1, 1, 0, 0])
    1.0
    """
    return adjusted_rand_score(np.asarray(labels_true), np.asarray(labels_pred))


def clustering_mcc_pair(labels_true, labels_pred, cluster, label) -> float:
    """Single-pair (2x2) Matthews correlation: one cluster vs the rest against
    one label vs the rest.

    This is the binary building block of the multiclass `clustering_mcc`
    (Gorodkin's R_K): it scores only the agreement of the designated
    (`cluster`, `label`) pair, not the whole partition, so it needs no
    Hungarian matching. Used by the engine's target_metric scope='pair'.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Reference partition holding ``cluster`` among its ids.
    labels_pred : array-like of shape (n_samples,)
        Partition to score, holding ``label`` among its ids.
    cluster : hashable
        The one cluster id collapsed to the positive class in ``labels_true``.
    label : hashable
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
    u = (np.asarray(labels_true) == cluster).astype(int)
    v = (np.asarray(labels_pred) == label).astype(int)
    return float(matthews_corrcoef(u, v))


def clustering_mcc(labels_true, labels_pred):
    """
    Gorodkin's (2004) K-category correlation coefficient (R_K), the
    confusion-matrix generalization of MCC, adapted for comparing two
    clusterings.

    scikit-learn's `matthews_corrcoef` already implements R_K for the
    multiclass case (see its docstring: Gorodkin 2004; Jurman et al. 2012),
    but it assumes label i in y_true and label i in y_pred refer to the
    same category. That holds in classification but not in clustering,
    where cluster IDs are arbitrary and the number of clusters can differ
    between the ground truth and a candidate partition.

    This wrapper adds the missing step, matching to maximise R_K.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Reference partition, normally the ground-truth cluster ids. May hold any
        number of distinct values; ids need not be contiguous or integer.
    labels_pred : array-like of shape (n_samples,)
        Partition to score. Its number of distinct values may differ from
        ``labels_true``; surplus predicted clusters are given fresh codes so they
        cannot spuriously agree with any true cluster.

    Returns
    -------
    float
        In ``[-1, 1]``, maximised over cluster-label matchings. ``1.0`` is a
        perfect correspondence under some relabelling.

    See Also
    --------
    clustering_ari : Matching-free alternative.
    clustering_mcc_pair : The 2x2 building block used by ``scope='pair'``.

    Notes
    -----
    The matching maximises R_K rather than accuracy: the assignment runs on
    ``n * C - outer(t, p)`` for contingency matrix ``C`` with row sums ``t`` and
    column sums ``p``. Reading R_K off the accuracy-optimal matching instead can
    understate agreement, and could previously return a negative value.

    Examples
    --------
    >>> from clmsynth import clustering_mcc
    >>> clustering_mcc([0, 0, 1, 1], [7, 7, 3, 3])
    1.0
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

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
    return matthews_corrcoef(true_codes, recoded_pred)
