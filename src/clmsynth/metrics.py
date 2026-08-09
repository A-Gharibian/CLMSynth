# metrics.py
"""Evaluation metrics for cluster-label agreement.
"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, matthews_corrcoef
from sklearn.metrics.cluster import contingency_matrix

try:
    import pyivm
except ImportError:
    pyivm = None

log = logging.getLogger(__name__)

def clustering_ari(labels_true, labels_pred) -> float:
    """
    Adjusted Rand Index between two clusterings.
    """
    return adjusted_rand_score(np.asarray(labels_true), np.asarray(labels_pred))


def clustering_mcc_pair(labels_true, labels_pred, cluster, label) -> float:
    """Single-pair (2x2) Matthews correlation: one cluster vs the rest against
    one label vs the rest.

    This is the binary building block of the multiclass `clustering_mcc`
    (Gorodkin R_K): it scores only the agreement of the designated
    (`cluster`, `label`) pair, not the whole partition, so it needs no
    Hungarian matching. Used by the engine's target_metric scope='pair'.
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

    This wrapper adds the missing step.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples)
    labels_pred : array-like of shape (n_samples)

    Returns
    -------
    float in [-1, 1]
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

    # Best one-to-one correspondence maximizing total overlap.
    # linear_sum_assignment handles rectangular matrices natively,
    # returning min(n_true, n_pred) matched pairs.
    row_ind, col_ind = linear_sum_assignment(-C)

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

def evaluate_cluster_label_matching(
        df: pd.DataFrame,
        label_col: str = "Cohort_Class"
) -> dict[str, float]:
    """
    Evaluates the Cluster-Label Matching (CLM) of a dataset.
    This measures how well the provided ground-truth labels align with the
    actual data clusters, using Adjusted Internal Validation Measures (IVMAs)
    from TPAMI 2025 (https://arxiv.org/abs/2503.01097).
    """
    if pyivm is None:
        log.warning("The 'pyivm' package is not installed. Cannot compute CLM metrics. Run 'pip install pyivm'")
        return {}

    if df is None or df.empty:
        log.warning("DataFrame is empty. Skipping metrics calculation.")
        return {}

    if label_col not in df.columns:
        log.warning(f"Label column '{label_col}' not found in dataset. Skipping CLM evaluation.")
        return {}

    log.info("Computing Adjusted IVMs for Cluster-Label Matching (CLM)...")

    # 1. Separate features and labels
    feature_cols = [col for col in df.columns if col != label_col]
    X = df[feature_cols].to_numpy(dtype=np.float64)

    # 2. Handle string/categorical labels by converting them to integers
    # (pyivm, like most clustering metrics, requires numeric labels).
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        _, labels = np.unique(df[label_col].astype(str).to_numpy(), return_inverse=True)
    else:
        labels = df[label_col].to_numpy()

    # 3. Guard against single-class datasets (IVMs require at least 2 clusters)
    if len(np.unique(labels)) < 2:
        log.warning("Dataset contains fewer than 2 distinct classes. Cannot compute CLM.")
        return {}

    metrics = {}

    try:
        # 4. Compute Adjusted Metrics (higher = better CLM)
        # Using adjusted=True ensures fair cross-dataset evaluation, removing biases
        # related to dimensionality, dataset size, and cluster count.
        metrics["adj_silhouette"] = pyivm.silhouette(X, labels, adjusted=True)
        metrics["adj_calinski_harabasz"] = pyivm.calinski_harabasz(X, labels, adjusted=True)
        metrics["adj_davies_bouldin"] = pyivm.davies_bouldin(X, labels, adjusted=True)

        log.info(
            f"CLM Evaluation Complete: Adj_Sil={metrics['adj_silhouette']:.3f}, "
            f"Adj_CH={metrics['adj_calinski_harabasz']:.3f}, "
            f"Adj_DB={metrics['adj_davies_bouldin']:.3f}"
        )

    except Exception as e:
        log.error(f"Error computing CLM metrics: {e}")

    return metrics