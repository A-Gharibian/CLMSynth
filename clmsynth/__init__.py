# clmsynth/__init__.py
"""CLMSynth: synthetic label generation with configurable cluster-label
matching (CLM) against ground-truth clusters.

Public API re-exports the pieces most callers need; the pipeline itself is
run as ``python -m clmsynth.main`` or via the ``clmsynth`` console script.
"""

from .clm_errors import InfeasibleAllocationError
from .clm_label_engine import generate_clm_labels
from .label_context import DatasetContext, build_context
from .label_generator import generate_additional_labels
from .metrics import (
    clustering_ari,
    clustering_mcc,
    clustering_mcc_pair,
    evaluate_cluster_label_matching,
)

__version__ = "0.3.0"

__all__ = [
    "InfeasibleAllocationError",
    "generate_clm_labels",
    "DatasetContext",
    "build_context",
    "generate_additional_labels",
    # The three agreement coefficients the manual documents: the multiclass
    # Gorodkin R_K, the 2x2 Matthews phi of one (cluster, label) pair that
    # target_metric.scope='pair' inverts in closed form, and the ARI.
    "clustering_ari",
    "clustering_mcc",
    "clustering_mcc_pair",
    "evaluate_cluster_label_matching",
    "__version__",
]
