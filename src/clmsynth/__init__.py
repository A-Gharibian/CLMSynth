# clmsynth/__init__.py
"""CLMSynth: synthetic label generation with configurable cluster-label
matching (CLM).
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

__version__ = "0.6.6"

__all__ = [
    "DatasetContext",
    "InfeasibleAllocationError",
    "__version__",
    "build_context",
    "clustering_ari",
    "clustering_mcc",
    "clustering_mcc_pair",
    "evaluate_cluster_label_matching",
    "generate_additional_labels",
    "generate_clm_labels",
]
