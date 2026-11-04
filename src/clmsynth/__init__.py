# SPDX-License-Identifier: MIT
# clmsynth/__init__.py
"""CLMSynth: synthetic label generation with configurable cluster-label
matching.
run as ``python -m clmsynth``, ``python -m clmsynth.main`` or via the ``clmsynth`` console script.
"""

from .clm_errors import InfeasibleAllocationError, MissingConfigKey
from .clm_label_engine import generate_clm_labels
from .labeling import DatasetContext, build_context, generate_additional_labels
from .metrics import (
    clustering_ari,
    clustering_mcc,
    clustering_mcc_pair,
)

__version__ = "0.7.6"

__all__ = [
    "DatasetContext",
    "InfeasibleAllocationError",
    "MissingConfigKey",
    "__version__",
    "build_context",
    "clustering_ari",
    "clustering_mcc",
    "clustering_mcc_pair",
    "generate_additional_labels",
    "generate_clm_labels",
]
