# SPDX-License-Identifier: MIT
# clmsynth/__init__.py
"""CLMSynth: synthetic label generation with configurable cluster-label
matching.
run as ``python -m clmsynth`` or via the ``clmsynth`` console script.
"""

# Before the imports: _single_run reads it.
__version__ = "0.7.7"

from ._single_run import load_config, run_pipeline
from .clm_errors import InfeasibleAllocationError, MissingConfigKey
from .clm_label_engine import generate_clm_labels
from .labeling import DatasetContext, build_context, generate_additional_labels
from .metrics import (
    clustering_ari,
    clustering_mcc,
    clustering_mcc_pair,
)

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
    "load_config",
    "run_pipeline",
]
