# SPDX-License-Identifier: MIT
# clmsynth/__init__.py
"""CLMSynth: synthetic label generation with configurable cluster-label
matching.
run as ``python -m clmsynth`` or via the ``clmsynth`` console script.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Before the imports: _single_run reads it.
__version__ = "0.7.9"

# Each public name imports its module on first use (PEP 562), so importing the
# package alone loads no numpy, pandas, scipy or scikit-learn.
_LAZY: dict[str, str] = {
    "load_config": "._utils.config_files",
    "run_pipeline": "._single_run",
    "InfeasibleAllocationError": ".clm_errors",
    "MissingConfigKey": ".clm_errors",
    "generate_clm_labels": ".clm_label_engine",
    "DatasetContext": ".labeling",
    "build_context": ".labeling",
    "generate_additional_labels": ".labeling",
    "clustering_ari": ".metrics",
    "clustering_mcc": ".metrics",
    "clustering_mcc_pair": ".metrics",
}

# mypy reads these imports; a module-level __getattr__ visible to it would type
# every attribute of the package as Any.
if TYPE_CHECKING:
    from ._single_run import run_pipeline
    from ._utils.config_files import load_config
    from .clm_errors import InfeasibleAllocationError, MissingConfigKey
    from .clm_label_engine import generate_clm_labels
    from .labeling import DatasetContext, build_context, generate_additional_labels
    from .metrics import (
        clustering_ari,
        clustering_mcc,
        clustering_mcc_pair,
    )
else:
    def __getattr__(name: str) -> Any:
        if name not in _LAZY:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


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
