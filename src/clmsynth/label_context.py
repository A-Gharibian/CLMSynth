# SPDX-License-Identifier: MIT
# label_context.py
"""Deprecated import path, removed in 0.7.8."""

import warnings

from .labeling import DatasetContext, build_context

__all__ = ["DatasetContext", "build_context"]

warnings.warn("clmsynth.label_context is deprecated and is removed in 0.7.8; "
              "use clmsynth.labeling", DeprecationWarning, stacklevel=2)
