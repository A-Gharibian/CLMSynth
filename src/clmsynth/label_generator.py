# SPDX-License-Identifier: MIT
# label_generator.py
"""Deprecated import path, removed in 0.7.8."""

import warnings

from .labeling import generate_additional_labels

__all__ = ["generate_additional_labels"]

warnings.warn("clmsynth.label_generator is deprecated and is removed in 0.7.8; "
              "use clmsynth.labeling", DeprecationWarning, stacklevel=2)
