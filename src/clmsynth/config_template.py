# SPDX-License-Identifier: MIT
# config_template.py
"""Deprecated import path, removed in 0.7.8."""

import warnings

from ._config_gen.config_template import PERFECT_SIZING, SIZING_TEMPLATE, YAML_TEMPLATE

__all__ = ["PERFECT_SIZING", "SIZING_TEMPLATE", "YAML_TEMPLATE"]

warnings.warn("clmsynth.config_template is deprecated and is removed in 0.7.8",
              DeprecationWarning, stacklevel=2)
