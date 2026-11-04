# SPDX-License-Identifier: MIT
# generate_config.py
"""Deprecated import path, removed in 0.8.0. The clmsynth-config command stays."""

import warnings

from ._config_gen.generate_config import (
    VALID_SOURCES,
    format_list_or_all,
    format_yaml_snippet,
    generate_base_config,
    main,
)
from ._utils.write_files import write_text_atomic

__all__ = [
    "VALID_SOURCES",
    "format_list_or_all",
    "format_yaml_snippet",
    "generate_base_config",
    "main",
    "write_text_atomic",
]

warnings.warn("clmsynth.generate_config is deprecated and is removed in 0.8.0; "
              "use the clmsynth-config command", DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    warnings.warn("python -m clmsynth.generate_config is removed in 0.8.0; "
                  "run clmsynth-config instead", FutureWarning, stacklevel=1)
    main()
