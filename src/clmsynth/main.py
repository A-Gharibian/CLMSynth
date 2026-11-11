# SPDX-License-Identifier: MIT
# main.py
"""Deprecated import path, removed in 0.7.8. The clmsynth command stays."""

import logging
import sys
import warnings
from typing import Any

from .__main__ import main
from ._single_run import (
    FETCHER_SUITE_KEYS,
    FETCHERS,
    PER_DATASET_CODES,
    SEEDED_SOURCES,
    SOURCE_DISPLAY,
    SUITE_KEYS,
    ConfigFileError,
    precheck_byoc_matching_ids,
    run_pipeline,
)
from ._single_run import load_config as _load_config
from ._utils.write_files import build_run_dir, discard_failed_run

__all__ = [
    "FETCHERS",
    "FETCHER_SUITE_KEYS",
    "PER_DATASET_CODES",
    "SEEDED_SOURCES",
    "SOURCE_DISPLAY",
    "SUITE_KEYS",
    "build_run_dir",
    "discard_failed_run",
    "load_config",
    "main",
    "precheck_byoc_matching_ids",
    "run_pipeline",
]

log: logging.Logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Loads the pipeline config; exits 2 with a message if absent, malformed or
    empty, and 1 if a YAML config finds no PyYAML."""
    try:
        return _load_config(config_path)
    except ConfigFileError as e:
        log.critical(str(e))
        sys.exit(2)
    except ImportError as e:
        log.critical(str(e))
        sys.exit(1)


warnings.warn("clmsynth.main is deprecated and is removed in 0.7.8; use the clmsynth command, "
              "or import run_pipeline and load_config from clmsynth",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    warnings.warn("python -m clmsynth.main is removed in 0.7.8; run clmsynth instead",
                  FutureWarning, stacklevel=1)
    main()
