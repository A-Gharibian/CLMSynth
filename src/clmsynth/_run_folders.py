# SPDX-License-Identifier: MIT
# _run_folders.py
"""The run folder: its name, created as it is handed out, its csv/, png/ and info/
folders, how a batch recognises one, and its removal after a failed run that left
nothing of its own."""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Final

from ._utils.cli_logging import removed_line

__all__ = [
    "SUBFOLDERS",
    "build_run_dir",
    "discard_failed_run",
    "is_run_folder",
    "make_subfolders",
    "subfolders",
]

log: logging.Logger = logging.getLogger(__name__)

# The datasets' CSVs and summaries, the plots, and the run's config and log.
_INFO: Final = "info"
SUBFOLDERS: Final = ("csv", "png", _INFO)


def build_run_dir(base_dir: Path, friendly_source: str) -> Path:
    """One folder per run: YYYYMMDD_HHMMSS_Source/, then _2, _3 for runs
    started in the same second.
    """
    now = datetime.now()
    stem = f"{now:%Y%m%d_%H%M%S}_{friendly_source}"
    candidate, n = base_dir / stem, 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            # The name is taken only if the candidate itself exists. On Windows a
            # FILE where `base_dir` or a folder above it should be also raises
            # FileExistsError, and a retry under the next suffix meets the same
            # file every time: the loop never ends.
            if not candidate.exists(follow_symlinks=False):
                raise
            n += 1
            candidate = base_dir / f"{stem}_{n}"


def subfolders(run_dir: Path) -> tuple[Path, Path, Path]:
    """The run's csv/, png/ and info/ folders, in that order."""
    csv_dir, png_dir, info_dir = (run_dir / sub for sub in SUBFOLDERS)
    return csv_dir, png_dir, info_dir


def make_subfolders(run_dir: Path) -> tuple[Path, Path, Path]:
    """Creates the run's csv/, png/ and info/ folders and returns them, in that order."""
    folders = subfolders(run_dir)
    for folder in folders:
        folder.mkdir(exist_ok=True)
    return folders


def is_run_folder(path: Path) -> bool:
    """Whether `path` holds a run's csv/, png/ and info/ folders."""
    return all((path / sub).is_dir() for sub in SUBFOLDERS)


def discard_failed_run(run_dir: Path, keep_failed: bool) -> bool:
    """Remove a failed run's folder, unless --keep-failed keeps its config and log.

    Only a folder holding what a run starts with: empty csv/ and png/, and info/
    with the run's config and log. Anything else, such as a dataset written
    before the failure, keeps the folder.
    """
    if keep_failed:
        log.info(f"Failed run kept: its config and log are in '{run_dir / _INFO}'.")
        return False
    if not run_dir.is_dir():
        return False
    own = {f"{run_dir.name}{ext}" for ext in (".yaml", ".json", ".log")}
    for entry in run_dir.iterdir():
        allowed = own if entry.name == _INFO else set()
        if (entry.name not in SUBFOLDERS or not entry.is_dir()
                or any(p.name not in allowed for p in entry.iterdir())):
            return False

    try:
        shutil.rmtree(run_dir)
    except OSError as e:
        # Never fatal: the run has already failed and reported why, and a folder
        # that could not be removed is untidy rather than wrong.
        log.warning(f"Could not remove the failed run's folder '{run_dir}': {e}")
        return False

    log.info(f"{removed_line(run_dir)} Run with --keep-failed to keep its config and log.")
    return True
