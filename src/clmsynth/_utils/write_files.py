# SPDX-License-Identifier: MIT
# write_files.py
"""Files written whole or not at all, run outputs and the configs the guides write,
and the run folder they go into: created as its name is handed out, removed after a
failed run that left nothing of its own. Also a path's absolute form, for messages."""

import contextlib
import logging
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

__all__ = [
    "build_run_dir",
    "discard_failed_run",
    "file_in_the_way",
    "resolved_for_report",
    "write_text_atomic",
    "write_then_replace",
]

log: logging.Logger = logging.getLogger(__name__)


def write_then_replace(path: str | Path, write: Callable[[Path], object]) -> None:
    """Runs `write` on a temporary file beside `path`, then os.replace swaps it in,
    so `path` never holds part of the new content. The temporary file is removed
    when `write` fails or is interrupted. A symlinked `path` is followed, so the
    link survives."""
    target = Path(path).resolve()
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(path: str | Path, text: str) -> None:
    """Writes `text` to `path` so the file holds either its old content or all of
    the new.

    The text goes to a temporary file in the same folder and os.replace swaps it
    in. A symlinked `path` is followed, so the link survives.
    """
    def write(tmp: Path) -> None:
        # open(..., "x") rather than tempfile.mkstemp, which creates the file 0600:
        # the config would lose the read permission the umask gives everyone else.
        with open(tmp, "x", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

    write_then_replace(path, write)


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


def file_in_the_way(path: Path) -> Path | None:
    """The first existing part of `path` that is a file rather than a folder, or None."""
    for part in (*reversed(path.parents), path):
        if part.exists() and not part.is_dir():
            return part
    return None


def resolved_for_report(path_value: str | Path) -> str:
    """Absolute form, or the raw value."""
    try:
        return str(Path(path_value).resolve())
    except (OSError, ValueError):
        return str(path_value)


def discard_failed_run(run_dir: Path, keep_failed: bool) -> bool:
    """Remove a failed run's folder, unless --keep-failed keeps its config and log.

    Only a folder holding what a run starts with: empty csv/ and png/, and info/
    with the run's config and log. Anything else, such as a dataset written
    before the failure, keeps the folder.
    """
    if keep_failed:
        log.info(f"Failed run kept: its config and log are in '{run_dir / 'info'}'.")
        return False
    if not run_dir.is_dir():
        return False
    own = {f"{run_dir.name}{ext}" for ext in (".yaml", ".json", ".log")}
    for entry in run_dir.iterdir():
        allowed = own if entry.name == "info" else set()
        if (entry.name not in ("csv", "png", "info") or not entry.is_dir()
                or any(p.name not in allowed for p in entry.iterdir())):
            return False

    try:
        shutil.rmtree(run_dir)
    except OSError as e:
        # Never fatal: the run has already failed and reported why, and a folder
        # that could not be removed is untidy rather than wrong.
        log.warning(f"Could not remove the failed run's folder '{run_dir}': {e}")
        return False

    log.info(f"The failed run's folder '{run_dir}' was removed. "
             "Run with --keep-failed to keep its config and log.")
    return True
