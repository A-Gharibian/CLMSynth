# SPDX-License-Identifier: MIT
# write_files.py
"""Files written whole or not at all, run outputs and the configs the guides write.
Also a path's absolute form and the file in its way, for messages."""

import contextlib
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

__all__ = [
    "file_in_the_way",
    "resolved_for_report",
    "write_all_then_replace",
    "write_text_atomic",
    "write_then_replace",
]

def write_all_then_replace(
        files: Sequence[tuple[str | Path, Callable[[Path], object] | str]]) -> None:
    """Writes each file to a temporary file beside its path, then os.replace swaps
    them in, only once all are written: a failure or an interrupt before that removes
    the temporary files and leaves every path as it was. A swap that fails also
    removes the files the earlier swaps created; one they replaced keeps its new
    content. A str is written as text, a callable is run on the temporary path.
    Symlinked paths are followed, so the links survive."""
    staged: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for path, content in files:
            target = Path(path).resolve()
            # A short name: the temporary path is no longer than nearly any path it stands for.
            tmp = target.with_name(f".{uuid.uuid4().hex[:8]}.tmp")
            staged.append((tmp, target))
            if isinstance(content, str):
                _write_text(tmp, content)
            else:
                content(tmp)
        for tmp, target in staged:
            new = not target.exists()
            os.replace(tmp, target)
            if new:
                created.append(target)
    except BaseException:
        for leftover in (*(tmp for tmp, _ in staged), *created):
            with contextlib.suppress(OSError):
                leftover.unlink(missing_ok=True)
        raise


def write_then_replace(path: str | Path, write: Callable[[Path], object]) -> None:
    """Runs `write` on a temporary file beside `path`, then os.replace swaps it in,
    so `path` never holds part of the new content. The temporary file is removed
    when `write` fails or is interrupted. A symlinked `path` is followed, so the
    link survives."""
    write_all_then_replace([(path, write)])


def _write_text(tmp: Path, text: str) -> None:
    # open(..., "x") rather than tempfile.mkstemp, which creates the file 0600:
    # the config would lose the read permission the umask gives everyone else.
    with open(tmp, "x", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())


def write_text_atomic(path: str | Path, text: str) -> None:
    """Writes `text` to `path` so the file holds either its old content or all of
    the new.

    The text goes to a temporary file in the same folder and os.replace swaps it
    in. A symlinked `path` is followed, so the link survives.
    """
    write_all_then_replace([(path, text)])


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
