# SPDX-License-Identifier: MIT
# dataset_cache.py
"""The dataset cache behind ``dataset_sources.fetch``: each dataset of a cached source
kept in one file and read back instead of fetched again. The folder is
``CLMSYNTH_CACHE_DIR`` when set, otherwise the user's cache folder. byoc and
fabricated_data are not cached."""

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from . import ENV_VAR, INCOMPLETE, NOT_CACHED, Source, engine_features

__all__ = ["ENV_VAR", "NOT_CACHED", "cache_folder", "fetch", "is_cached", "ready"]

log: logging.Logger = logging.getLogger(__name__)

# An entry of another format is fetched again and replaced.
_FORMAT: Final = 1
# Seconds between tries at an entry Windows refuses while another process swaps in a copy.
_WAITS: Final = (0.05, 0.1, 0.2, 0.4)


def cache_folder(environ: Mapping[str, str] | None = None,
                 platform: str | None = None) -> Path:
    """Where the entries are kept: ``CLMSYNTH_CACHE_DIR`` as it is when set, otherwise
    the user's cache folder. RuntimeError when no home folder can be found."""
    env = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if env.get(ENV_VAR):
        return Path(env[ENV_VAR]).expanduser()
    if platform == "win32":
        local = env.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "clmsynth" / "Cache"
    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / "clmsynth"
    xdg = env.get("XDG_CACHE_HOME", "")
    # The XDG specification ignores a relative path.
    return (Path(xdg) if os.path.isabs(xdg) else Path.home() / ".cache") / "clmsynth"


def _make(folder: Path) -> None:
    """Creates the folder; one clmsynth creates is the user's alone (0700 on POSIX).
    One that exists keeps its mode."""
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)


def ready(source: Source) -> bool:
    """Whether a run of `source` uses the cache, said in the log with the folder. False
    for a source that is not cached, and False with a warning when the folder cannot
    be made or read, so the run fetches as without it. A folder the run cannot write
    is only read."""
    if source.key in NOT_CACHED:
        return False
    try:
        folder = cache_folder()
    except RuntimeError as e:
        return _off(e)
    try:
        _make(folder)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as e:
        return _read_only(folder, e) or _off(e)
    log.info(f"Dataset cache: '{folder}'.")
    return True


def _off(error: Exception) -> bool:
    """False, with the warning that the run fetches without the cache."""
    log.warning(f"The dataset cache is off for this run: {error}. Set {ENV_VAR} to a folder "
                "you can write.")
    return False


def _read_only(folder: Path, error: OSError) -> bool:
    """Whether the entries of a folder the run cannot write can still be read; said in
    the log when they can."""
    # An entry is opened by its name, which needs search permission on the folder only.
    if not (os.path.isdir(folder) and os.access(folder, os.X_OK)):
        return False
    log.info(f"Dataset cache: '{folder}', which this run cannot write "
             f"({error.strerror or error}): its entries are read, and none is kept.")
    return True


def _plain(value: Any) -> bool:
    """Whether `value` is data as a config file gives it. A tuple, an array or a numpy
    value is not: json writes it like a list or a number, and a source may read it
    otherwise."""
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _plain(v) for k, v in value.items())
    return False


def _key(source: Source, battery: Any, dataset: Any, seed: Any,
         kwargs: Mapping[str, Any]) -> str | None:
    """What makes an entry, as one string: the source, battery and dataset, the suite's
    other keys, and the seed of a seeded source. None when a value is not plain data."""
    identity = {"source": source.key, "battery": battery, "dataset": dataset,
                "seed": seed if source.seeded else None, "suite": dict(kwargs)}
    if not _plain(identity):
        return None
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _safe(name: Any) -> str:
    """A name as part of a file name: letters, digits, '.', '_' and '-', at most 40."""
    return re.sub(r"[^\w.-]", "_", str(name), flags=re.ASCII)[:40]


def _entry(source: Source, battery: Any, dataset: Any, seed: Any,
           kwargs: Mapping[str, Any]) -> tuple[Path, str] | str:
    """The entry's file and its key, or why the dataset has none."""
    if source.key in NOT_CACHED:
        return f"'{source.key}' is not cached"
    key = _key(source, battery, dataset, seed, kwargs)
    if key is None:
        return f"a setting of {source.key}_suite is not plain data"
    try:
        folder = cache_folder()
    except RuntimeError as e:
        return f"{e}. Set {ENV_VAR} to a folder you can write"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    name = "-".join(_safe(part) for part in (source.key, battery, dataset))
    return folder / f"{name}-{digest}.npz", key


def _installed(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _requirements(package: str) -> list[str]:
    """The packages `package` needs to run, by name; none when it is not installed."""
    try:
        listed = metadata.requires(package) or []
    except metadata.PackageNotFoundError:
        return []
    names = []
    for requirement in listed:
        spec, _, marker = requirement.partition(";")
        name = re.match(r"[A-Za-z0-9._-]+", spec.strip())
        if name and "extra" not in marker:
            names.append(name.group(0).lower())
    return names


def _versions(source: Source) -> dict[str, str | None]:
    """The versions an entry is valid for: clmsynth, numpy, pandas, and the package
    the source imports with the packages it needs (mdcgenpy's rotations are scipy's)."""
    from .. import __version__

    found: dict[str, str | None] = {
        "clmsynth": __version__, "numpy": np.__version__, "pandas": pd.__version__}
    if source.needs is not None:
        for package in (source.needs, *_requirements(source.needs)):
            found.setdefault(package, _installed(package))
    return found


def _current(meta: Mapping[str, Any], key: str, source: Source) -> bool:
    """Whether a stored record is this dataset's, kept by the versions installed now."""
    return (meta.get("format") == _FORMAT and meta.get("key") == key
            and meta.get("versions") == _versions(source))


def _frame(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> pd.DataFrame:
    """The frame built as the sources build theirs: the leading columns from one 2-D
    array, each later column added on its own."""
    columns, lead = meta["columns"], meta["lead"]
    frame = pd.DataFrame(arrays["lead"], columns=columns[:lead])
    for i in range(lead, len(columns)):
        frame[columns[i]] = arrays[f"c{i}"]
    start, stop, step = meta["index"]
    if (start, stop, step) != (0, len(frame), 1):
        frame.index = pd.RangeIndex(start, stop, step)
    frame.attrs = dict(meta["attrs"])
    return frame


def _feature_order(frame: pd.DataFrame) -> tuple[bool, bool]:
    """How the features the engine reads lie in memory. Summing rows in C or F order
    rounds differently, and a centroid moves a placement."""
    coords = engine_features(frame).to_numpy()
    return coords.flags.c_contiguous, coords.flags.f_contiguous


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Equal values, names, types, index and attrs, and features in the same order."""
    return (a.equals(b) and list(a.columns) == list(b.columns)
            and list(a.dtypes) == list(b.dtypes) and a.index.equals(b.index)
            and a.attrs == b.attrs and _feature_order(a) == _feature_order(b))


def _stored(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]] | str:
    """The arrays and the record an entry keeps for `frame`, or why it cannot keep the
    frame exactly. Nothing that needs pickle: an entry is read with allow_pickle=False."""
    columns = list(frame.columns)
    if (not columns or not all(isinstance(c, str) for c in columns)
            or len(set(columns)) < len(columns) or frame.columns.name is not None):
        return "its columns are not distinct names"
    index = frame.index
    if not isinstance(index, pd.RangeIndex) or index.name is not None:
        return "its index is not a plain range"
    dtypes = list(frame.dtypes)
    if not all(isinstance(t, np.dtype) and t.kind in "biuf" for t in dtypes):
        return "a column holds something other than numbers"
    try:
        attrs = json.loads(json.dumps(frame.attrs, allow_nan=False))
    except (TypeError, ValueError):
        attrs = None
    if attrs != frame.attrs:
        return "its attrs are not plain data"
    lead = next((i for i, t in enumerate(dtypes) if t != dtypes[0]), len(dtypes))
    arrays = {"lead": np.ascontiguousarray(frame.iloc[:, :lead].to_numpy())}
    arrays |= {f"c{i}": frame.iloc[:, i].to_numpy() for i in range(lead, len(columns))}
    meta = {"columns": columns, "lead": lead,
            "index": [index.start, index.stop, index.step], "attrs": attrs}
    try:
        same = _same(_frame(arrays, meta), frame)
    except Exception:
        same = False
    if not same:
        return "it would not read back as the same frame"
    return arrays, meta


def _read(path: Path, key: str, source: Source, where: str) -> pd.DataFrame | None:
    """The entry at `path` when it is this dataset's and current, else None. One that
    cannot be read is reported, and replaced once the dataset is fetched; a link is
    never followed."""
    try:
        if path.is_symlink():
            log.warning(f"'{path}' is a link, not a cache entry: '{where}' is fetched again "
                        "and the link replaced.")
            return None
        with _open(path) as entry:
            meta = json.loads(entry["meta"].item())
            if not _current(meta, key, source):
                log.info(f"The cached '{where}' was kept by other versions; fetching it again.")
                return None
            arrays = {name: entry[name] for name in entry.files if name != "meta"}
        frame = _frame(arrays, meta)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except Exception as e:
        log.warning(f"Cannot read the cached '{where}' ('{path}'): {type(e).__name__}: {e}. "
                    "Fetching it again.")
        return None
    log.info(f"'{where}' read from the dataset cache: '{path}'.")
    return frame


def _open(path: Path) -> Any:
    """The entry opened with np.load. On Windows, one refused while another process
    swaps in a new copy is tried again before it counts as unreadable."""
    try:
        return np.load(path, allow_pickle=False)
    except PermissionError:
        if sys.platform == "win32":
            return _open_when_free(path)
        else:
            raise


def _open_when_free(path: Path) -> Any:
    """np.load tried again after each pause in _WAITS, 0.75 s in all; the last refusal
    is raised."""
    for wait in _WAITS[:-1]:
        time.sleep(wait)
        with contextlib.suppress(PermissionError):
            return np.load(path, allow_pickle=False)
    time.sleep(_WAITS[-1])
    return np.load(path, allow_pickle=False)


def _replace(path: Path, members: Mapping[str, Any]) -> None:
    """Writes the entry to a new file beside `path` and swaps it in. The swap replaces
    a link at `path` itself: it never writes where a link points."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "xb") as file:
            np.savez_compressed(file, **members)
            file.flush()
            os.fsync(file.fileno())
        _swap(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _swap(tmp: Path, path: Path) -> None:
    """os.replace. On Windows, one refused while another process reads or swaps the
    entry is tried again before it counts as failed."""
    try:
        os.replace(tmp, path)
    except PermissionError:
        if sys.platform == "win32":
            _swap_when_free(tmp, path)
        else:
            raise


def _swap_when_free(tmp: Path, path: Path) -> None:
    """os.replace tried again after each pause in _WAITS, 0.75 s in all; the last
    refusal is raised."""
    for wait in _WAITS[:-1]:
        time.sleep(wait)
        with contextlib.suppress(PermissionError):
            os.replace(tmp, path)
            return
    time.sleep(_WAITS[-1])
    os.replace(tmp, path)


def _write(path: Path, key: str, source: Source, frame: pd.DataFrame, where: str) -> None:
    """Keeps `frame` at `path`, written whole to a file beside it and then swapped in.
    A failure is reported and never stops the fetch. A folder that exists and cannot be
    written keeps nothing, said without a warning."""
    if os.path.isdir(path.parent) and not os.access(path.parent, os.W_OK | os.X_OK):
        log.info(f"'{where}' is not cached: the cache folder '{path.parent}' cannot be "
                 "written.")
        return
    stored = _stored(frame)
    if isinstance(stored, str):
        log.info(f"'{where}' is not cached: {stored}.")
        return
    arrays, meta = stored
    meta.update(format=_FORMAT, key=key, versions=_versions(source))
    try:
        _make(path.parent)
        _replace(path, {**arrays, "meta": np.array(json.dumps(meta))})
    except Exception as e:
        log.warning(f"Could not keep '{where}' in the dataset cache: {type(e).__name__}: {e}")
        return
    log.info(f"'{where}' kept in the dataset cache: '{path}'.")


class _Warnings(logging.Handler):
    """Counts the warnings the fetching thread logs while a dataset is fetched."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0
        self._thread = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        # Another thread's warning is about another dataset. None: threads not recorded.
        if record.thread in (self._thread, None):
            self.count += 1


def fetch(source: Source, battery: Any, dataset: Any, seed: Any,
          kwargs: Mapping[str, Any]) -> pd.DataFrame | None:
    """`source.fetch` through the cache: the dataset's entry when it is there and
    current, otherwise fetched and kept. A fetch that its source marks incomplete
    (`INCOMPLETE`), or that logs a warning, is not kept, so it is fetched, and warns,
    again. A cache that cannot be
    read or written is reported, and the dataset is fetched as without it."""
    where = f"{battery}/{dataset}"
    entry = _entry(source, battery, dataset, seed, kwargs)
    if isinstance(entry, str):
        if source.key not in NOT_CACHED:
            log.info(f"'{where}' is not cached: {entry}.")
        return source.fetch(battery, dataset, seed, **kwargs)
    path, key = entry
    frame = _read(path, key, source, where)
    if frame is not None:
        return frame
    warnings, package = _Warnings(), logging.getLogger("clmsynth")
    package.addHandler(warnings)
    try:
        frame = source.fetch(battery, dataset, seed, **kwargs)
    finally:
        package.removeHandler(warnings)
    if frame is not None and frame.attrs.get(INCOMPLETE):
        log.info(f"'{where}' is not cached: its fetch is incomplete.")
    elif frame is not None and warnings.count:
        log.info(f"'{where}' is not cached: its fetch logged a warning.")
    elif frame is not None:
        _write(path, key, source, frame, where)
    return frame


def is_cached(source: Source, battery: Any, dataset: Any, seed: Any,
              kwargs: Mapping[str, Any]) -> bool:
    """Whether the dataset has a current entry, found without fetching anything."""
    entry = _entry(source, battery, dataset, seed, kwargs)
    if isinstance(entry, str):
        return False
    path, key = entry
    try:
        if path.is_symlink():
            return False
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(stored["meta"].item())
    except Exception:
        return False
    return isinstance(meta, Mapping) and _current(meta, key, source)
