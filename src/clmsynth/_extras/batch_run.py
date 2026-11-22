# SPDX-License-Identifier: MIT
# _extras/batch_run.py
"""Several configs as a batch, each run as its own clmsynth process, and the
configs a command line names, folders included."""

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Final

from .._input_sources import dataset_cache
from .._run_folders import is_run_folder
from .._utils.cli_logging import run_named_in, run_removed_in

__all__ = ["batch_configs", "run_batch"]

log: logging.Logger = logging.getLogger(__name__)

_CONFIG_FORMATS = {".json": "JSON", ".yaml": "YAML", ".yml": "YAML"}
# Seconds the batch waits for a run to end before it looks for a stop again: on
# Windows, a wait may not see a Ctrl-C until it returns.
_POLL: Final = 0.1


def _folder_configs(folder: Path) -> list[str]:
    """The configs directly inside `folder`, sorted by name. ValueError for a
    folder holding run folders, both JSON and YAML configs, or no config."""
    try:
        entries = sorted(folder.iterdir())
        runs = [e.name for e in entries if e.is_dir() and is_run_folder(e)]
        configs = [e for e in entries
                   if e.is_file() and e.suffix.lower() in _CONFIG_FORMATS]
    except OSError as e:
        raise ValueError(f"Cannot read the folder '{folder}': {e}") from None
    if runs:
        raise ValueError(f"The folder '{folder}' holds run folders ('{runs[0]}'): it is an "
                         "output folder, not a folder of configs.")
    if len({_CONFIG_FORMATS[e.suffix.lower()] for e in configs}) > 1:
        # A config and the file rendered from it would run the same settings twice.
        raise ValueError(f"The folder '{folder}' holds both JSON and YAML configs. Keep "
                         "one format in it, or name the configs to run.")
    if not configs:
        raise ValueError(f"The folder '{folder}' holds no config (.json, .yaml or .yml).")
    return [str(e) for e in configs]


def batch_configs(paths: list[str]) -> tuple[list[str], bool]:
    """The configs `paths` name, each folder replaced by its configs, and whether
    they run as a batch: a folder is given, or more than one distinct config. A
    config named twice, however its path is written, runs once, where it first
    appears. ValueError for a folder `_folder_configs` refuses."""
    configs: list[str] = []
    first: dict[str, str] = {}
    folder_given = False
    for path in paths:
        if Path(path).is_dir():
            folder_given = True
            found = _folder_configs(Path(path))
            log.info(f"Folder '{path}': {len(found)} config(s).")
        else:
            found = [path]
        for config in found:
            key = os.path.normcase(os.path.realpath(config))
            if key in first:
                same = ("is given more than once" if first[key] == config
                        else f"is the same config as '{first[key]}'")
                log.warning(f"'{config}' {same}; it runs once.")
                continue
            first[key] = config
            configs.append(config)
    return configs, folder_given or len(configs) > 1


def _run_config(config: str, flags: list[str], env: dict[str, str]) -> tuple[int, str]:
    """One config as its own clmsynth process: its exit status and its output."""
    try:
        # After "--", a config whose name starts with a dash is still a path. A byte
        # outside the batch's encoding is shown escaped.
        proc = subprocess.run([sys.executable, "-m", "clmsynth", *flags, "--", config],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              errors="backslashreplace", env=env, check=False)
    except Exception as e:
        # Such as a process that cannot be started: this config failed, the batch goes on.
        log.error(f"Could not run '{config}': {type(e).__name__}: {e}")
        return 1, ""
    # A child killed by a signal reports -N; a shell reports it as 128 + N.
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    return code, proc.stdout


def _batch_status(codes: list[int]) -> int:
    """A batch read like a run whose datasets are its configs: 0 when every config
    exited 0, 3 when some wrote their datasets and others did not, else 1 before
    2. Any status outside 0 to 3, a signal's 128 + N or a wrapper's, comes first."""
    foreign = [code for code in codes if not 0 <= code <= 3]
    if foreign:
        return max(foreign)
    if all(code == 0 for code in codes):
        return 0
    if any(code in (0, 3) for code in codes):
        return 3
    return 1 if 1 in codes else 2


class _StopOnSignal:
    """Ctrl-C and SIGTERM while a batch runs. The first is recorded with its exit
    status, so no further config starts, and puts back the handlers from before the
    batch: a second one quits the batch without waiting for the running configs."""

    def __init__(self) -> None:
        self.status = 0
        self._previous: dict[int, Any] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, self._record)

    def _record(self, signum: int, frame: object) -> None:
        self.status = 130 if signum == signal.SIGINT else 143
        self.restore()

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def run_batch(configs: list[str], jobs: int, render_plots: bool = False,
              keep_failed: bool = False, cache: bool = True,
              allow_network_paths: bool = False) -> int:
    """Each config in its own clmsynth process, `jobs` at a time, so each run keeps
    its own folder, log and exit status. A run's output is printed whole when it
    ends. A Ctrl-C or SIGTERM starts no further config and waits for the running
    ones; a second one quits. Returns the batch's exit status (`_batch_status`)."""
    # Every run is told: a run started on its own draws plots by default and does
    # not cache.
    flags = ["--viz" if render_plots else "--no-viz", "--cache" if cache else "--no-cache"]
    if keep_failed:
        flags.append("--keep-failed")
    if allow_network_paths:
        flags.append("--allow-network-paths")
    env = dict(os.environ)
    if jobs > 1:
        # Otherwise each worker may start a BLAS thread per core.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
    n = len(configs)
    plots = "with plots" if render_plots else "without plots (--viz draws them)"
    log.info(f"Batch of {n} config(s), {min(jobs, n)} at a time, {plots}.")
    if cache:
        try:
            log.info(f"Dataset cache: '{dataset_cache.cache_folder()}'; --no-cache fetches "
                     "every dataset again.")
        except RuntimeError as e:
            log.warning(f"Dataset cache: {e}. Set {dataset_cache.ENV_VAR} to a folder.")

    # Each ended config: its exit status, its run folder's name and whether that
    # folder was removed. Its output is printed when it ends and not kept.
    results: dict[int, tuple[int, str | None, bool]] = {}

    def report(i: int, code: int, output: str) -> None:
        results[i] = (code, run_named_in(output), run_removed_in(output))
        log.info(f"Config {i + 1} of {n} ended with exit {code}: {configs[i]}")
        sys.stderr.write(output)

    ended: queue.Queue[tuple[int, int, str]] = queue.Queue()

    def run_one(i: int) -> None:
        ended.put((i, *_run_config(configs[i], flags, env)))

    # Only this loop starts a config, so none starts once a stop is recorded. A
    # scheduler or Ctrl-C signals every process of the job: each run stops and
    # cleans up itself.
    stop = _StopOnSignal()
    next_config, running, failed, announced = 0, 0, False, False

    def schedule() -> None:
        nonlocal next_config, running, announced
        while True:
            while not (stop.status or failed) and next_config < n and running < jobs:
                # Daemon threads: a second signal quits without waiting for them.
                threading.Thread(target=run_one, args=(next_config,), daemon=True).start()
                next_config += 1
                running += 1
            if stop.status and not announced:
                log.warning("Stopping: no further config starts; waiting for the running ones.")
                announced = True
            if not running:
                return
            try:
                i, code, output = ended.get(timeout=_POLL)
            except queue.Empty:
                continue
            running -= 1
            report(i, code, output)

    try:
        try:
            schedule()
        except Exception:
            log.critical("Unexpected error in the batch: no further config starts; "
                         "waiting for the running ones.", exc_info=True)
            failed = True
            schedule()
            raise
    finally:
        stop.restore()

    for i, config in enumerate(configs):
        if i not in results:
            log.warning(f"{config}: not started")
            continue
        code, run, removed = results[i]
        if run is None:
            where = "no run folder"
        elif removed:
            where = f"run {run}, folder removed"
        else:
            where = f"run {run}"
        (log.info if code == 0 else log.warning)(f"{config}: exit {code}, {where}")
    codes = [code for code, _, _ in results.values()]
    status = _batch_status([*codes, stop.status] if stop.status else codes)
    n_ok = codes.count(0)
    log.info(f"Batch: {n_ok} of {n} config(s) exited 0; exit status {status}.")
    return status
