# SPDX-License-Identifier: MIT
# _extras/batch_run.py
"""Several configs as a batch, each run as its own clmsynth process, and the
configs a command line names, folders included."""

import logging
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .._utils.signals import Terminated, raise_terminated

__all__ = ["batch_configs", "run_batch"]

log: logging.Logger = logging.getLogger(__name__)

_CONFIG_FORMATS = {".json": "JSON", ".yaml": "YAML", ".yml": "YAML"}


def _is_run_folder(path: Path) -> bool:
    return all((path / sub).is_dir() for sub in ("csv", "png", "info"))


def _folder_configs(folder: Path) -> list[str]:
    """The configs directly inside `folder`, sorted by name. ValueError for a
    folder holding run folders, both JSON and YAML configs, or no config."""
    try:
        entries = sorted(folder.iterdir())
        runs = [e.name for e in entries if e.is_dir() and _is_run_folder(e)]
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
    proc = subprocess.run([sys.executable, "-m", "clmsynth.main", config, *flags],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          env=env, check=False)
    # A child killed by a signal reports -N; a shell reports it as 128 + N.
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    return code, proc.stdout


_RUN_LINE = re.compile(r" - INFO - Run (\S+) from config ")

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


def run_batch(configs: list[str], jobs: int, flags: list[str]) -> int:
    """Each config in its own clmsynth process, `jobs` at a time, so each run keeps
    its own folder, log and exit status. A run's output is printed whole when it
    ends. Returns the batch's exit status (`_batch_status`)."""
    env = dict(os.environ)
    if jobs > 1:
        # Otherwise each worker may start a BLAS thread per core.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
    n = len(configs)
    plots = "with plots" if "--viz" in flags else "without plots (--viz draws them)"
    log.info(f"Batch of {n} config(s), {min(jobs, n)} at a time, {plots}.")

    results: dict[int, tuple[int, str]] = {}

    def report(i: int, code: int, output: str) -> None:
        results[i] = (code, output)
        log.info(f"Config {i + 1} of {n} ended with exit {code}: {configs[i]}")
        sys.stderr.write(output)

    stopped = 0
    pool = ThreadPoolExecutor(max_workers=jobs)
    futures = {pool.submit(_run_config, config, flags, env): i
               for i, config in enumerate(configs)}
    # A scheduler or Ctrl-C signals every process of the job: each run stops and
    # cleans up itself, and the batch starts nothing new.
    previous_sigterm = signal.signal(signal.SIGTERM, raise_terminated)
    try:
        for future in as_completed(futures):
            report(futures[future], *future.result())
    except KeyboardInterrupt:
        stopped = 130
    except Terminated:
        stopped = 143
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    if stopped:
        log.warning("Stopping: no further config starts; waiting for the running ones.")
    pool.shutdown(wait=True, cancel_futures=True)
    for future, i in futures.items():
        if i not in results and future.done() and not future.cancelled():
            report(i, *future.result())

    for i, config in enumerate(configs):
        if i not in results:
            log.warning(f"{config}: not started")
            continue
        code, output = results[i]
        run = _RUN_LINE.search(output)
        where = f"run {run.group(1)}" if run else "no run folder"
        (log.info if code == 0 else log.warning)(f"{config}: exit {code}, {where}")
    codes = [code for code, _ in results.values()]
    status = _batch_status([*codes, stopped] if stopped else codes)
    n_ok = codes.count(0)
    log.info(f"Batch: {n_ok} of {n} config(s) exited 0; exit status {status}.")
    return status
