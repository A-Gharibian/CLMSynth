# SPDX-License-Identifier: MIT
# main.py
"""Pipeline entry point.

Reads a config file (default test_data_config.yaml), fetches each requested
dataset from the configured source, generates the CLM labels, and writes one
timestamped run folder with the CSV, plots, and a metrics summary per dataset,
and in info/ the config as the run read it (resolved_config.py) and the run's log.
Several config files run as a batch, each in its own process.
"""

import argparse
import difflib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from . import __version__
from ._cli_logging import LOG_FORMAT, SingleLineFilter, configure_cli_logging
from ._compat import import_optional_dependency
from .byoc_source import fetch_byoc_data
from .clm_errors import InfeasibleAllocationError, MissingConfigKey
from .clm_label_engine import (
    _check_config,
    pair_reachable_range,
    validate_matching_ids,
)
from .dataset_sources import (
    fetch_clustbench_data,
    fetch_fabricated_data,
    fetch_mdcgen_data,
    get_datasets_for_battery,
    is_heavy,
    print_battery_info,
    resolve_selection,
    resolved_for_report,
)
from .generate_config import _write_then_replace, write_text_atomic
from .label_context import DatasetContext, build_context
from .label_generator import _require_clm_config, generate_additional_labels
from .metrics import clustering_ari, clustering_mcc
from .resolved_config import (
    DATA_SOURCE,
    N_LABELS,
    OUTPUT_DIR,
    SOURCE_LABELING,
    _is_int,
    plain,
    resolve_config,
)

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

FETCHERS: Final[Mapping[str, Callable[..., pd.DataFrame | None]]] = {
    "clustbench": lambda group, name, seed, **kw: fetch_clustbench_data(dataset_group=group, dataset_name=name, **kw),
    "mdcgen": lambda group, name, seed, **kw: fetch_mdcgen_data(dataset_group=group, dataset_name=name, seed=seed, **kw),
    "fabricated_data": lambda group, name, seed, **kw: fetch_fabricated_data(dataset_group=group, dataset_name=name, seed=seed, **kw),
    "byoc": lambda group, name, seed, **kw: fetch_byoc_data(dataset_group=group, dataset_name=name, seed=seed, **kw),
}
# Sources whose data depend on the suite seed. clustbench downloads fixed data
# and byoc reads a file, so the seed reaches neither.
SEEDED_SOURCES = ("mdcgen", "fabricated_data")

# The <source>_suite keys run_pipeline takes for every source, and those each
# fetcher reads. A key outside them is refused: byoc and fabricated_data would
# run a misspelt key on the default it meant to change, and clustbench would
# fail every dataset on it. mdcgen is not listed: its other keys go on to
# mdcgenpy.
SUITE_KEYS = ("batteries", "datasets", "seed")
FETCHER_SUITE_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "clustbench": ("base_url",),
    "byoc": ("input_dir", "cluster_column", "tag_columns", "standardize"),
    "fabricated_data": ("n_samples",),
}

# Codes judged against one dataset's K, ids or features: that dataset is
# skipped and the run goes on.
PER_DATASET_CODES = (102, 105, 119, 125, 127, 133)

# Display names for the run folder, plot titles and summaries, capitalized apart
# from the data_source keys. Clustbench is the package that fetches; its
# datasets come from many authors.
SOURCE_DISPLAY: Final[Mapping[str, str]] = {
    "clustbench": "Clustbench",
    "mdcgen": "MDCGen",
    "fabricated_data": "Fabricated",
    "byoc": "BYOC",
}


def _write_summary_txt(path: Path, friendly: str, source: str, battery: str, dataset: str,
                       source_labeling: str, gt_col, n_rows: int, clm_config, label_results,
                       seed_lines: list[str], label_seed: int,
                       pair_lines: list[str] | None = None) -> None:
    """One txt per dataset: the configuration used, the seeds that reproduce it
    (a drawn seed is otherwise only in the log), the reachable pair MCC under
    scope 'pair', the MCC/ARI shown on the plots, and per label what each
    assignment_matrix row and competing_noise entry placed and what target_metric
    reached."""
    gt_disp = gt_col if gt_col else "(none)"
    out = [
        f"Source : {friendly} ({source})",
        f"Dataset: {battery}/{dataset}",
        f"Rows   : {n_rows}",
        f"Ground-truth labeling: {source_labeling}  ->  {gt_disp}",
        "",
        "===== Seeds =====",
        *seed_lines,
        "",
        "===== CLM label configuration =====",
        # The standard library, so the summary reads the same whatever is installed.
        json.dumps(plain(clm_config), indent=2, ensure_ascii=False, default=str)
        if clm_config else "(legacy noise mode, no clm_label config)",
        "",
    ]
    if pair_lines:
        out += [*pair_lines, ""]
    out.append("===== Results (identical to each plot's subtitle) =====")
    if not label_results:
        out.append("(no generated labels)")
    # Label i was generated with seed + i (generate_additional_labels).
    for i, r in enumerate(label_results):
        if r["mcc"] is not None:
            out.append(f"{r['name']}:  MCC = {r['mcc']:.3f}   ARI = {r['ari']:.3f}   (vs {gt_disp})")
        else:
            out.append(f"{r['name']}:  (no ground-truth labeling to compare against)")
        out.append(f"    seed = {label_seed + i}")
        out.append(f"    label counts = {r['counts']}")
        report = r.get("report") or {}
        out += [f"    {line}" for line in _placed_lines(report)]
        if "type" in report:
            out += [f"    {line}" for line in _target_lines(report)]

    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, "\n".join(out) + "\n")
    log.info(f"Saved summary: {path}")

def _is_plain_name(name) -> bool:
    """True when `name` is a bare name rather than anything path-shaped.

    Battery and dataset names are used to READ (byoc resolves
    `input_dir/<dataset>.csv`) and to WRITE (`csv/<source>__<battery>__<dataset>.csv`),
    so a separator or a `..` in one reaches outside the folders the configuration
    declared. The registry sources filter their names against a known list and are
    safe by construction.
    """
    text = str(name)
    if not text or text in (".", ".."):
        return False
    # A separator or a drive letter is what every absolute or traversing form has
    # in common, on both POSIX and Windows.
    return not ("/" in text or "\\" in text or ":" in text)


def _drop_path_shaped_names(jobs: list) -> list:
    """Filter out jobs whose battery or dataset name is not a plain name."""
    kept = []
    for battery, dataset in jobs:
        offenders = [repr(n) for n in (battery, dataset) if not _is_plain_name(n)]
        if offenders:
            log.error(
                f"Skipping {battery}/{dataset}: {', '.join(offenders)} is not a plain "
                "name. Battery and dataset names are used to build file paths, so a "
                "name containing '/', '\\', ':' or '..' would read or write outside "
                "the configured input and output folders. Give the file stem only, "
                "without any directory part."
            )
            continue
        kept.append((battery, dataset))
    return kept


def _byoc_cluster_ids(input_dir, dataset: str, cluster_column: str):
    """
    Returns the cluster ids, or None when the file cannot be read or its ids are
    not integers, which the import refuses ([CLM-133]).
    """
    path = (Path(input_dir) / f"{dataset}.csv") if input_dir else Path(f"{dataset}.csv")
    try:
        column = pd.read_csv(path, usecols=[cluster_column])[cluster_column]
    except Exception:
        return None
    if not pd.api.types.is_integer_dtype(column):
        return None
    return sorted(column.unique().tolist())


def _configured_cluster_ids(clm_config: dict) -> list:
    """Cluster ids the configuration itself names."""
    mode = clm_config.get("matching_mode")
    # A block of the wrong shape names no ids here; the engine refuses it ([CLM-141]).
    if mode == "single":
        sm = clm_config.get("single_match") or {}
        return [sm["cluster"]] if isinstance(sm, Mapping) and "cluster" in sm else []
    if mode == "custom":
        # A clusters value that is not a list is left to [CLM-141] in _check_pair.
        matrix = clm_config.get("assignment_matrix")
        return [k for row in (matrix if isinstance(matrix, (list, tuple)) else [])
                if isinstance(row, dict) and isinstance(row.get("clusters"), (list, tuple))
                for k in row["clusters"]]
    return []


def precheck_byoc_matching_ids(jobs: list[tuple[str, str]], fetch_kwargs: dict[str, Any],
                               clm_config: dict[str, Any] | None) -> None:
    """Resolve [CLM-105] for every BYOC dataset before any work begins.
    [CLM-104] is checked once, before the loop.
    Raises the first offending dataset's coded error, after logging every one of
    them, so a batch with several mismatches is fixed in one pass rather than
    one run at a time.
    """
    # A clm_label that is not a mapping is left to the engine ([CLM-141]).
    if not clm_config or not isinstance(clm_config, Mapping):
        return

    # [CLM-104] bounds labels by num_classes: dataset-independent, so it is
    # checked before anything that depends on a dataset.
    try:
        validate_matching_ids(clm_config, _configured_cluster_ids(clm_config))
    except (ValueError, TypeError) as e:
        log.critical(f"Configuration error, aborting run: {e}")
        raise

    cluster_column = fetch_kwargs.get("cluster_column")
    if not isinstance(cluster_column, str):
        return                       # byoc_source rejects a bad cluster_column itself

    failures = []
    for battery, dataset in jobs:
        ids = _byoc_cluster_ids(fetch_kwargs.get("input_dir"), dataset, cluster_column)
        if ids is None:
            continue
        try:
            validate_matching_ids(clm_config, ids)
        except ValueError as e:
            failures.append((f"{battery}/{dataset}", e))

    if not failures:
        return
    for tag, error in failures:
        log.critical(f"{tag}: {error}")
    log.critical(
        f"{len(failures)} of {len(jobs)} BYOC dataset(s) do not contain the cluster "
        "ids this configuration matches on. Aborting before any output is written; "
        "every offending dataset is listed above."
    )
    raise failures[0][1]


def _not_a_mapping(key: str, section: object, example: str) -> str | None:
    """The message for a config section that is not a mapping, or None."""
    if isinstance(section, Mapping):
        return None
    return f"'{key}' must be a mapping of settings such as '{example}', got {section!r}."


def _is_json(path: str | Path) -> bool:
    """A .json config is read and written with the standard library, any other
    with PyYAML, which is optional."""
    return Path(path).suffix.lower() == ".json"


def load_config(config_path: str) -> dict[str, Any]:
    """Loads the pipeline config; exits 2 with a message if absent, malformed or
    empty, and 1 if a YAML config finds no PyYAML."""
    path = Path(config_path)
    if not path.is_file():
        log.critical(f"Configuration file not found at '{config_path}'. "
                     f"Run clmsynth-wizard or clmsynth-config first.")
        sys.exit(2)
    # utf-8-sig drops a BOM (Notepad, PowerShell 5.1), as PyYAML does and json does not.
    text = path.read_text(encoding="utf-8-sig")
    # Each parser's message carries the line and column of the mistake.
    if _is_json(path):
        try:
            config = json.loads(text) if text.strip() else None
        except json.JSONDecodeError as e:
            log.critical(f"Configuration file '{config_path}' is not valid JSON: {e}")
            sys.exit(2)
    else:
        try:
            yaml = import_optional_dependency("yaml", f"Reading {config_path!r}")
        except ImportError as e:
            log.critical(f"{e}. A .json config needs nothing more.")
            sys.exit(1)
        try:
            config = yaml.safe_load(text)
        except yaml.YAMLError as e:
            log.critical(f"Configuration file '{config_path}' is not valid YAML: {e}")
            sys.exit(2)
    # An empty file parses to None; anything non-mapping means main()'s config.get(...)
    if not isinstance(config, dict):
        log.critical(f"Configuration file '{config_path}' is empty or not a mapping of "
                     "settings.")
        sys.exit(2)
    return config


_RESOLVED_HEADER = (
    "# The configuration this run read, written by clmsynth {version}: every key with\n"
    "# its value or its default, seeds as numbers, and the datasets that ran. Keys the\n"
    "# run never read, or read and ignored, are left out. Running this file repeats the run.\n"
)


def _write_resolved_config(path: Path, resolved: Mapping[str, Any]) -> None:
    """Written in the format of `path`, as load_config reads it. Best-effort, like
    the plots: the CSVs are the deliverable, so a record that cannot be written is
    reported and the run goes on."""
    data = plain(resolved)
    try:
        if _is_json(path):
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        else:
            yaml = import_optional_dependency("yaml", "Writing it")
            text = _RESOLVED_HEADER.format(version=__version__) + yaml.safe_dump(
                data, sort_keys=False, default_flow_style=False, allow_unicode=True)
        write_text_atomic(path, text)
    except Exception as e:
        # A full disk, PyYAML missing, or a value no writer represents, which a
        # dict run can pass.
        log.error(f"Could not write the resolved config '{path}': {e}")
        return
    log.info(f"Saved resolved config: {path}")


def _resolve_jobs(source: str, batteries: list[str], datasets_cfg, fetch_kwargs: dict) -> list:
    """(battery, dataset) pairs to run."""
    if any(is_heavy(source, b) for b in batteries):
        log.warning(f"Selection includes a heavy '{source}' battery: expect long fetch/generation times.")

    jobs = _drop_path_shaped_names(
        [(b, d) for b in batteries for d in get_datasets_for_battery(source, b, datasets_cfg)])
    log.info(f"Resolved {len(jobs)} dataset(s) across {len(batteries)} batter(y/ies) from source '{source}'.")
    if fetch_kwargs:
        log.info(f"Extra fetch kwargs forwarded to '{source}': {fetch_kwargs}")

    if source == "byoc":
        log.info(f"BYOC input directory: "
                 f"'{resolved_for_report(fetch_kwargs.get('input_dir', '.'))}'.")
    return jobs


def _precheck_clm_config(clm_config: Any) -> None:
    """The labelling's checks on clm_label that need no dataset, run before any
    dataset is fetched. A refusal stops the run as it would at the first dataset;
    its warnings come with each label. An uncoded error is left to the per-dataset
    handling, which reports it."""
    try:
        _require_clm_config(clm_config)
        _check_config(clm_config, warn=False)
    except Exception as e:
        if getattr(e, "code", None) is None:
            return
        log.critical(f"Configuration error, aborting run: {e}")
        raise


def _label_dataset(context: DatasetContext, battery: str, dataset: str,
                   label_cfg: Mapping[str, Any], n_add_labels: int, source_labeling: str,
                   clm_config: dict | None, seed: int) -> None:
    """Labels stop at the first that fails for this dataset; the ones before it
    stay attached."""
    error: Exception
    try:
        generate_additional_labels(
            context, n_labels=n_add_labels, source_labeling=source_labeling,
            clm_config=clm_config, noise=label_cfg.get("noise", 0.1), seed=seed,
        )
        return
    except MissingConfigKey:
        raise
    except (KeyError, InfeasibleAllocationError) as e:
        error = e
    except ValueError as e:
        # [CLM-120] is judged against THIS dataset's cluster sizes: the same
        # config can fit the next dataset, so it skips the label, not the run.
        if getattr(e, "code", None) != 120:
            raise
        error = e
    # Each label has its own seed, so a later one can fail where the first fitted.
    written = len(context.generated_labels)
    after = f" after {written} of {n_add_labels} labels" if written else ""
    log.error(f"Skipping label generation for {battery}/{dataset}{after}: {error}")


def _score_labels(final_df: pd.DataFrame, context: DatasetContext,
                  gt_col: str | None) -> list[dict[str, Any]]:
    """MCC, ARI and counts per generated label."""
    label_results: list[dict[str, Any]] = []
    for label_name in context.generated_labels:
        mcc = ari = None
        if gt_col is not None:
            mcc = float(clustering_mcc(final_df[gt_col], final_df[label_name]))
            ari = float(clustering_ari(final_df[gt_col], final_df[label_name]))
        counts = final_df[label_name].value_counts().sort_index().to_dict()
        label_results.append({"name": label_name, "mcc": mcc, "ari": ari, "counts": counts,
                              "report": context.label_reports.get(label_name)})
    return label_results


def _placed_lines(report: Mapping[str, Any]) -> list[str]:
    """What each assignment_matrix row and competing_noise entry placed, for the
    summary txt, beside the points it asks for. Under target_metric scope 'global'
    the rows ask at the solved level, alpha."""
    lines = []
    if report.get("assignment_matrix"):
        lines.append("assignment_matrix placed:")
        for i, row in enumerate(report["assignment_matrix"]):
            level = (f"alpha {report['alpha']:.4f}" if "alpha" in report
                     else f"recall_target {row['recall_target']}")
            lines.append(f"  row {i}: clusters {row['clusters']} -> label {row['label']}, "
                         f"{level}: {row['placed']} points (asks {round(row['asks'], 2)})")
    if report.get("competing_noise"):
        lines.append("competing_noise placed:")
        for i, entry in enumerate(report["competing_noise"]):
            lines.append(f"  entry {i}: cluster {entry['cluster']} -> label {entry['label']}, "
                         f"share {entry['share']}: {entry['placed']} points "
                         f"(asks {round(entry['asks'], 2)})")
    return lines


def _target_lines(report: Mapping[str, Any]) -> list[str]:
    """What a label's target_metric run reported, for the summary txt: the
    request, the delivered value, and under scope 'global' the highest value the
    search found and the size ceiling of the label counts."""
    kind = report["type"]
    lines = [(f"target_metric = {kind} {report['requested']} requested, "
              f"{report['achieved']:.4f} achieved (scope {report['scope']})")]
    if report["scope"] != "global":
        return lines
    lines.append(f"highest found = {report['highest']:.4f} (alpha {report['highest_alpha']:.4f})")
    ceiling = report["ceiling"]
    if ceiling is None:
        lines.append("size ceiling  = none: the label counts can move (spillover_rule "
                     "'uniform'/'concentrated')")
    elif kind == "mcc":
        lines.append(f"size ceiling  = {ceiling:.4f} (no labeling with these label counts "
                     "scores higher)")
    else:
        lines.append(f"size bound    = {ceiling:.4f} (no labeling with these label counts "
                     "scores higher; not always reached)")
    return lines


def _pair_range_lines(clm_config, final_df: pd.DataFrame, gt_col: str | None) -> list[str]:
    """The reachable pair MCC under target_metric scope 'pair', for the summary
    txt, under its own heading.

    It depends only on N and the size of the single_match cluster, so it is
    reported whether or not the label was generated or met its target.
    The plot box leaves it out.
    """
    if not isinstance(clm_config, Mapping) or gt_col is None:
        return []
    tm, sm = clm_config.get("target_metric"), clm_config.get("single_match")
    if (clm_config.get("matching_mode") != "single" or not isinstance(tm, Mapping)
            or tm.get("scope", "global") != "pair" or tm.get("type") != "mcc"
            or not isinstance(sm, Mapping) or "cluster" not in sm
            or isinstance(sm["cluster"], bool)):
        return []
    try:
        n_k = int((final_df[gt_col] == sm["cluster"]).sum())
    except (TypeError, ValueError):  # an id of the wrong shape, refused as [CLM-141]
        return []
    if n_k < 1:
        return []
    phi_min, phi_max = pair_reachable_range(len(final_df), n_k)
    line = (f"cluster {sm['cluster']}, label {sm.get('label')}: "
            f"[{phi_min:.4f}, {phi_max:.4f}]   (N = {len(final_df)}, cluster size = {n_k})")
    return ["===== Reachable pair MCC (target_metric scope: pair) =====", line]


def _write_dataset(context: DatasetContext, source: str, battery: str, dataset: str,
                   source_labeling: str, clm_config: dict | None,
                   csv_dir: Path, png_dir: Path, txt_dir: Path,
                   render_plots: bool, seed_lines: list[str], label_seed: int) -> None:
    """CSV, summary and plots for one dataset."""
    final_df = context.to_dataframe()
    stem = f"{source}-{battery}-{dataset}"
    csv_path = csv_dir / f"{stem}.csv"
    # Stopped mid-write, a run leaves no partial CSV under the final name.
    _write_then_replace(csv_path, lambda tmp: final_df.to_csv(tmp, index=False))
    log.info(f"Saved: {csv_path}")

    friendly = SOURCE_DISPLAY.get(source, source)
    plot_title = f"{friendly}: {battery}/{dataset}"
    gt_col = (context.gt_column_name(source_labeling)
              if source_labeling in context.ground_truths else None)

    label_results = _score_labels(final_df, context, gt_col)

    _write_summary_txt(txt_dir / f"{stem}.txt", friendly, source, battery, dataset,
                       source_labeling, gt_col, len(final_df), clm_config, label_results,
                       seed_lines, label_seed,
                       pair_lines=_pair_range_lines(clm_config, final_df, gt_col))

    if render_plots:
        # Deferred so --no-viz never imports matplotlib.
        from ._visualization import render_dataset_plots

        render_dataset_plots(final_df, list(context.features.columns), png_dir, stem,
                             gt_col, source_labeling, plot_title, label_results,
                             clm_config)


def _aborts_run(e: ValueError, battery: str, dataset: str) -> bool:
    """Log the error; True for configuration-wide errors."""
    code = getattr(e, "code", None)
    if code in PER_DATASET_CODES:
        log.error(f"Skipping {battery}/{dataset}: {e}")
        return False
    if code is not None and not isinstance(e, InfeasibleAllocationError):
        log.critical(f"Configuration error, aborting run: {e}")
        return True
    log.error(f"Skipping {battery}/{dataset}: unexpected {type(e).__name__}: {e}")
    return False


def _resolve_seed(block: Mapping[str, Any], name: str) -> tuple[int, str]:
    """The seed to use and how it was decided, for the summary txt. 42 when the
    key is absent. A valueless key asks for a random seed: one integer is drawn
    and logged, so the run can be reproduced. (Passing None on would also be
    random, but nothing could reproduce it.)"""
    if "seed" not in block:
        return 42, "default; the key is absent"
    seed = block["seed"]
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))
        log.warning(f"'{name}' has no value: using the randomly drawn seed {seed}. "
                    f"Set '{name}: {seed}' to reproduce this run.")
        return seed, "drawn at random; the key has no value"
    return seed, "set in the config"


def _unknown_suite_keys(source: str, suite: Mapping[str, Any]) -> str | None:
    """The refusal naming the `<source>_suite` keys the source does not take,
    with the closest key it does take as a suggestion; None when there are none
    or the source is unchecked."""
    if source not in FETCHER_SUITE_KEYS:
        return None
    takes = (*SUITE_KEYS, *FETCHER_SUITE_KEYS[source])
    unknown = []
    for key in suite:
        if key not in takes:
            close = difflib.get_close_matches(str(key), takes, n=1)
            unknown.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    if not unknown:
        return None
    return (f"Unknown key(s) in '{source}_suite': {', '.join(unknown)}. "
            f"'{source}_suite' takes: {', '.join(takes)}.")


def run_pipeline(source: str, config: Mapping[str, Mapping[str, Any]], csv_dir: Path,
                 png_dir: Path, txt_dir: Path, render_plots: bool = True,
                 config_out: Path | None = None,
                 report: dict[str, int] | None = None) -> int:
    """Runs every configured dataset through fetch, label generation, and
    output writing. Returns the number of datasets processed without error.
    With `config_out`, the config as the run read it is written there once the
    datasets and seeds are resolved, before the first dataset runs.
    With `report`, a run that reaches its datasets fills it with how many were
    `resolved`, written with all labels (`full`) or not all (`short`), and
    skipped by a per-dataset `[CLM-###]` code (`refused`); it stays empty when
    the config is refused first."""
    if source not in FETCHERS:
        log.error(f"Unknown data_source '{source}'. Valid options: {list(FETCHERS)}")
        return 0

    print_battery_info(source)

    # Read, never popped: the caller may reuse this config. Typed as a Mapping,
    # so a .pop() or assignment here is a mypy error.
    source_config = config.get(f"{source}_suite") or {}
    label_cfg = config.get("label_generation") or {}
    # Only the sections this run reads: an unused suite of any shape still runs.
    for key, section, example in ((f"{source}_suite", source_config, "batteries: all"),
                                  ("label_generation", label_cfg, "n_labels: 1")):
        problem = _not_a_mapping(key, section, example)
        if problem:
            log.error(problem)
            return 0
    problem = _unknown_suite_keys(source, source_config)
    if problem:
        log.error(problem)
        return 0

    batteries_cfg = source_config.get("batteries")
    if not batteries_cfg:
        log.error(f"'{source}_suite.batteries' not set in config. Specify 'all' or an explicit list.")
        return 0
    datasets_cfg = source_config.get("datasets") or "all"
    # A string other than "all" would be iterated letter by letter below:
    # datasets "mydata" became m, y, d, a, t, a, and batteries matched nothing.
    for key, value in (("batteries", batteries_cfg), ("datasets", datasets_cfg)):
        if isinstance(value, str) and value != "all":
            log.error(f"'{source}_suite.{key}' is the string {value!r}. Specify 'all' or a "
                      f"list, e.g. [{value}].")
            return 0
    batteries = resolve_selection(source, batteries_cfg)
    seed_lines = []   # recorded in every dataset's summary txt
    if source in SEEDED_SOURCES:
        fetch_seed, fetch_how = _resolve_seed(source_config, f"{source}_suite.seed")
        seed_lines.append(f"{source}_suite.seed = {fetch_seed} ({fetch_how})")
    else:
        fetch_seed = 42   # passed to the fetcher, never read: nothing to draw or report

    # Every other key is forwarded straight to the fetcher as extra kwargs
    # (e.g. mdcgen's `cp`/`out`, fabricated_data's `n_samples`, clustbench's
    # `base_url`).
    fetch_kwargs = {k: v for k, v in source_config.items() if k not in SUITE_KEYS}

    label_seed, label_how = _resolve_seed(label_cfg, "label_generation.seed")
    n_add_labels = label_cfg.get("n_labels", N_LABELS)
    source_labeling = label_cfg.get("source_labeling", SOURCE_LABELING)
    clm_config: dict | None = label_cfg.get("clm_label")

    seed_lines.append(
        f"label_generation.seed = {label_seed} ({label_how}); label i uses seed + i")

    jobs = _resolve_jobs(source, batteries, datasets_cfg, fetch_kwargs)
    if config_out is not None:
        _write_resolved_config(config_out, resolve_config(
            config, source, batteries, datasets_cfg, jobs,
            fetch_seed if source in SEEDED_SOURCES else None, fetch_kwargs, label_seed))
    # After the config is written, so a failed run kept with --keep-failed has it.
    # Other sources surface [CLM-105] per dataset.
    if not (_is_int(n_add_labels) and n_add_labels <= 0):
        _precheck_clm_config(clm_config)
    if source == "byoc":
        precheck_byoc_matching_ids(jobs, fetch_kwargs, clm_config)

    fetcher = FETCHERS[source]
    n_full = 0
    n_short = 0
    n_refused = 0

    for i, (battery, dataset) in enumerate(jobs, 1):
        # Every line until the next header is this dataset's: the engine's do not
        # name it.
        log.info(f"Dataset {i} of {len(jobs)}: {battery}/{dataset}")
        try:
            df = fetcher(battery, dataset, fetch_seed, **fetch_kwargs)
            if df is None:
                continue

            context = build_context(source, battery, dataset, df)

            # A dataset short of its labels is still written and processed.
            if n_add_labels > 0:
                _label_dataset(context, battery, dataset, label_cfg, n_add_labels,
                               source_labeling, clm_config, label_seed)

            _write_dataset(context, source, battery, dataset, source_labeling, clm_config,
                           csv_dir, png_dir, txt_dir, render_plots, seed_lines, label_seed)
            if len(context.generated_labels) < n_add_labels:
                n_short += 1
            else:
                n_full += 1
        except MissingConfigKey as e:
            # A required key with no default: no dataset can be labelled.
            log.critical(f"Configuration error, aborting run: {e}")
            raise
        except ValueError as e:
            if _aborts_run(e, battery, dataset):
                raise
            if getattr(e, "code", None) in PER_DATASET_CODES:
                n_refused += 1
        except Exception as e:
            log.error(f"Skipping {battery}/{dataset}: unexpected {type(e).__name__}: {e}")

    # Path-shaped names never became jobs; they were reported at resolution.
    n_ok = n_full + n_short
    n_skipped = len(jobs) - n_ok
    if n_short or n_skipped:
        log.warning(f"Of the {len(jobs)} resolved dataset(s): {n_full} written with all "
                    f"labels, {n_short} written with not all labels, {n_skipped} skipped "
                    "(see the errors above).")
    if report is not None:
        report.update(resolved=len(jobs), full=n_full, short=n_short, refused=n_refused)
    return n_ok


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


def _file_in_the_way(path: Path) -> Path | None:
    """The first existing part of `path` that is a file rather than a folder, or None."""
    for part in (*reversed(path.parents), path):
        if part.exists() and not part.is_dir():
            return part
    return None


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


class _Terminated(BaseException):
    """SIGTERM, which a scheduler sends to cancel a job or at its time limit. A
    BaseException, like KeyboardInterrupt, so no `except Exception` catches it."""


def _raise_terminated(signum: int, frame: object) -> None:
    raise _Terminated


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


def _run_batch(configs: list[str], jobs: int, flags: list[str]) -> int:
    """Each config in its own clmsynth process, `jobs` at a time, so each run keeps
    its own folder, log and exit status. A run's output is printed whole when it
    ends. Returns the batch's exit status (`_batch_status`)."""
    env = dict(os.environ)
    if jobs > 1:
        # Otherwise each worker may start a BLAS thread per core.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
    n = len(configs)
    log.info(f"Batch of {n} config(s), {min(jobs, n)} at a time.")

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
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        for future in as_completed(futures):
            report(futures[future], *future.result())
    except KeyboardInterrupt:
        stopped = 130
    except _Terminated:
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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Config paths, plus --jobs, --no-viz and --keep-failed."""
    parser = argparse.ArgumentParser(
        prog="clmsynth",
        description="Generate CLM-labeled datasets from configuration files.")
    parser.add_argument("configs", nargs="*", default=["test_data_config.yaml"],
                        metavar="config",
                        help="Configuration file: .json, or YAML with PyYAML installed "
                             "(default: test_data_config.yaml). Several run as a batch, "
                             "each in its own process.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="How many configs of a batch run at once (default: 1).")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip plot rendering; the plotting stack is never imported.")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep a failed run's folder, with its config and the log of "
                             "why it failed.")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be 1 or more")
    return args


def _exit_status(outcome: Mapping[str, int]) -> int:
    """0 when every dataset was written with all its labels, 3 when only some
    were; 2 when the config fits none (refused before its datasets, or each one
    refused by a code), 1 when nothing was written for another reason."""
    if not outcome.get("resolved"):
        return 2
    if outcome["full"] + outcome["short"] == 0:
        return 2 if outcome["refused"] == outcome["resolved"] else 1
    return 0 if outcome["full"] == outcome["resolved"] else 3


def main() -> None:
    """CLI entry point: ``python -m clmsynth.main [config ...] [--jobs N] [--no-viz]
    [--keep-failed]`` (or the ``clmsynth`` console script)."""
    # Parsed first: --help and a usage error print before any log line.
    args = _parse_args(sys.argv[1:])
    configure_cli_logging()
    if len(args.configs) > 1:
        flags = [flag for flag, on in (("--no-viz", args.no_viz),
                                       ("--keep-failed", args.keep_failed)) if on]
        sys.exit(_run_batch(args.configs, args.jobs, flags))
    log.info("Starting Test Data Orchestrator...")

    config_path = args.configs[0]
    config = load_config(config_path)
    global_settings = config.get("global_settings") or {}
    problem = _not_a_mapping("global_settings", global_settings, "data_source: byoc")
    if problem:
        # Exit 2 before anything is created, like a config that cannot be loaded.
        log.critical(f"Configuration file '{config_path}': {problem}")
        sys.exit(2)
    data_source = str(global_settings.get("data_source", DATA_SOURCE)).lower()
    render_plots = not args.no_viz

    base_dir = Path(global_settings.get("output_dir", OUTPUT_DIR))
    # Reporting where `output_dir` is (a configuration is a shareable artifact, see
    # SECURITY.md) rather than restricting.
    log.info(f"Output directory: '{resolved_for_report(base_dir)}'.")
    try:
        run_dir = build_run_dir(base_dir, SOURCE_DISPLAY.get(data_source, data_source))
    except OSError as e:
        # No `[CLM-###]` code: where a run writes is a property of the file
        # layout, not of the labeling. Exit 1, as for any folder that cannot be written.
        blocker = _file_in_the_way(base_dir)
        if blocker is None:
            log.critical(f"Cannot create the output folder '{resolved_for_report(base_dir)}': {e}")
        else:
            log.critical(
                f"Cannot create the output folder '{resolved_for_report(base_dir)}': "
                f"'{resolved_for_report(blocker)}' is a file, not a folder. Rename or move "
                "that file, or set global_settings.output_dir to another folder."
            )
        sys.exit(1)
    csv_dir, png_dir, info_dir = run_dir / "csv", run_dir / "png", run_dir / "info"
    for folder in (csv_dir, png_dir, info_dir):
        folder.mkdir(exist_ok=True)

    # The run's log and config sit in info/, named after the run folder.
    log_file = logging.FileHandler(info_dir / f"{run_dir.name}.log", encoding="utf-8")
    log_file.setFormatter(logging.Formatter(LOG_FORMAT))
    log_file.addFilter(SingleLineFilter())
    logging.getLogger().addHandler(log_file)
    # Ties the folder, named by time and source only, to the config that made it.
    log.info(f"Run {run_dir.name} from config '{resolved_for_report(config_path)}'.")
    if not render_plots:
        log.info("Plot rendering is off: png/ stays empty and matplotlib is never imported.")

    # Not a copy of the file: the config as the run read it, written by
    # run_pipeline in the format of the file it was given, which is the user's
    # to keep and share.
    config_file = info_dir / f"{run_dir.name}{'.json' if _is_json(config_path) else '.yaml'}"

    exit_code = 0
    outcome: dict[str, int] = {}
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        n_ok = run_pipeline(data_source, config, csv_dir, png_dir, info_dir, render_plots,
                            config_out=config_file, report=outcome)
        exit_code = _exit_status(outcome)
        if n_ok == 0:
            log.error("No datasets were successfully processed.")
        else:
            log.info(f"Pipeline ready. {n_ok} dataset(s) processed. Output folder: '{run_dir}'.")
    except (ValueError, TypeError, MissingConfigKey):
        # Coded [CLM-1xx] or [CLM-2xx] configuration error re-raised by
        # run_pipeline, or a malformed config value that reached an operator,
        # which has already logged it at CRITICAL.
        exit_code = 2
    except KeyboardInterrupt:
        log.warning("Interrupted; stopping.")
        exit_code = 130
    except _Terminated:
        log.warning("Terminated; stopping.")
        exit_code = 143
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        # Closed before the folder can be removed: Windows keeps an open file.
        logging.getLogger().removeHandler(log_file)
        log_file.close()

    if exit_code:
        if exit_code != 3:
            discard_failed_run(run_dir, args.keep_failed)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
