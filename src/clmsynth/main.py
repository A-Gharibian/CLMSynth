# main.py
"""Pipeline entry point.

Reads a config YAML (default test_data_config.yaml), fetches each requested
dataset from the configured source, generates the CLM labels, and writes one
timestamped run folder with the CSV, plots, and a metrics summary per dataset.
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .dataset_sources import (
    fetch_clustbench_data, fetch_mdcgen_data, fetch_fabricated_data,
    print_battery_info, resolve_selection, get_datasets_for_battery, is_heavy,
)

from .visualization import plot_feature_scatter
from .label_context import build_context
from .label_generator import generate_additional_labels
from .clm_label_engine import InfeasibleAllocationError
from .metrics import clustering_mcc, clustering_ari
from .byoc_source import fetch_byoc_data

log = logging.getLogger(__name__)

FETCHERS = {
    "clustbench": lambda group, name, seed, **kw: fetch_clustbench_data(dataset_group=group, dataset_name=name, **kw),
    "mdcgen": lambda group, name, seed, **kw: fetch_mdcgen_data(dataset_group=group, dataset_name=name, seed=seed, **kw),
    "fabricated_data": lambda group, name, seed, **kw: fetch_fabricated_data(dataset_group=group, dataset_name=name, seed=seed, **kw),
    "byoc": lambda group, name, seed, **kw: fetch_byoc_data(dataset_group=group, dataset_name=name, seed=seed, **kw),
}

# "clustbench"/"mdcgen" are fetcher keys, not the real dataset origin.
SOURCE_DISPLAY = {
    "clustbench": "Gagolewski",
    "mdcgen": "MDCGen",
    "fabricated_data": "Fabricated",
    "byoc": "BYOC",
}


def _clm_info_text(clm_config: dict) -> str:
    """Right-margin annotation: the CLM knobs that shaped this generated label.
    Deliberately kept narrow, the variable-length fields (proportions, and the
    centrality favors/profile) wrap onto extra indented lines so the box grows in
    HEIGHT, never in width, and can't bleed off the fixed-width figure."""
    props = clm_config.get("proportions") or []
    cd = clm_config.get("centroid_dependence", {}) or {}
    tm = clm_config.get("target_metric")

    lines = ["CLM config"]
    if props:
        lines.append("proportions:")
        vals = [f"{float(p):.3g}" for p in props]
        for i in range(0, len(vals), 2):          # two values per line
            lines.append("  " + ", ".join(vals[i:i + 2]))
    else:
        lines.append(f"balance: {clm_config.get('balance', '-')}")
    if cd.get("enabled"):
        lines.append("centrality:")
        lines.append(f"  {cd.get('favors', 'core')}/{cd.get('profile', 'linear')}")
    else:
        lines.append("centrality: off")
    if tm:
        lines.append(f"target: {tm.get('type')}={tm.get('value')}")
    else:
        lines.append(f"matching: {clm_config.get('matching_mode', '-')}")
    return "\n".join(lines)


def _write_summary_txt(path: Path, friendly: str, source: str, battery: str, dataset: str,
                       source_labeling: str, gt_col, n_rows: int, clm_config, label_results) -> None:
    """One human-readable txt per dataset: the configuration used plus the exact
    MCC/ARI shown on the plots (both come from `label_results`, so the txt and the
    plot subtitles can never disagree)."""
    gt_disp = gt_col if gt_col else "(none)"
    out = [
        f"Source : {friendly} ({source})",
        f"Dataset: {battery}/{dataset}",
        f"Rows   : {n_rows}",
        f"Ground-truth labeling: {source_labeling}  ->  {gt_disp}",
        "",
        "===== CLM label configuration =====",
        yaml.dump(clm_config, sort_keys=False, default_flow_style=False).rstrip()
        if clm_config else "(legacy noise mode, no clm_label config)",
        "",
        "===== Results (identical to each plot's subtitle) =====",
    ]
    if not label_results:
        out.append("(no generated labels)")
    for r in label_results:
        if r["mcc"] is not None:
            out.append(f"{r['name']}:  MCC = {r['mcc']:.3f}   ARI = {r['ari']:.3f}   (vs {gt_disp})")
        else:
            out.append(f"{r['name']}:  (no ground-truth labeling to compare against)")
        out.append(f"    label counts = {r['counts']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    log.info(f"Saved summary: {path}")

def load_config(config_path: str) -> dict:
    """Loads the pipeline config YAML; exits with a coded message if absent or empty."""
    path = Path(config_path)
    if not path.is_file():
        log.critical(f"Configuration file not found at '{config_path}'. "
                     f"Run generate_config.py or config_wizard.py first.")
        sys.exit(1)
    with open(path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # An empty file parses to None; anything non-mapping means main()'s config.get(...)
    if not isinstance(config, dict):
        log.critical(f"Configuration file '{config_path}' is empty or not a YAML mapping.")
        sys.exit(1)
    return config

def run_pipeline(source: str, config: dict, csv_dir: Path, png_dir: Path, txt_dir: Path) -> int:
    """Runs every configured dataset through fetch, label generation, and
    output writing. Returns the number of datasets processed without error."""
    if source not in FETCHERS:
        log.error(f"Unknown data_source '{source}'. Valid options: {list(FETCHERS)}")
        return 0

    print_battery_info(source)

    # Copy so popping keys below can't mutate the caller's master config dict.
    source_config = config.get(f"{source}_suite", {}).copy()

    batteries_cfg = source_config.pop("batteries", None)
    if not batteries_cfg:
        log.error(f"'{source}_suite.batteries' not set in config. Specify 'all' or an explicit list.")
        return 0
    batteries = resolve_selection(source, batteries_cfg)
    datasets_cfg = source_config.pop("datasets", "all")
    fetch_seed = source_config.pop("seed", 42)

    # Whatever's left after popping the three known keys is forwarded
    # straight to the fetcher as extra kwargs (e.g. mdcgen's `cp`/`out`,
    # fabricated_data's `n_samples`, clustbench's `base_url`).
    fetch_kwargs = source_config

    label_cfg = config.get("label_generation", {})
    n_add_labels = label_cfg.get("n_labels", 0)
    source_labeling = label_cfg.get("source_labeling", "labels0")
    clm_config: Optional[dict] = label_cfg.get("clm_label")

    if any(is_heavy(source, b) for b in batteries):
        log.warning(f"Selection includes a heavy '{source}' battery: expect long fetch/generation times.")

    jobs = [(b, d) for b in batteries for d in get_datasets_for_battery(source, b, datasets_cfg)]
    log.info(f"Resolved {len(jobs)} dataset(s) across {len(batteries)} batter(y/ies) from source '{source}'.")
    if fetch_kwargs:
        log.info(f"Extra fetch kwargs forwarded to '{source}': {fetch_kwargs}")

    fetcher = FETCHERS[source]
    n_ok = 0

    for battery, dataset in jobs:
        try:
            df = fetcher(battery, dataset, fetch_seed, **fetch_kwargs)
            if df is None:
                continue

            context = build_context(source, battery, dataset, df)

            if n_add_labels > 0:
                try:
                    generate_additional_labels(
                        context, n_labels=n_add_labels, source_labeling=source_labeling,
                        clm_config=clm_config,
                        noise=label_cfg.get("noise", 0.1), seed=label_cfg.get("seed", 42),
                    )
                except (KeyError, InfeasibleAllocationError) as e:
                    log.error(f"Skipping label generation for {battery}/{dataset}: {e}")

            final_df = context.to_dataframe()
            stem = f"{source}__{battery}__{dataset}"
            final_df.to_csv(csv_dir / f"{stem}.csv", index=False)
            log.info(f"Saved: {csv_dir / f'{stem}.csv'}")

            friendly = SOURCE_DISPLAY.get(source, source)
            plot_title = f"{friendly}: {battery}/{dataset}"
            gt_col = (context.gt_column_name(source_labeling)
                      if source_labeling in context.ground_truths else None)

            label_results = []
            for label_name in context.generated_labels:
                mcc = ari = None
                if gt_col is not None:
                    mcc = float(clustering_mcc(final_df[gt_col], final_df[label_name]))
                    ari = float(clustering_ari(final_df[gt_col], final_df[label_name]))
                counts = final_df[label_name].value_counts().sort_index().to_dict()
                label_results.append({"name": label_name, "mcc": mcc, "ari": ari, "counts": counts})

            _write_summary_txt(txt_dir / f"{stem}.txt", friendly, source, battery, dataset,
                               source_labeling, gt_col, len(final_df), clm_config, label_results)

            feature_cols = list(context.features.columns)
            if len(feature_cols) < 2:
                log.warning(f"'{stem}' has fewer than 2 features; skipping plots.")
                n_ok += 1
                continue
            x_col, y_col = feature_cols[0], feature_cols[1]

            if gt_col is not None and gt_col in final_df.columns:
                if not plot_feature_scatter(
                    final_df, x_col=x_col, y_col=y_col,
                    hue_col=gt_col,
                    output_path=str(png_dir / f"{stem}__{gt_col}.png"),
                    title=plot_title,
                    subtitle=f"ground-truth clusters ({gt_col})",
                ):
                    log.warning(f"'{stem}': ground-truth plot failed; CSV/labels are unaffected.")
            else:
                log.warning(f"ground-truth for '{source_labeling}' not found for '{stem}'; skipping plot.")

            # Subtitle metrics compare each generated label against the SAME ground-truth
            # labeling the CLM engine keyed off (source_labeling), reusing label_results.
            clm_info = _clm_info_text(clm_config) if clm_config else None
            for r in label_results:
                subtitle = None
                if r["mcc"] is not None:
                    subtitle = f"MCC {r['mcc']:.3f}  ·  ARI {r['ari']:.3f}   (vs {gt_col})"
                if not plot_feature_scatter(
                    final_df, x_col=x_col, y_col=y_col,
                    hue_col=r["name"],
                    output_path=str(png_dir / f"{stem}__{r['name']}.png"),
                    title=plot_title,
                    subtitle=subtitle,
                    info_text=clm_info,
                ):
                    log.warning(f"'{stem}': plot for '{r['name']}' failed; CSV/labels are unaffected.")

            # Plotting is best-effort (see above); the dataset's real deliverable
            # is the CSV/labels written earlier, so a plot failure alone does not
            # make this dataset a failed run.
            n_ok += 1
        except ValueError as e:
            # A coded [CLM-1xx] error means the *configuration* is wrong, which is
            # equally wrong for every remaining dataset.
            if getattr(e, "code", None) is not None and not isinstance(e, InfeasibleAllocationError):
                log.critical(f"Configuration error, aborting run: {e}")
                raise
            log.error(f"Skipping {battery}/{dataset}: unexpected error: {e}")
            continue
        except Exception as e:
            log.error(f"Skipping {battery}/{dataset}: unexpected error: {e}")
            continue

    return n_ok


def build_run_dir(base_dir: Path, friendly_source: str) -> Path:
    """One self-packaging folder per run: DDMMYY_Source_HHMMSS/"""
    now = datetime.now()
    run_dir = base_dir / f"{now:%d%m%y}_{friendly_source}_{now:%H%M%S}"
    unique, n = run_dir, 1
    while unique.exists():
        n += 1
        unique = base_dir / f"{run_dir.name}_{n}"
    return unique


def main() -> None:
    """CLI entry point: ``python -m clmsynth.main [config.yaml]`` (or the
    ``clmsynth`` console script)."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.info("Starting Test Data Orchestrator...")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "test_data_config.yaml"
    config = load_config(config_path)
    global_settings = config.get("global_settings", {})
    data_source = str(global_settings.get("data_source", "clustbench")).lower()

    base_dir = Path(global_settings.get("output_dir", "OUTPUT"))
    run_dir = build_run_dir(base_dir, SOURCE_DISPLAY.get(data_source, data_source))
    csv_dir = run_dir / "csv"
    png_dir = run_dir / "png"
    txt_dir = run_dir / "txt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Copy the exact config used into the run folder for provenance.
    shutil.copy(config_path, run_dir / Path(config_path).name)

    try:
        n_ok = run_pipeline(data_source, config, csv_dir, png_dir, txt_dir)
    except ValueError:
        # Coded [CLM-1xx] configuration error re-raised by run_pipeline, which has
        # already logged it at CRITICAL.
        sys.exit(2)

    if n_ok == 0:
        log.error("No datasets were successfully processed.")
        sys.exit(1)
    log.info(f"Pipeline ready. {n_ok} dataset(s) processed. Output folder: '{run_dir}'.")


if __name__ == "__main__":
    main()