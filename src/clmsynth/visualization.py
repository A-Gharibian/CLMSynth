# visualization.py
"""Scatter-plot rendering for the pipeline outputs."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# Windows refuses paths at or beyond 260 characters unless long-path support is
# enabled. PNG names are the longest the pipeline writes, so with a deep
# output_dir they are what crosses the limit first -- while the CSV and TXT of
# the same dataset still write cleanly.
_MAX_PATH = 260


def _max_path_hint(output_path: str | None) -> str:
    """Name MAX_PATH when a plot failure is most likely really a path-length one.

    Windows reports it as `[Errno 2] No such file or directory` naming a path
    whose folder plainly exists, which points at entirely the wrong cause.
    """
    if not output_path or os.name != "nt":
        return ""
    try:
        length = len(str(Path(output_path).resolve()))
    except Exception:
        return ""
    if length < _MAX_PATH - 20:
        return ""
    return (f" -- NOTE: the output path is {length} characters. Windows rejects paths at "
            f"or beyond {_MAX_PATH} (MAX_PATH) and reports it as a missing file even "
            "though the folder exists. Use a shorter output_dir, or enable long-path "
            "support.")


def plot_feature_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: str | None = None,
        output_path: str | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        info_text: str | None = None
) -> bool:
    """Renders a scatter of two features, optionally colored by `hue_col`, to
    `output_path` (or shows it interactively when the path is omitted).

    `subtitle` is the smaller gray line under the title (the MCC/ARI scores);
    `info_text` is the monospace CLM-config box in the right margin. Returns True
    on success and False on any failure: plotting is best-effort, since the
    CSV/labels are the pipeline's real deliverable, so callers log a plot failure
    rather than treating it as a dataset failure.
    """
    if df is None or df.empty:
        log.warning("Empty DataFrame provided. Skipping plot generation.")
        return False
    if x_col not in df.columns or y_col not in df.columns:
        log.warning(f"Columns '{x_col}' or '{y_col}' not found in data. Skipping plot.")
        return False

    log.info(f"Generating scatter plot for {x_col} vs {y_col}...")

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(9, 6))

    try:
        # Cluster/label ids are categorical, not a continuous scale.
        plot_df = df
        if hue_col is not None and pd.api.types.is_numeric_dtype(df[hue_col]):
            plot_df = df.copy()
            order = sorted(df[hue_col].dropna().unique())
            plot_df[hue_col] = pd.Categorical(df[hue_col], categories=order, ordered=True)

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette="viridis" if hue_col else None,  # viridis is colorblind-friendly
            alpha=0.8,
            edgecolor=None,
            ax=ax,
        )

        # Fixed axes rectangle: the scatter panel occupies the same position and
        # width on every plot.
        fig.subplots_adjust(left=0.09, right=0.68, top=0.88, bottom=0.10)

        # Bold title, with an optional smaller gray subtitle stacked beneath it.
        # x centers them over the fixed axes rectangle, not the whole figure.
        display_title = title if title else f"Scatter Plot: {x_col} vs {y_col}"
        fig.suptitle(display_title, fontsize=14, fontweight='bold', x=0.385, y=0.965)
        if subtitle:
            ax.set_title(subtitle, fontsize=10, color='dimgray', pad=8)

        # Legend at the top of the right margin (2 columns once there are many
        # categories, so a bijection over K clusters keeps a bounded height).
        leg = None
        if hue_col:
            n_cats = df[hue_col].nunique()
            leg = ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0.,
                            title=hue_col, fontsize=8, ncol=(2 if n_cats > 16 else 1))

        # CLM-config box tucked just under the legend, high on the right margin.
        if info_text:
            y_top = 1.0
            if leg is not None:
                fig.canvas.draw()  # realize legend geometry so we can sit beneath it
                leg_box = leg.get_window_extent().transformed(ax.transAxes.inverted())
                y_top = leg_box.y0 - 0.04
            ax.text(1.02, y_top, info_text, transform=ax.transAxes, fontsize=7.5,
                    va='top', ha='left', family='monospace',
                    bbox={'boxstyle': 'round', 'facecolor': '#f7f7f7',
                          'edgecolor': '#cccccc'})

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300)  # fixed figsize (no tight bbox) -> constant panel
            log.info(f"Plot saved successfully to: '{path}'")
        else:
            plt.show()
        return True

    except Exception as e:
        # Plotting is best-effort, not the pipeline's deliverable (the CSV is);
        # tagged so it reads as a plot-only failure in logs, not a dataset failure.
        log.error(f"[PLOT-FAIL] Failed to generate scatter plot: {e}{_max_path_hint(output_path)}")
        return False
    finally:
        plt.close(fig)
