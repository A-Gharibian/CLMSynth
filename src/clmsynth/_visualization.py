# _visualization.py
"""Scatter-plot rendering for the pipeline outputs."""

import logging
import os
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

# Windows refuses paths at or beyond 260 characters unless long-path support is
# enabled. PNG names are the longest the pipeline writes, so with a deep
# output_dir they are what crosses the limit first -- while the CSV and TXT of
# the same dataset still write cleanly.
_MAX_PATH = 260

# matplotlib is not thread-safe and leaves locking to the caller: saved plots are
# built and written one at a time. Labels and CSVs do not wait on it.
_RENDER_LOCK = threading.Lock()


def _max_path_hint(output_path: str | None) -> str:
    """Name MAX_PATH when a plot failure is most likely really a path-length one.
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

    A saved plot does not go through pyplot: it is a `Figure` on its own Agg
    canvas, so it needs no GUI backend and is in no process-wide figure list, and
    can be rendered from a worker thread. Showing one goes through pyplot and the
    main thread.
    """
    if df is None or df.empty:
        log.warning("Empty DataFrame provided. Skipping plot generation.")
        return False
    if x_col not in df.columns or y_col not in df.columns:
        log.warning(f"Columns '{x_col}' or '{y_col}' not found in data. Skipping plot.")
        return False

    log.info(f"Generating scatter plot for {x_col} vs {y_col}...")

    fig = None
    shown = False
    if output_path:
        _RENDER_LOCK.acquire()  # released in finally
    try:
        if output_path:
            fig = Figure(figsize=(9, 6))
            FigureCanvasAgg(fig)  # attaches itself to fig
        else:
            fig = plt.figure(figsize=(9, 6))
        ax = fig.subplots()

        # Cluster/label ids are categorical, not a continuous scale: one viridis
        # color per id (colorblind-friendly), sampled clear of both ends of the map.
        handles: list[Line2D] = []
        if hue_col is not None:
            hue = df[hue_col]
            order = (sorted(hue.dropna().unique()) if pd.api.types.is_numeric_dtype(hue)
                     else list(hue.dropna().unique()))
            colors = np.asarray(colormaps["viridis"](np.linspace(0, 1, len(order) + 2)[1:-1]))
            codes = pd.Categorical(hue, categories=order).codes
            has_id = codes >= 0                    # a missing id is not drawn
            ax.scatter(df[x_col][has_id], df[y_col][has_id], c=colors[codes[has_id]],
                       s=36, alpha=0.8, linewidths=0.48, edgecolors="face")
            handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=c,
                              markeredgewidth=0.48, alpha=0.8, label=str(cat))
                       for cat, c in zip(order, colors, strict=True)]
        else:
            ax.scatter(df[x_col], df[y_col], s=36, alpha=0.8, linewidths=0.48,
                       edgecolors="face")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

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
            leg = ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc='upper left',
                            borderaxespad=0., title=hue_col, fontsize=8,
                            ncol=(2 if n_cats > 16 else 1))

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
            fig.savefig(path, dpi=300)  # fixed figsize (no tight bbox) -> constant panel
            log.info(f"Plot saved successfully to: '{path}'")
        else:
            plt.show()
            shown = True               # the viewer owns a shown figure
        return True

    except Exception as e:
        # Plotting is best-effort, not the pipeline's deliverable (the CSV is);
        # tagged so it reads as a plot-only failure in logs, not a dataset failure.
        log.error(f"[PLOT-FAIL] Failed to generate scatter plot: {e}{_max_path_hint(output_path)}")
        return False
    finally:
        # Only pyplot keeps a figure open, and a shown one belongs to the viewer.
        # A saved Figure goes with its last reference.
        if fig is not None and not output_path and not shown:
            plt.close(fig)
        if output_path:
            _RENDER_LOCK.release()
