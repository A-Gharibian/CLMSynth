# visualization.py
"""Scatter-plot rendering for the pipeline outputs: any two feature columns,
colored by a chosen label column, with optional subtitle (metrics) and a
right-margin CLM-config annotation box."""

import logging
from typing import Optional
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)


def plot_feature_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        info_text: Optional[str] = None
) -> bool:
    """
    Creates a scatter plot to visualize feature relationships.

    Args:
        df: The pandas DataFrame containing the data.
        x_col: Name of the column for the X axis.
        y_col: Name of the column for the Y axis.
        hue_col: (Optional) Name of the column to color-code the points (e.g., class labels).
        output_path: (Optional) Filepath to save the plot (e.g., 'scatter.png').
        title: (Optional) Custom title for the plot.
        subtitle: (Optional) Smaller gray line under the title (e.g., MCC/ARI scores).
        info_text: (Optional) Monospace annotation box in the right margin
            (used for the CLM config summary).

    Returns:
        True if the plot was rendered and saved, False otherwise. The CSV/label
        output is the pipeline's actual deliverable; plotting is best-effort, so
        callers should log a plot failure, not treat it as a dataset failure.
    """
    # Guard clause
    if df is None or df.empty:
        log.warning("Empty DataFrame provided. Skipping plot generation.")
        return False
    if x_col not in df.columns or y_col not in df.columns:
        log.warning(f"Columns '{x_col}' or '{y_col}' not found in data. Skipping plot.")
        return False

    log.info(f"Generating scatter plot for {x_col} vs {y_col}...")

    # Set the plot theme
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(9, 6))

    try:
        # Cluster/label ids are categorical, not a continuous scale. seaborn
        # treats a numeric hue as continuous and samples the legend down to a
        # few representative ticks (e.g. 1,3,4,6,7,8 for 8 categories), dropping
        # the rest. Cast to an order-preserving categorical so every id gets its
        # own discrete color and a full legend entry. (String ids, which a byoc
        # CSV's cluster column may hold, are already categorical, so are untouched.)
        plot_df = df
        if hue_col is not None and pd.api.types.is_numeric_dtype(df[hue_col]):
            plot_df = df.copy()
            order = sorted(df[hue_col].dropna().unique())
            plot_df[hue_col] = pd.Categorical(df[hue_col], categories=order, ordered=True)

        # Generate the scatter plot
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
        # width on every plot, so the right margin (legend + CLM box) can grow or
        # shrink without ever moving or resizing the data panel.
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
                    bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='#cccccc'))

        # Save or Show
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
        log.error(f"[PLOT-FAIL] Failed to generate scatter plot: {e}")
        return False
    finally:
        # Close the figure to free memory and prevent leaks in loops
        plt.close(fig)