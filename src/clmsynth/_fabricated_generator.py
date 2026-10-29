# SPDX-License-Identifier: MIT
# _fabricated_generator.py
"""Offline synthetic-feature generator.
Six numeric feature types, plus a categorical column generator.
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)


# ==========================================
# --- Main Generator -----------------------
# ==========================================

def generate_synthetic_data(
        n_samples: int = 2048,
        output_file: str | None = "synthetic_features_scaled.csv",
        seed: int = 42,
        config_dict: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """
    Generates a synthetic dataset driven by a configuration dictionary.
    """
    if config_dict is None:
        config_dict = {}

    log.info(f"Generating synthetic data (Seed: {seed})...")
    # Deterministic per seed.
    rng = np.random.default_rng(seed)
    index_start = 1000

    # 1. Base Continuous Features (Linear, Non-linear, etc.)
    f1 = rng.uniform(0, 100, n_samples)
    f2 = f1 + rng.normal(0, 5, n_samples)
    f3 = rng.uniform(0, 100, n_samples)
    f4_raw = np.sin(np.argsort(np.argsort(f3)) * (2 * np.pi / n_samples)) * 50 + 50
    f5 = rng.uniform(0, 100, n_samples)
    f6_raw = 50 + rng.choice([-1, 1], size=n_samples) * (f5 / 2) + rng.normal(0, 5, n_samples)

    raw_df = pd.DataFrame({
        'Feature_1': f1, 'Feature_2': f2, 'Feature_3': f3, 'Feature_4': f4_raw,
        'Feature_5': f5, 'Feature_6': f6_raw
    }, index=np.arange(index_start, index_start + n_samples))

    feature_cols = raw_df.columns
    raw_df[feature_cols] = raw_df[feature_cols].clip(0, 100)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(raw_df[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=raw_df.index)

    # 2. Process Configuration: Categorical Features & Ground Truths
    cat_config = config_dict.get("feature_generators", {}).get("categorical_features", {})
    ground_truths = config_dict.get("ground_truths", {})

    # labels_only: emit the reserved 'Cohort_Class' column and nothing else.
    labels_only = bool(cat_config.get("enable", False) and cat_config.get("labels_only", False))
    if labels_only:
        log.warning(
            "'labels_only': emitting ONLY the reserved 'Cohort_Class' column, drawn "
            "uniformly, and none of the six engineered features. The result carries "
            "labels and no geometry. Use it for label-only workflows; it is not a "
            "general-purpose fabrication mode."
        )

    final_dfs_to_concat: list = [] if labels_only else [scaled_df]

    if cat_config.get("enable", False):
        labels = cat_config.get("labels", ["Class_0", "Class_1"])

        clustering_config = ground_truths.get("class_clustering", {})
        if labels_only:
            cat_series = pd.Series(rng.choice(labels, size=n_samples), name="Cohort_Class")
        elif clustering_config.get("enforce_perfect_separation", False):
            log.info("Applying perfect separation ground truth to categorical labels...")
            # Perfect separation: class is the Feature_1 percentile bin, so a
            # clustering algorithm recovers a 1:1 label/cluster match.
            percentiles = pd.qcut(scaled_df['Feature_1'], q=len(labels), labels=labels)
            cat_series = pd.Series(percentiles, name=clustering_config.get("target_label", "Cohort_Class"))
        else:
            # A stream of its own from the seed, not the features' stream.
            cat_series = pd.Series(np.random.default_rng(seed).choice(labels, size=n_samples),
                                   name="Cohort_Class")

        cat_series.index = scaled_df.index
        final_dfs_to_concat.append(cat_series)

    # 3. Final Assembly
    final_df: pd.DataFrame = pd.concat(final_dfs_to_concat, axis=1)

    # 4. Save Output
    if output_file:
        try:
            final_df.to_csv(output_file, index=False)
            log.info(f"Saved synthetic data to: {output_file}")
        except Exception as e:
            log.error(f"Failed to save file: {e}")

    return final_df
