# fabricated_generator.py
"""Offline synthetic-feature generator backing the "fabricated_data" data source.

Produces six engineered numeric features plus an optional categorical
column.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from faker import Faker
except ImportError:
    Faker = None  # optional dependency; numpy fallback is used when absent

log = logging.getLogger(__name__)


# ==========================================
# --- Modular Injectors --------------------
# ==========================================

def generate_faker_categories(n_samples: int, labels: list, seed: int) -> pd.Series:
    """Generates categorical labels using Faker, falling back to numpy if unavailable."""
    # uses default_rng, and mutating global RNG state is not thread/process-safe.
    rng = np.random.default_rng(seed)
    if Faker is not None:
        Faker.seed(seed)
        fake = Faker()
        # Create a mapping of generic labels to realistic Fake categories (e.g., medical conditions or cohorts)
        group_mapping = {label: fake.company() for label in labels}
        # Randomly assign the base labels to maintain distribution, then map to the Faker strings
        base_assignments = rng.choice(labels, size=n_samples)
        faker_assignments = [group_mapping[val] for val in base_assignments]
        return pd.Series(faker_assignments, name="Cohort_Class")
    else:
        log.warning("Faker not installed. Falling back to basic numpy categories.")
        return pd.Series(rng.choice(labels, size=n_samples), name="Cohort_Class")


# ==========================================
# --- Main Generator -----------------------
# ==========================================

def generate_synthetic_data(
        n_samples: int = 2048,
        output_file: Optional[str] = "synthetic_features_scaled.csv",
        seed: int = 42,
        config_dict: Optional[Dict[str, Any]] = None
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

    # Assemble Numeric DataFrame
    raw_df = pd.DataFrame({
        'Feature_1': f1, 'Feature_2': f2, 'Feature_3': f3, 'Feature_4': f4_raw,
        'Feature_5': f5, 'Feature_6': f6_raw
    }, index=np.arange(index_start, index_start + n_samples))

    # Clean and Scale
    feature_cols = raw_df.columns
    raw_df[feature_cols] = raw_df[feature_cols].clip(0, 100)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(raw_df[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=raw_df.index)

    # 2. Process Configuration: Categorical Features & Ground Truths
    cat_config = config_dict.get("feature_generators", {}).get("categorical_features", {})
    ground_truths = config_dict.get("ground_truths", {})

    final_dfs_to_concat: list = [scaled_df]

    if cat_config.get("enable", False):
        labels = cat_config.get("labels", ["Class_0", "Class_1"])

        # Apply Perfect Separation Ground Truth if requested
        clustering_config = ground_truths.get("class_clustering", {})
        if clustering_config.get("enforce_perfect_separation", False):
            log.info("Applying perfect separation ground truth to categorical labels...")
            # Example logic: Assign class purely based on the percentile of Feature_1
            # This guarantees your pipeline's clustering algorithms will find a 1:1 match
            percentiles = pd.qcut(scaled_df['Feature_1'], q=len(labels), labels=labels)

            if cat_config.get("use_faker", False):
                # Map the perfect percentile splits to Faker names
                if Faker is not None:
                    fake = Faker()
                    Faker.seed(seed)
                    group_mapping = {label: fake.company() for label in labels}
                    percentiles = percentiles.map(group_mapping)
                else:
                    log.warning("Faker requested but not installed. Using base labels.")

            cat_series = pd.Series(percentiles, name=clustering_config.get("target_label", "Cohort_Class"))
        else:
            # Generate random categories if perfect separation is not enforced
            cat_series = generate_faker_categories(n_samples, labels, seed)

        cat_series.index = scaled_df.index
        final_dfs_to_concat.append(cat_series)

    # 3. Final Assembly
    final_df = pd.concat(final_dfs_to_concat, axis=1)

    # 4. Save Output
    if output_file:
        try:
            final_df.to_csv(output_file, index=False)
            log.info(f"Saved synthetic data to: {output_file}")
        except Exception as e:
            log.error(f"Failed to save file: {e}")

    return final_df