# config_template.py
"""The commented YAML template that generate_config.py renders into a
runnable pipeline config. Placeholders are filled via str.format, so the
braces here are template slots, not f-string expressions."""

YAML_TEMPLATE = """\
global_settings:
  data_source: "{data_source}"  # clustbench | mdcgen | fabricated_data | byoc
  output_dir: "{output_dir}"    # base folder; created on the run, one timestamped subfolder per run

# --- {data_source_suite_key} CONFIGURATION ---
{data_source_suite_key}:
  batteries: {batteries}   # "all" or a list, e.g. ["wut"]
  datasets: {datasets}     # "all" or a list, e.g. ["smile"]
  seed: {source_seed}{byoc_extra}

# --- LABEL GENERATION ---
label_generation:
  n_labels: {n_labels}          # produces Label_0, Label_1, ...
  source_labeling: "{source_labeling}"
  noise: {noise}                # fallback only, used if clm_label is absent
  seed: {label_seed}

  # --- CLUSTER-LABEL MATCHING (CLM) CONFIGURATION ---
  clm_label:
    # 1. CARDINALITY & PROPORTIONS
    num_classes: {num_classes}  # M: Number of labels to generate
    proportions: {proportions}  # Must sum to 1.0.

    # 2. BALANCE
    # 'balanced': uniform 1/M split, proportions ignored.
    # 'unbalanced': 'proportions' above used directly if set;
    #   otherwise falls through to skew_rule below.
    balance: "{balance}"
    # Only consulted if balance != 'balanced' AND proportions is empty/unset.
    # Options: 'geometric', 'dominant_minority', 'dirichlet'
    skew_rule: "{skew_rule}"

    # 3. MATCHING MODE
    # Options: 'perfect', 'single', 'random', 'custom'
    matching_mode: "{matching_mode}"

    # Used ONLY if matching_mode is 'single'
    single_match:
{single_match}

    # Used ONLY if matching_mode is 'custom'
    assignment_matrix:
{assignment_matrix}

    # 4. DISTRIBUTION OF NON-TARGET (NOISE) LABELS
    # Options: 'proportional_to_size' or 'equal'
    split_rule: "{split_rule}"
    # Options: 'proportional_to_marginal', 'uniform', 'concentrated'
    spillover_rule: "{spillover_rule}"

    # 5. SPATIAL ASSIGNMENT (CENTROID PROXIMITY)
    centroid_dependence:
      enabled: {centroid_enabled}  # If False, labels are randomly distributed
      profile: "{centroid_profile}"  # Options: 'linear', 'exponential', 'step'
      favors: "{centroid_favors}"    # Options: 'core', 'boundary'
"""