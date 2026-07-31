# generate_config.py
"""Renders config_template.YAML_TEMPLATE into a runnable pipeline config.

The input is a flat "upstream payload" mapping (see upstream_payload.yaml);
the output is the full, commented YAML that main.py reads. Import
generate_base_config to drive it from another system, or run this module
as a script to render a payload file.
"""

import logging
import sys

import yaml

from .config_template import YAML_TEMPLATE

log = logging.getLogger(__name__)

VALID_SKEW_RULES = {"geometric", "dominant_minority", "dirichlet"}  # only what clm_label_engine.py actually implements
VALID_SOURCES = {"clustbench", "mdcgen", "fabricated_data", "byoc"}


def format_yaml_snippet(data, indent_level=6):
    """Converts a dict/list into a formatted YAML string for injection.
    indent_level=6 matches single_match/assignment_matrix sitting two
    levels deep (label_generation -> clm_label -> field)."""
    if not data:
        return " " * indent_level + ("[]" if isinstance(data, list) else "{}")
    snippet = yaml.dump(data, default_flow_style=False, sort_keys=False).strip()
    return "\n".join(" " * indent_level + line for line in snippet.split("\n"))


def format_list_or_all(value) -> str:
    """Renders a batteries/datasets value: the literal "all" or a YAML list."""
    if value == "all":
        return '"all"'
    return f"[{', '.join(repr(v) for v in value)}]"


def generate_base_config(upstream_data: dict, output_path: str = "test_data_config.yaml"):
    """Generates the YAML configuration using upstream dynamic data."""

    data_source = upstream_data.get("data_source", "clustbench")
    if data_source not in VALID_SOURCES:
        log.warning(f"data_source '{data_source}' is not one of {VALID_SOURCES}; "
                    "main.py's FETCHERS dict will reject this at runtime.")

    balance = upstream_data.get("balance", "unbalanced")
    skew_rule = upstream_data.get("skew_rule", "geometric")
    has_proportions = bool(upstream_data.get("proportions"))

    if balance == "balanced" and has_proportions:
        log.warning(
            "balance='balanced' with 'proportions' also set: the engine enforces a "
            "uniform 1/M split and ignores explicit proportions. Set balance to "
            "'unbalanced' to have your proportions used directly."
        )
    if balance == "unbalanced" and not has_proportions and skew_rule not in VALID_SKEW_RULES:
        log.warning(f"skew_rule '{skew_rule}' is not implemented by clm_label_engine.py "
                    f"(valid: {VALID_SKEW_RULES}). This config will fail at runtime unless fixed.")

    proportions_str = f"[{', '.join(map(str, upstream_data.get('proportions', [])))}]"
    single_match_yaml = format_yaml_snippet(upstream_data.get("single_match", {"cluster": None, "label": None}))
    assignment_matrix_yaml = format_yaml_snippet(upstream_data.get("assignment_matrix", []))
    centroid_enabled = "true" if upstream_data.get("centroid_enabled", True) else "false"

    # byoc carries three extra suite keys the other sources don't have; they are
    # appended under the suite block only when the source is byoc (empty otherwise
    # so the template stays valid for clustbench/mdcgen/fabricated_data).
    if data_source == "byoc":
        byoc_extra = (
            f"\n  input_dir: {upstream_data.get('input_dir', 'INPUT')!r}"
            f"\n  cluster_column: {upstream_data.get('cluster_column', 'cluster')!r}"
            f"\n  standardize: {'true' if upstream_data.get('standardize', False) else 'false'}"
        )
    else:
        byoc_extra = ""

    rendered_yaml = YAML_TEMPLATE.format(
        data_source=data_source,
        output_dir=upstream_data.get("output_dir", "OUTPUT"),
        byoc_extra=byoc_extra,
        data_source_suite_key=f"{data_source}_suite",
        batteries=format_list_or_all(upstream_data.get("batteries", "all")),
        datasets=format_list_or_all(upstream_data.get("datasets", "all")),
        source_seed=upstream_data.get("source_seed", 42),

        n_labels=upstream_data.get("n_labels", 1),
        source_labeling=upstream_data.get("source_labeling", "labels0"),
        noise=upstream_data.get("noise", 0.1),
        label_seed=upstream_data.get("label_seed", 42),

        num_classes=upstream_data.get("num_classes", 4),
        proportions=proportions_str,
        balance=balance,
        skew_rule=skew_rule,
        matching_mode=upstream_data.get("matching_mode", "custom"),
        single_match=single_match_yaml,
        assignment_matrix=assignment_matrix_yaml,
        split_rule=upstream_data.get("split_rule", "proportional_to_size"),
        spillover_rule=upstream_data.get("spillover_rule", "proportional_to_marginal"),
        centroid_enabled=centroid_enabled,
        centroid_profile=upstream_data.get("centroid_profile", "exponential"),
        centroid_favors=upstream_data.get("centroid_favors", "core"),
    )

    try:
        with open(output_path, 'w') as file:
            file.write(rendered_yaml)
        log.info(f"Test data configuration successfully written to '{output_path}'")
    except Exception as e:
        log.error(f"Failed to write configuration file: {e}")


def main() -> None:
    """CLI entry point: ``python generate_config.py [payload.yaml] [output.yaml]``.

    Loads the upstream payload from the given YAML file (default
    ``upstream_payload.yaml``) and renders the pipeline config (default
    ``test_data_config.yaml``).
    """
    logging.basicConfig(level=logging.INFO)
    payload_path = sys.argv[1] if len(sys.argv) > 1 else "upstream_payload.yaml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_data_config.yaml"

    try:
        with open(payload_path, encoding="utf-8") as file:
            payload = yaml.safe_load(file)
    except FileNotFoundError:
        log.critical(f"Payload file '{payload_path}' not found. "
                     f"Copy or edit upstream_payload.yaml, or pass a payload path.")
        sys.exit(1)

    if not isinstance(payload, dict):
        log.critical(f"Payload file '{payload_path}' must contain a YAML mapping.")
        sys.exit(1)

    generate_base_config(upstream_data=payload, output_path=output_path)


if __name__ == "__main__":
    main()