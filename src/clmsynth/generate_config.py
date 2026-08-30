# generate_config.py
"""Renders config_template.YAML_TEMPLATE into a runnable pipeline config.
The input is a flat "upstream payload" mapping (see upstream_payload.yaml).
"""

import json
import logging
import sys

import yaml

from .cli_logging import configure_cli_logging
from .config_template import YAML_TEMPLATE

log = logging.getLogger(__name__)

VALID_SKEW_RULES = {"geometric", "dominant_minority", "dirichlet"}
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
    if isinstance(value, str):
        value = [value]
    return f"[{', '.join(_yaml_scalar(v) for v in value)}]"


def _yaml_scalar(value) -> str:
    """Renders one payload value as a YAML scalar: bools lowercased, strings
    quoted, numbers as-is. Booleans are checked first, since bool is a subclass
    of int and would otherwise render as True/False rather than true/false."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # JSON strings are valid YAML double-quoted scalars.
        return json.dumps(value)
    return str(value)


def _optional_block(lines: list[str]) -> str:
    """Joins one optional clm_label block, or returns '' when there is nothing
    to emit. Each block supplies its own leading newline so an omitted one
    leaves no blank line in the rendered YAML, the same trick `byoc_extra` uses
    for the byoc-only suite keys."""
    return ("\n" + "\n".join(lines)) if lines else ""


def generate_base_config(upstream_data: dict, output_path: str = "test_data_config.yaml"):
    """Generates the YAML configuration using upstream dynamic data."""

    def pick(key, default):
        """Payload value, or the default when absent or null."""
        value = upstream_data.get(key)
        return default if value is None else value

    data_source = pick("data_source", "clustbench")
    if data_source not in VALID_SOURCES:
        log.warning(f"data_source '{data_source}' is not one of {VALID_SOURCES}; "
                    "main.py's FETCHERS dict will reject this at runtime.")

    balance = pick("balance", "unbalanced")
    skew_rule = pick("skew_rule", "geometric")
    has_proportions = bool(pick("proportions", []))

    if balance == "balanced" and has_proportions:
        log.warning(
            "balance='balanced' with 'proportions' also set: the engine enforces a "
            "uniform 1/M split and ignores explicit proportions. Set balance to "
            "'unbalanced' to have your proportions used directly."
        )
    if balance == "unbalanced" and not has_proportions and skew_rule not in VALID_SKEW_RULES:
        log.warning(f"skew_rule '{skew_rule}' is not implemented by clm_label_engine.py "
                    f"(valid: {VALID_SKEW_RULES}). This config will fail at runtime unless fixed.")

    # --- optional clm_label blocks ------------------------------------------

    mode = pick("matching_mode", "custom")
    target_metric = pick("target_metric", {})
    competing_noise = pick("competing_noise", [])

    if target_metric and mode not in ("single", "custom"):
        log.warning(f"target_metric is set but matching_mode is '{mode}': the engine rejects "
                    "this ([CLM-111]/[CLM-114]). Use 'single'/'custom', or drop target_metric.")
    if target_metric.get("scope") == "pair" and (
            target_metric.get("type") != "mcc" or mode != "single"):
        log.warning("target_metric.scope 'pair' requires type 'mcc' and matching_mode 'single' "
                    "([CLM-123]/[CLM-124]); this config will fail at runtime unless fixed.")
    if competing_noise and mode == "random":
        log.warning("competing_noise is set with matching_mode 'random': the engine rejects this "
                    "([CLM-115]). Use 'single' or 'custom'.")

    skew_params = pick("skew_params", {})
    skew_params_block = _optional_block(
        ["    skew_params:"] + [f"      {k}: {_yaml_scalar(v)}" for k, v in skew_params.items()]
        if skew_params else []
    )

    concentrated = pick("concentrated_labels", [])
    concentrated_block = _optional_block(
        [(f"    concentrated_labels: [{', '.join(str(x) for x in concentrated)}]"
          "  # spillover_rule 'concentrated' only")] if concentrated else []
    )

    competing_lines: list[str] = []
    if competing_noise:
        competing_lines = [
            "",
            "    # STRUCTURED COMPETING NOISE (single/custom only). Converts a share of ONE",
            "    # cluster's UNCLAIMED points to one competing label; bypasses 'proportions'.",
            "    competing_noise:",
        ]
        for entry in competing_noise:
            inner = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in entry.items())
            competing_lines.append(f"      - {{{inner}}}")
    competing_block = _optional_block(competing_lines)

    target_lines: list[str] = []
    if target_metric:
        target_lines = [
            "",
            "    # TARGET METRIC (single/custom only): solves the recall level.",
            "    # scope 'global' (default) searches numerically; 'pair' is exact but needs",
            "    # type 'mcc' AND matching_mode 'single'. tolerance applies to both scopes;",
            "    # max_iter is global-only (the pair scope never iterates).",
            "    target_metric:",
        ]
        # Rendered generically: an unrecognised key stays visible.
        for key, value in target_metric.items():
            if value is not None:
                target_lines.append(f"      {key}: {_yaml_scalar(value)}")
    target_block = _optional_block(target_lines)

    steepness = pick("centroid_steepness", None)
    steepness_block = _optional_block(
        [f"      steepness: {steepness}  # exponential profile only"]
        if steepness is not None and pick("centroid_profile", "exponential") == "exponential" else []
    )

    proportions_str = f"[{', '.join(map(str, pick('proportions', [])))}]"
    single_match_yaml = format_yaml_snippet(pick("single_match", {"cluster": None, "label": None}))
    assignment_matrix_yaml = format_yaml_snippet(pick("assignment_matrix", []))
    centroid_enabled = "true" if pick("centroid_enabled", True) else "false"

    if data_source == "byoc":
        tags = pick("tag_columns", [])
        if isinstance(tags, str):
            tags = [tags]
        tag_line = (f"\n  tag_columns: "
                    f"[{', '.join(_yaml_scalar(t) for t in tags)}]") if tags else ""
        byoc_extra = (
            f"\n  input_dir: {_yaml_scalar(pick('input_dir', 'INPUT'))}"
            f"\n  cluster_column: {_yaml_scalar(pick('cluster_column', 'cluster'))}"
            + tag_line +
            f"\n  standardize: {'true' if pick('standardize', False) else 'false'}"
        )
    else:
        byoc_extra = ""

    rendered_yaml = YAML_TEMPLATE.format(
        data_source=data_source,
        output_dir=_yaml_scalar(str(pick("output_dir", "OUTPUT"))),
        byoc_extra=byoc_extra,
        data_source_suite_key=f"{data_source}_suite",
        batteries=format_list_or_all(pick("batteries", "all")),
        datasets=format_list_or_all(pick("datasets", "all")),
        source_seed=pick("source_seed", 42),

        n_labels=pick("n_labels", 1),
        source_labeling=pick("source_labeling", "labels0"),
        noise=pick("noise", 0.1),
        label_seed=pick("label_seed", 42),

        num_classes=pick("num_classes", 4),
        proportions=proportions_str,
        balance=balance,
        skew_rule=skew_rule,
        matching_mode=mode,
        single_match=single_match_yaml,
        assignment_matrix=assignment_matrix_yaml,
        split_rule=pick("split_rule", "proportional_to_size"),
        spillover_rule=pick("spillover_rule", "proportional_to_marginal"),
        centroid_enabled=centroid_enabled,
        centroid_profile=pick("centroid_profile", "exponential"),
        centroid_favors=pick("centroid_favors", "core"),

        # Optional blocks: '' unless the payload asked for them.
        skew_params=skew_params_block,
        concentrated_labels=concentrated_block,
        competing_noise=competing_block,
        target_metric=target_block,
        steepness=steepness_block,
    )

    try:
        yaml.safe_load(rendered_yaml)
    except yaml.YAMLError as e:
        log.critical(f"Rendered configuration is not valid YAML: {e}")
        sys.exit(1)

    try:
        with open(output_path, 'w', encoding="utf-8") as file:
            file.write(rendered_yaml)
        log.info(f"Test data configuration successfully written to '{output_path}'")
    except OSError as e:
        log.critical(f"Failed to write configuration file: {e}")
        sys.exit(1)


def main() -> None:
    """CLI entry point: ``python -m clmsynth.generate_config [payload.yaml] [output.yaml]``.

    Loads the upstream payload from the given YAML file (default
    ``upstream_payload.yaml``) and renders the pipeline config (default
    ``test_data_config.yaml``).
    """
    configure_cli_logging()
    payload_path = sys.argv[1] if len(sys.argv) > 1 else "upstream_payload.yaml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_data_config.yaml"

    try:
        with open(payload_path, encoding="utf-8") as file:
            payload = yaml.safe_load(file)
    except FileNotFoundError:
        log.critical(f"Payload file '{payload_path}' not found. It ships with the "
                     f"repository and the sdist, not the wheel. Pass a payload path, "
                     f"or run clmsynth-wizard instead.")
        sys.exit(1)

    if not isinstance(payload, dict):
        log.critical(f"Payload file '{payload_path}' must contain a YAML mapping.")
        sys.exit(1)

    try:
        generate_base_config(upstream_data=payload, output_path=output_path)
    except KeyboardInterrupt:
        log.warning("Interrupted; nothing written.")
        sys.exit(130)


if __name__ == "__main__":
    main()
