# SPDX-License-Identifier: MIT
# _config_gen/generate_config.py
"""Renders config_template.YAML_TEMPLATE into a runnable pipeline config.
The input is a flat "upstream payload" mapping (see docs/examples/hepta_payload.yaml).
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .._utils.cli_logging import configure_cli_logging
from .._utils.compat import import_optional_dependency
from .._utils.write_files import write_text_atomic
from ..clm_errors import MissingConfigKey
from ..config_tree import default as setting_default
from ..dataset_sources import SOURCES
from .config_template import PERFECT_SIZING, SIZING_TEMPLATE, YAML_TEMPLATE

__all__ = [
    "VALID_SOURCES",
    "format_list_or_all",
    "format_yaml_snippet",
    "generate_base_config",
    "main",
]

log: logging.Logger = logging.getLogger(__name__)

# A tuple, not a set: `in` must not hash a payload value, which may be a list.
VALID_SOURCES = tuple(SOURCES)


def format_yaml_snippet(data: dict[str, Any] | list[Any], indent_level: int = 6) -> str:
    """Converts a dict/list into a formatted YAML string for injection.
    indent_level=6 matches single_match/assignment_matrix sitting two
    levels deep (label_generation -> clm_label -> field)."""
    import yaml

    if not data:
        return " " * indent_level + ("[]" if isinstance(data, list) else "{}")
    snippet = yaml.dump(data, default_flow_style=False, sort_keys=False).strip()
    return "\n".join(" " * indent_level + line for line in snippet.split("\n"))


def format_list_or_all(value: str | Iterable[object]) -> str:
    """Renders a batteries/datasets value: the literal "all" or a YAML list."""
    if value == "all":
        return '"all"'
    if isinstance(value, str):
        value = [value]
    return f"[{', '.join(_yaml_scalar(v) for v in value)}]"


def _yaml_scalar(value) -> str:
    """Renders one payload value as a YAML scalar: bools lowercased, strings
    quoted, None as null, numbers as-is. Booleans are checked first, since bool
    is a subclass of int and would otherwise render as True/False rather than
    true/false."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        # JSON strings are valid YAML double-quoted scalars. Not as ASCII: JSON writes
        # a character past U+FFFF as two escapes, which YAML reads as two characters.
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    if isinstance(value, float) and "e" in text and "." not in text:
        return _dotted_exponent(text)
    return text


def _dotted_exponent(text: str) -> str:
    """`1e-05` as `1.0e-05`: PyYAML reads a float without a dot as text."""
    mantissa, exponent = text.split("e")
    return f"{mantissa}.0e{exponent}"


def _yaml_list(value) -> str:
    """A list as a YAML flow sequence, and anything else as given."""
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(_yaml_scalar(v) for v in value)}]"
    return _yaml_scalar(value)


def _optional_block(lines: list[str]) -> str:
    """Joins one optional clm_label block, or returns '' when there is nothing
    to emit. Each block supplies its own leading newline so an omitted one
    leaves no blank line in the rendered YAML, the same trick `byoc_extra` uses
    for the byoc-only suite keys."""
    return ("\n" + "\n".join(lines)) if lines else ""


def generate_base_config(upstream_data: dict[str, Any],
                         output_path: str = "test_data_config.yaml") -> None:
    """Generates the configuration using upstream dynamic data: JSON for a .json
    output path, YAML otherwise. The clm_label block is written as given and
    checked as generate_clm_labels checks it, without a dataset; a refusal is
    raised with its [CLM-###] code (ValueError or MissingConfigKey) and nothing
    is written."""
    import yaml

    from ..clm_label_engine import _check_config

    def pick(key, default):
        """Payload value, or the default when absent or null."""
        value = upstream_data.get(key)
        return default if value is None else value

    data_source = pick("data_source", setting_default("global_settings.data_source"))
    if data_source not in VALID_SOURCES:
        log.warning(f"data_source '{data_source}' is not one of {VALID_SOURCES}; "
                    "the run will reject this.")

    mode = pick("matching_mode", "custom")
    if mode == "perfect":
        sizing = PERFECT_SIZING
    else:
        skew_params = pick("skew_params", {})
        sizing = SIZING_TEMPLATE.format(
            proportions=_yaml_list(pick("proportions", [])),
            balance=_yaml_scalar(pick("balance", "balanced")),
            skew_rule=_yaml_scalar(pick("skew_rule", "geometric")),
            skew_params=_optional_block(
                ["    skew_params:"]
                + [f"      {k}: {_yaml_scalar(v)}" for k, v in skew_params.items()]
                if skew_params else []),
        )

    # --- optional clm_label blocks ------------------------------------------

    target_metric = pick("target_metric", {})
    competing_noise = pick("competing_noise", [])

    concentrated = pick("concentrated_labels", [])
    concentrated_block = _optional_block(
        [(f"    concentrated_labels: {_yaml_list(concentrated)}"
          "  # spillover_rule 'concentrated' only")] if concentrated else []
    )

    competing_lines: list[str] = []
    if competing_noise:
        competing_lines = [
            "",
            "    # STRUCTURED COMPETING NOISE (single/custom only). Converts a share of ONE",
            "    # cluster's UNCLAIMED points to one competing label, at most what that",
            "    # label has left under spillover_rule 'proportional_to_marginal'.",
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
        [f"      steepness: {_yaml_scalar(steepness)}  # exponential profile only"]
        if steepness is not None and pick("centroid_profile", "linear") == "exponential" else []
    )

    follow_lines: list[str] = []
    feature_dependence = pick("feature_dependence", [])
    if feature_dependence:
        follow_lines = [
            "",
            "    # 6. LABELS THAT FOLLOW A FEATURE (single/custom only). Each entry places the",
            "    # points the matching gives one label by one feature column: favors 'high',",
            "    # 'low', or 'value' with a value; profile 'linear' or 'step'. Not together",
            "    # with centroid_dependence enabled.",
            "    feature_dependence:",
        ]
        for entry in feature_dependence:
            inner = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in entry.items())
            follow_lines.append(f"      - {{{inner}}}")
    follow_block = _optional_block(follow_lines)

    single_match_yaml = format_yaml_snippet(pick("single_match", {}))
    assignment_matrix_yaml = format_yaml_snippet(pick("assignment_matrix", []))

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
            # As written: a quoted "false" is refused by the run, not read as true here.
            f"\n  standardize: {_yaml_scalar(pick('standardize', False))}"
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
        source_labeling=_yaml_scalar(str(pick("source_labeling", "labels0"))),
        label_seed=pick("label_seed", 42),

        num_classes=_yaml_scalar(pick("num_classes", 4)),
        sizing=sizing,
        matching_mode=_yaml_scalar(mode),
        single_match=single_match_yaml,
        assignment_matrix=assignment_matrix_yaml,
        split_rule=_yaml_scalar(pick("split_rule", "proportional_to_size")),
        spillover_rule=_yaml_scalar(pick("spillover_rule", "proportional_to_marginal")),
        centroid_enabled=_yaml_scalar(pick("centroid_enabled", False)),
        centroid_profile=_yaml_scalar(pick("centroid_profile", "linear")),
        centroid_favors=_yaml_scalar(pick("centroid_favors", "core")),

        # Optional blocks: '' unless the payload asked for them.
        concentrated_labels=concentrated_block,
        competing_noise=competing_block,
        target_metric=target_block,
        steepness=steepness_block,
        feature_dependence=follow_block,
    )

    try:
        config = yaml.safe_load(rendered_yaml)
    except yaml.YAMLError as e:
        # A payload value the template cannot hold: a bad payload, as any other.
        log.critical(f"Rendered configuration is not valid YAML: {e}")
        sys.exit(2)
    _check_config(config["label_generation"]["clm_label"])

    # The suffix picks the format, as clmsynth reads it. JSON has no comments, so
    # the template's stay behind.
    text = (json.dumps(config, indent=2, ensure_ascii=False) + "\n"
            if Path(output_path).suffix.lower() == ".json" else rendered_yaml)
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(output_path, text)
        log.info(f"Test data configuration successfully written to '{output_path}'")
    except OSError as e:
        log.critical(f"Failed to write configuration file: {e}. "
                     f"'{output_path}' was left unchanged.")
        sys.exit(1)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """The payload path and the optional output path."""
    parser = argparse.ArgumentParser(
        prog="clmsynth-config",
        usage="%(prog)s [-h] payload [output]",
        description="Render a pipeline configuration from a payload file.")
    # Optional here: a missing payload is reported by main(), after the PyYAML check.
    parser.add_argument("payload", nargs="?",
                        help="Payload file, a YAML mapping. The repository has an example, "
                             "docs/examples/hepta_payload.yaml.")
    parser.add_argument("output", nargs="?", default="test_data_config.yaml",
                        help="Where the configuration is written; JSON when the path ends "
                             "in .json (default: test_data_config.yaml).")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point: ``clmsynth-config <payload.yaml> [output.yaml]``.

    Loads the upstream payload from the given YAML file and renders the pipeline
    config (default ``test_data_config.yaml``; JSON when the output path ends in
    ``.json``).
    """
    # Parsed first: --help and a mistyped option print before any log line.
    args = _parse_args(sys.argv[1:])
    configure_cli_logging()
    try:
        yaml = import_optional_dependency("yaml", "clmsynth-config")
    except ImportError as e:
        log.critical(f"{e}.")
        sys.exit(1)
    if args.payload is None:
        log.critical("Usage: clmsynth-config <payload.yaml> [output]. The repository has an "
                     "example payload, docs/examples/hepta_payload.yaml; clmsynth-wizard "
                     "needs no payload.")
        sys.exit(2)
    payload_path = args.payload
    output_path = args.output

    try:
        with open(payload_path, encoding="utf-8") as file:
            payload = yaml.safe_load(file)
    except FileNotFoundError:
        log.critical(f"Payload file '{payload_path}' not found. Pass a payload path, "
                     f"or run clmsynth-wizard instead.")
        sys.exit(2)
    except yaml.YAMLError as e:
        log.critical(f"Payload file '{payload_path}' is not valid YAML: {e}")
        sys.exit(2)
    except (OSError, UnicodeDecodeError) as e:
        # A folder, a file that cannot be opened, or text that is not UTF-8.
        log.critical(f"Payload file '{payload_path}' cannot be read as UTF-8 text: {e}")
        sys.exit(2)

    if not isinstance(payload, dict):
        log.critical(f"Payload file '{payload_path}' must contain a YAML mapping.")
        sys.exit(2)

    if os.path.exists(output_path) and os.path.samefile(payload_path, output_path):
        log.critical(f"Output path '{output_path}' is the payload file; writing the config "
                     "there would replace the payload. Pass a different output path.")
        sys.exit(2)

    try:
        generate_base_config(upstream_data=payload, output_path=output_path)
    except KeyboardInterrupt:
        log.warning("Interrupted; nothing written.")
        sys.exit(130)
    except (AttributeError, TypeError) as e:
        # Raised while rendering, before anything is written.
        log.critical(f"Payload file '{payload_path}': a value has the wrong shape "
                     f"({type(e).__name__}: {e}). target_metric and skew_params take a "
                     "mapping, competing_noise and feature_dependence a list of mappings. "
                     "Nothing was written.")
        sys.exit(2)
    except (ValueError, MissingConfigKey) as e:
        if getattr(e, "code", None) is None:
            raise
        log.critical(f"Payload file '{payload_path}': {e} Nothing was written.")
        sys.exit(2)


if __name__ == "__main__":
    main()
