# SPDX-License-Identifier: MIT
# config_files.py
"""Reading a config file (`load_config`, public as ``clmsynth.load_config``), and
parsing the text of a config or a payload file, for the run and clmsynth-config.
The parsers' own errors pass through; UnusableDocument is what they let through
but neither tool can use. Loads no numpy or pandas."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from .compat import import_optional_dependency

__all__ = [
    "MAX_ALIAS_NODES",
    "ConfigFileError",
    "UnusableDocument",
    "is_json",
    "load_config",
    "parse_json",
    "parse_yaml",
]

# How many nodes YAML aliases may add to a document: a few hundred bytes of
# aliases can otherwise expand to gigabytes in an error message or a written config.
MAX_ALIAS_NODES: Final = 10_000


class UnusableDocument(ValueError):
    """A document that parses, but whose aliases expand it too far, whose text is not
    valid Unicode, or one of whose mappings holds a key twice."""


def parse_json(text: str) -> Any:
    """The JSON document in `text`; None when it holds only whitespace."""
    data = json.loads(text, object_pairs_hook=_once_each) if text.strip() else None
    _check_unicode(data)
    return data


def _once_each(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """One JSON object; UnusableDocument for a key written twice, which json would
    read as its last value."""
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise UnusableDocument(f"the key {key!r} is written twice in one mapping; "
                                   "keep one")
        data[key] = value
    return data


def parse_yaml(text: str, yaml: Any) -> Any:
    """The YAML document in `text`, as `yaml.safe_load` reads it, PyYAML being
    passed in as `yaml`; None when it is empty."""
    loader = yaml.SafeLoader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_nodes(node, yaml)
        data = loader.construct_document(node)
    finally:
        loader.dispose()
    _check_unicode(data)
    return data


def _check_nodes(root: Any, yaml: Any) -> None:
    """UnusableDocument when a mapping holds a key twice, or when aliases add more than
    MAX_ALIAS_NODES nodes. An alias is the node it names again, so the walk stops as
    soon as the count passes."""
    seen: set[int] = set()
    visits = 0
    stack = [root]
    while stack:
        node = stack.pop()
        visits += 1
        first = id(node) not in seen
        seen.add(id(node))
        if visits - len(seen) > MAX_ALIAS_NODES:
            raise UnusableDocument(
                f"its aliases (*name) expand it by more than {MAX_ALIAS_NODES} values; "
                "write the values out or use fewer aliases")
        if isinstance(node, yaml.MappingNode):
            if first:
                _check_keys_once(node, yaml)
            for key, value in node.value:
                stack += (key, value)
        elif isinstance(node, yaml.SequenceNode):
            stack += node.value


# A merge key (<<) brings in another mapping's keys; it is not a setting of its own.
_MERGE: Final = "tag:yaml.org,2002:merge"


def _check_keys_once(node: Any, yaml: Any) -> None:
    """UnusableDocument for a key written twice in one YAML mapping, which PyYAML
    would read as its last value. Keys are compared as written with their type, so
    `1` and `"1"` are two keys."""
    lines: dict[tuple[str, str], int] = {}
    for key, _ in node.value:
        if not isinstance(key, yaml.ScalarNode) or key.tag == _MERGE:
            continue
        line = key.start_mark.line + 1
        if (key.tag, key.value) in lines:
            raise UnusableDocument(
                f"the key {key.value!r} is written twice in one mapping (line "
                f"{lines[key.tag, key.value]} and line {line}); keep one")
        lines[key.tag, key.value] = line


def _check_unicode(data: Any) -> None:
    """UnusableDocument for text that UTF-8 cannot hold, such as the lone surrogate
    a JSON escape like \\ud800 gives."""
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as e:
                raise UnusableDocument(
                    f"it holds text that is not valid Unicode ({e.reason}) in "
                    f"{item[:40]!r}") from None
        elif isinstance(item, Mapping):
            stack += item.keys()
            stack += item.values()
        elif isinstance(item, (list, tuple)):
            stack += item


def is_json(path: str | Path) -> bool:
    """A .json config is read and written with the standard library, any other
    with PyYAML, which is optional."""
    # Windows opens 'cfg.json ' and 'cfg.json.' as 'cfg.json'.
    return Path(str(path).rstrip(" .")).suffix.lower() == ".json"


class ConfigFileError(ValueError):
    """A config file that is absent, malformed or empty."""


def load_config(config_path: str) -> dict[str, Any]:
    """Loads the pipeline config. ConfigFileError, a ValueError, if it is absent,
    malformed or empty; ImportError if a YAML config finds no PyYAML."""
    path = Path(config_path)
    if not path.is_file():
        raise ConfigFileError(f"Configuration file not found at '{config_path}'. "
                              f"Run clmsynth-wizard or clmsynth-config first.")
    # utf-8-sig drops a BOM (Notepad, PowerShell 5.1), as PyYAML does and json does not.
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        # PowerShell 5.1's `>` writes UTF-16.
        raise ConfigFileError(f"Configuration file '{config_path}' is not UTF-8 text "
                              f"({e.reason} at byte {e.start}). Save it as UTF-8.") from None
    # Each parser's message carries the line and column of the mistake.
    if is_json(path):
        try:
            config = parse_json(text)
        except UnusableDocument as e:
            raise ConfigFileError(f"Configuration file '{config_path}': {e}.") from None
        except json.JSONDecodeError as e:
            raise ConfigFileError(
                f"Configuration file '{config_path}' is not valid JSON: {e}") from None
        except (ValueError, RecursionError) as e:
            # A number longer than Python converts, or nesting deeper than it follows.
            raise ConfigFileError(
                f"Configuration file '{config_path}' cannot be read as JSON: {e}") from None
    else:
        try:
            yaml = import_optional_dependency("yaml", f"Reading {config_path!r}")
        except ImportError as e:
            raise ImportError(f"{e}. A .json config needs nothing more.") from None
        try:
            config = parse_yaml(text, yaml)
        except UnusableDocument as e:
            raise ConfigFileError(f"Configuration file '{config_path}': {e}.") from None
        except yaml.YAMLError as e:
            raise ConfigFileError(
                f"Configuration file '{config_path}' is not valid YAML: {e}") from None
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as e:
            # A value PyYAML cannot build, such as the date 2026-13-45, or nesting
            # deeper than it follows.
            raise ConfigFileError(
                f"Configuration file '{config_path}' holds a value that cannot be read as "
                f"YAML ({type(e).__name__}: {e}).") from None
    # An empty file parses to None; anything non-mapping means run_single's config.get(...)
    if not isinstance(config, dict):
        raise ConfigFileError(f"Configuration file '{config_path}' is empty or not a mapping "
                              "of settings.")
    return config
