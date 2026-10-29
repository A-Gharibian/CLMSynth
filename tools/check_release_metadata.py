#!/usr/bin/env python
"""Every declaration of the version and the release date has to agree."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tomllib
from typing import Any

import yaml

# The repository root is the nearest folder above this script holding
# pyproject.toml, so the script keeps working wherever it is moved.
REPO = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "pyproject.toml").is_file())

# PyPI name -> conda-forge name, where they differ.
CONDA_NAMES = {"matplotlib": "matplotlib-base"}
# A conda recipe disagreement is a warning up to this version and an error after it.
CONDA_WARN_THROUGH = (0, 7, 9)

def _capture(pattern: str, text: str, flags: int = 0) -> str:
    """sentinel"""
    m = re.search(pattern, text, flags)
    return m.group(1) if m else "(absent)"


def read_yaml(text: str) -> Any:
    """A YAML file with every scalar as text, as written: a date or a version
    stays the string the file holds."""
    return yaml.load(text, Loader=yaml.BaseLoader)


def read_recipe(text: str) -> dict[str, list[str]]:
    """The values of a conda recipe by dotted key path (`about.summary`). A list
    item is filed under its list's path, and a mapping inside a list adds its keys
    to that path (`tests.python.imports`)."""
    found: dict[str, list[str]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for item in node:
                walk(item, path)
        elif node not in (None, ""):
            found.setdefault(path, []).append(node)

    walk(read_yaml(text), "")
    return found


def _requirements(specs: list[str]) -> dict[str, str]:
    """conda-forge name -> version constraint without spaces, from PyPI or conda specs."""
    out = {}
    for spec in specs:
        m = re.match(r"\s*([A-Za-z0-9][\w.-]*)\s*(.*)$", spec)
        if m:
            name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
            out[CONDA_NAMES.get(name, name)] = m.group(2).replace(" ", "")
    return out


def recipe_faults(text: str) -> tuple[str, list[str]]:
    """The version a conda recipe pins, and where the recipe disagrees with
    pyproject.toml or the package."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    recipe = read_recipe(text)
    pinned = (recipe.get("context.version") or recipe.get("package.version") or ["(absent)"])[0]
    faults = []

    def says(key: str) -> str:
        return ", ".join(recipe.get(key, [])) or "(absent)"

    urls = project.get("urls", {})
    python_min = re.search(r">=\s*([\d.]+)", project.get("requires-python", ""))
    restated = {
        "about.homepage": urls.get("Homepage"),
        "about.repository": urls.get("Repository"),
        "about.documentation": urls.get("Documentation"),
        "about.summary": project.get("description"),
        "about.license": project.get("license"),
        "about.license_file": ", ".join(project.get("license-files", [])),
        "context.python_min": python_min.group(1) if python_min else None,
    }
    for key, want in restated.items():
        if want and says(key) != want:
            faults.append(f"{key}: pyproject says {want!r}, the recipe says {says(key)!r}")
    if not recipe.get("about.description"):
        faults.append("about.description is missing")

    for key, specs, what in (
        ("requirements.host", pyproject.get("build-system", {}).get("requires", []),
         "a build requirement"),
        ("requirements.run", project.get("dependencies", []), "a dependency"),
    ):
        want_reqs = _requirements(specs)
        have_reqs = _requirements(recipe.get(key, []))
        for name, spec in want_reqs.items():
            if have_reqs.get(name) != spec:
                faults.append(f"{key} {name}: pyproject says {spec!r}, "
                              f"the recipe says {have_reqs.get(name, '(absent)')!r}")
        for name in sorted(have_reqs.keys() - want_reqs.keys() - {"python", "pip"}):
            faults.append(f"{key} {name}: not {what} in pyproject")

    scripts = project.get("scripts", {})
    entry_points = {}
    for entry in recipe.get("build.python.entry_points", []):
        name, _, target = entry.partition("=")
        entry_points[name.strip()] = target.strip()
    for name in sorted(scripts.keys() | entry_points.keys()):
        if scripts.get(name) != entry_points.get(name):
            faults.append(f"entry point {name}: pyproject says "
                          f"{scripts.get(name, '(absent)')!r}, "
                          f"the recipe says {entry_points.get(name, '(absent)')!r}")

    for module in recipe.get("tests.python.imports", []):
        path = REPO / "src" / module.replace(".", "/")
        if not (path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()):
            faults.append(f"tests.python.imports {module}: the package has no such module")
    return pinned, faults


def collect() -> tuple:
    """version, version declarations, dates, changelog date, faults, doi."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    found = {"pyproject.toml": version}

    init = (REPO / "src/clmsynth/__init__.py").read_text(encoding="utf-8")
    found["__init__.py"] = _capture(r'__version__\s*=\s*"([^"]+)"', init)

    # Read as fields, not grepped: a quoted value ('2026-09-17') or a DOI under
    # `identifiers` is the same fact as its plain or top-level form.
    cff = read_yaml((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    found["CITATION.cff"] = cff.get("version", "(absent)")

    codemeta = json.loads((REPO / "codemeta.json").read_text(encoding="utf-8"))
    found["codemeta.json"] = codemeta.get("version", "(absent)")
    found["codemeta.json softwareVersion"] = codemeta.get("softwareVersion", "(absent)")
    tag = re.search(r"/v([^/]+)\.tar\.gz$", codemeta.get("downloadUrl", ""))
    found["codemeta.json downloadUrl"] = tag.group(1) if tag else "(unparsable)"

    # codemeta.softwareRequirements restates pyproject.
    declared = {}
    for entry in codemeta.get("softwareRequirements", []):
        if isinstance(entry, dict) and entry.get("name"):
            declared[entry["name"]] = entry.get("version", "")
    # The software DOI is declared twice.
    # In CITATION.cff it sits under `identifiers` (type: doi) or at the top level;
    # preferred-citation.doi is the article's, never the software's.
    cff_dois = [i.get("value") for i in cff.get("identifiers") or []
                if isinstance(i, dict) and i.get("type") == "doi"]
    cff_doi = str(cff_dois[0] if cff_dois else cff.get("doi", "(absent)"))
    cm_doi = codemeta.get("identifier", "(absent)")
    doi_faults = []
    if cff_doi != cm_doi:
        doi_faults.append(f"CITATION.cff declares {cff_doi}, codemeta.json declares {cm_doi}")
    if not cm_doi.startswith("10."):
        doi_faults.append(f"codemeta.json identifier {cm_doi!r} is not a bare DOI")

    dep_faults = []
    for spec in project.get("dependencies", []):
        name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip()
        want = spec[len(name) :].strip()
        if name not in declared:
            dep_faults.append(f"{name} is a dependency but codemeta does not declare it")
        elif declared[name] != want:
            dep_faults.append(f"{name}: pyproject says {want!r}, codemeta says {declared[name]!r}")

    for tex in sorted((REPO / "docs").rglob("*.tex")):
        head = tex.read_text(encoding="utf-8", errors="replace")[:2000]
        m = re.search(r"^%\s*!CLMSynth-version\s*=\s*(\S+)", head, re.M)
        if m:
            found[tex.name] = m.group(1)

    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    dated = re.search(rf"^##\s*\[{re.escape(version)}\]\s*[-—]\s*(\S+)", changelog, re.M)

    dates = None
    if dated:
        dates = {
            # Text either way: quoted or not, the date reads as written.
            "CITATION.cff date-released": cff.get("date-released", "(absent)"),
            "codemeta.json dateModified": codemeta.get("dateModified", "(absent)"),
        }
    return (
        version,
        found,
        dates,
        dated.group(1) if dated else None,
        dep_faults + doi_faults,
        cff_doi,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="git tag being released, e.g. v0.7.0")
    parser.add_argument("--recipe", type=pathlib.Path,
                        help="a conda recipe (recipe.yaml, v1 format) to check against pyproject")
    args = parser.parse_args()

    version, found, dates, release_date, dep_faults, doi = collect()
    for name, value in found.items():
        print(f"  {name:45s} {value}")
    disagree = {n: v for n, v in found.items() if v != version}

    stale = {}
    if dates is not None:
        for name, value in dates.items():
            print(f"  {name:45s} {value}")
        stale = {n: v for n, v in dates.items() if v != release_date}

    if disagree:
        print(f"\nversion disagreement against pyproject's {version}")
        for name, value in disagree.items():
            print(f"  {name} declares {value}")
    if release_date is None:
        print(f"\nCHANGELOG.md has no dated [{version}] entry")
    if stale:
        print(f"\ndate disagreement against the CHANGELOG's {release_date}")
        for name, value in stale.items():
            print(f"  {name} declares {value}")

    print(f"  {'software DOI (CITATION.cff, codemeta.json)':45s} {doi}")

    conda_faults: list[str] = []
    if args.recipe is not None:
        pinned, conda_faults = recipe_faults(args.recipe.read_text(encoding="utf-8"))
        print(f"  {'conda recipe pins':45s} {pinned}")

    if dep_faults:
        print("\nmetadata disagreements")
        for fault in dep_faults:
            print(f"  {fault}")

    release = tuple(int(n) for n in re.findall(r"\d+", version)[:3])
    conda_wrong = bool(conda_faults) and release > CONDA_WARN_THROUGH
    if conda_faults:
        through = ".".join(map(str, CONDA_WARN_THROUGH))
        print("\nconda recipe disagreements"
              + ("" if conda_wrong else f" (warnings through {through})"))
        for fault in conda_faults:
            print(f"  {fault}")

    tag_wrong = False
    if args.tag is not None:
        expected = f"v{version}"
        print(f"\n  {'tag being released':45s} {args.tag}")
        print(f"  {'expected from pyproject':45s} {expected}")
        tag_wrong = args.tag != expected
        if tag_wrong:
            print(f"\ntag {args.tag} does not match the declared version {version}")

    return 1 if (disagree or release_date is None or stale or tag_wrong or dep_faults
                 or conda_wrong) else 0


if __name__ == "__main__":
    sys.exit(main())
