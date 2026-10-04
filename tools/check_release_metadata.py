#!/usr/bin/env python
"""Every declaration of the version and the release date has to agree."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tomllib

import yaml

# The repository root is the nearest folder above this script holding
# pyproject.toml, so the script keeps working wherever it is moved.
REPO = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "pyproject.toml").is_file())

def _capture(pattern: str, text: str, flags: int = 0) -> str:
    """sentinel"""
    m = re.search(pattern, text, flags)
    return m.group(1) if m else "(absent)"


def collect() -> tuple:
    """version, version declarations, dates, changelog date, faults, doi."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    found = {"pyproject.toml": version}

    init = (REPO / "src/clmsynth/__init__.py").read_text(encoding="utf-8")
    found["__init__.py"] = _capture(r'__version__\s*=\s*"([^"]+)"', init)

    # Read as YAML, not line by line: a quoted value ('2026-09-17') or a DOI
    # under `identifiers` is the same fact as its plain or top-level form.
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8")) or {}
    found["CITATION.cff"] = str(cff.get("version", "(absent)"))

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
            # str(): unquoted, YAML reads it as a date object; quoted, as a string.
            "CITATION.cff date-released": str(cff.get("date-released", "(absent)")),
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

    if dep_faults:
        print("\nmetadata disagreements")
        for fault in dep_faults:
            print(f"  {fault}")

    tag_wrong = False
    if args.tag is not None:
        expected = f"v{version}"
        print(f"\n  {'tag being released':45s} {args.tag}")
        print(f"  {'expected from pyproject':45s} {expected}")
        tag_wrong = args.tag != expected
        if tag_wrong:
            print(f"\ntag {args.tag} does not match the declared version {version}")

    return 1 if (disagree or release_date is None or stale or tag_wrong or dep_faults) else 0


if __name__ == "__main__":
    sys.exit(main())
