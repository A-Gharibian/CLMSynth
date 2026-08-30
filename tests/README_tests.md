# Tests modules

[![Version](https://img.shields.io/badge/version-0.7.0-brightgreen)](https://github.com/A-Gharibian/CLMSynth/releases)
<!-- 
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/{owner}/{repo}/badge)](https://scorecard.dev/viewer/?uri=github.com/{owner}/{repo})
-->

## Main tests
located on the main tests directory, refer to README for instructions.

| module                  | defends                                                              |
|-------------------------|----------------------------------------------------------------------|
| `test_smoke`            | First pass                                                           |
| `test_00_contract`      | right input produces right output, sensitive                         |
| `test_01_logic`         | named suspicions and previous bugs, selective                        |
| `test_02_edge_cases`    | input range edge cases: nothing, one, many, degenerate, non-ASCII    |
| `test_03_isolation`     | ownership of state: RNG, config, run folder, module registries       |
| `test_04_failure_modes` | the pipeline degrades rather than aborts, and reports it             |
| `test_05_config_safety` | the program's own defenses against a configuration it did not author |
| `test_06_diagnostics`   | the `[CLM-###]` registry safety net, driven by the catalog           |
| `test_07_text_wizard`   | the wizard's schema agrees with the engine                           |

Install the package with the test extra and run the suite:

```bash
pip install -e ".[test]"
```

```bash
pytest
```

No arguments: `[tool.pytest.ini_options]` in `pyproject.toml` points at `tests/`.
every test case runs against the
offline `fabricated_data` source or against the engine directly.


## Reproducibility tests (upcoming)

The reproducibility tests must pass for every major release (N.x.x) and are actionable errors if they fail to validate.

## Data generated for publication (upcoming)

The results of the data generated for the accompanying submitted paper is archived in Zenodo (embargoed).
The code producing those results will be public after a published version of the paper is available.

## STATIC CI tests

For the static analysis CI gates on, install the dev extra instead:

```bash
pip install -e ".[dev]"
```

```bash
ruff check src/ tests/ && mypy && bandit -c pyproject.toml -r src/clmsynth -q
```

All three are expected to report nothing. Where a finding was traced and
deliberately exempted, the exemption is recorded with its reason next to the
rule it exempts, in `[tool.ruff.lint]` and `[tool.bandit]` in `pyproject.toml`.

CI runs the suite on Python 3.11–3.14, the static analysis above, and a
packaging job that builds the distributions and checks that the version agrees
across `pyproject.toml`, `__init__.py`, `CITATION.cff` and the shipped manuals.


## Testing policy

**Ship criterion.** A testing script ships as pytest only if it is deterministic (no
wall-clock timing races, no unbounded network), fast (sub-second to a few
seconds), and asserts a standing invariant or regression rather than documenting a
one-time investigation.

**`06_diagnostics` is the safety net, not the owner.** Registry coverage is a
**union** property across the whole suite: a code asserted in `01_logic` or
`02_edge_cases` is covered and does not need repeating, and overlap is fine. 06
exists so no code falls through, every code that needs testing and has
no home elsewhere gets one there. Coded assertions are not to be stripped out of
the other modules to centralize them.

**Characterisation tests are a feature.** Several tests assert current *broken*
behavior on purpose, so that fixing the defect turns them red and the red is the
prompt to invert the assertion. They are listed under the release that closes
each one.

**What CI deliberately does not run.** Two bodies of work stay out of the gate
and out of the repository:

- **Regression against the manual's Test Data table and the article's results
  tables.** Article and manual material, deferred to CLMSynth-GUI, where a
  front end for research use is being introduced. 1.0 pins the published numbers
  at seed 42, which is the release that promises they will not move.
- **Property and fuzz tests over the config surface.** Valuable for *finding*
  new uncoded paths, which is what 0.7.0 uses them for, but a search that
  discovers something new on run 300 is not a gate.
- The research tests are deliberately not published, and will be published
with the accompanying paper.
