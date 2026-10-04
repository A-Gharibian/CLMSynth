# Test modules

[![Version](https://img.shields.io/badge/version-0.7.2-brightgreen)](https://github.com/A-Gharibian/CLMSynth/releases)
<!-- 
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/{owner}/{repo}/badge)](https://scorecard.dev/viewer/?uri=github.com/{owner}/{repo})
-->

## Main tests
Located in the main tests directory; refer to README for instructions.
Running in under a minute.

| module                  | defends                                                              |
|-------------------------|----------------------------------------------------------------------|
| `test_smoke`            | First pass                                                           |
| `test_00_contract`      | right input produces right output, sensitive                         |
| `test_01_logic`         | named suspicions and previous bugs, selective                        |
| `test_02_edge_cases`    | input edge cases: nothing, one, many, degenerate, duplicates, non-ASCII, BYOC import rules |
| `test_03_isolation`     | ownership of state: RNG, config, run folder, module registries, logging, plotting imports |
| `test_04_failure_modes` | the pipeline degrades rather than aborts, reports it, and exits with a distinguishable code |
| `test_05_config_safety` | the program's own defenses against a configuration it did not author |
| `test_06_diagnostics`   | the `[CLM-###]` registry safety net, driven by the catalog           |
| `test_07_text_wizard`   | the wizard's schema agrees with the engine; its target floor is wizard-only |

The suite needs no network and no optional dependency: `clustbench` fetches
  are patched, and every case runs against the offline `fabricated_data` source
  or against the engine directly. `pytest` is a development dependency declared
  as the `test` extra, not a runtime one, and the suite is not shipped in the
  sdist. Install the package with the test extra and run the suite:

```bash
pip install -e ".[test]"
```

```bash
pytest
```

No arguments: `[tool.pytest.ini_options]` in `pyproject.toml` points to `tests/`.
No test needs the network: each runs against the engine directly, the offline
`fabricated_data` source, local BYOC CSV files, or a `clustbench` fetch with the
download patched out.


## Reproducibility tests (upcoming)

The reproducibility tests must pass for every major release (N.x.x) and are actionable errors if they fail to validate.

## Data generated for publication (upcoming)

The results of the data generated for the accompanying article (*SoftwareX*, 2026) are archived in Zenodo (embargoed).
The code producing those results is not public yet.

## STATIC CI tests

For the static analysis CI gates to run, install the dev extra instead:

```bash
pip install -e ".[dev]"
```

```bash
ruff check src/ tests/ && mypy && bandit -c pyproject.toml -r src/clmsynth -q
```

All three should report nothing. Where a finding was traced and
deliberately exempted, the exemption is recorded with its reason next to the
rule it exempts, in `[tool.ruff.lint]` and `[tool.bandit]` in `pyproject.toml`.

CI runs the suite on Python 3.12–3.14, the static analysis above, and a
packaging job that builds the distributions. The suite itself checks that both
shipped manuals declare the current package version (`test_06_diagnostics`).


## Testing policy

**Ship criterion.** A testing script ships as pytest only if it is deterministic (no
wall-clock timing races, no unbounded network), fast (sub-second to a few
seconds), and asserts a standing invariant or regression rather than documenting a
one-time investigation.

**`06_diagnostics` is the safety net, not the owner.** Registry coverage is a
**union** property across the whole suite: a code asserted in `01_logic` or
`02_edge_cases` is covered and does not need repeating, and overlap is fine. 06
exists so no code falls through; every code that needs testing and has
no home elsewhere gets one there. Coded assertions are not to be stripped out of
the other modules to centralize them.

**Characterisation tests are a feature.** A test may assert current *broken*
behavior on purpose, so that fixing the defect turns it red and the red is the
prompt to invert the assertion. None does today: the last one, which pinned that
`proportions` given as a mapping got no coded diagnostic, went red when 0.7.2
refused it with `[CLM-134]`, and was replaced by
`test_01_logic::test_proportions_given_as_a_mapping_are_refused`.

**What CI deliberately does not run.** Four stay out of the gate
and out of the tests folder:

- **Regression against the manual's Test Data table and the article's results
  tables.** Article and manual material, deferred to CLMSynth-GUI, where a
  front end for research use is being introduced.
- **Property and fuzz tests over the config surface.** Valuable for *finding*
  new uncoded paths, which is what the 0.7.2 robustness work uses them for, but a
  search that discovers something new on run 300 is not a gate.
- The research tests are deliberately not published in this repository; the
  results they produced are archived in Zenodo (embargoed).
- The catalog is self-contained and portable, so any case can
  be reproduced directly:

      cd docs/troubleshooting_catalog
      python -m clmsynth.main ValueError_1xx/CLM-150.yaml