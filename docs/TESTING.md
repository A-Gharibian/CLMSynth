# Test modules

[![Version](https://img.shields.io/badge/version-0.7.9-brightgreen)](https://github.com/A-Gharibian/CLMSynth/releases)

<!-- 
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/{owner}/{repo}/badge)](https://scorecard.dev/viewer/?uri=github.com/{owner}/{repo})
-->

## Main tests

Located in the main tests directory; refer to README for instructions.

| module                       | defends                                                                                     |
|------------------------------|---------------------------------------------------------------------------------------------|
| `test_smoke`                 | First pass                                                                                  |
| `test_00_contract`           | right input produces right output, sensitive                                                |
| `test_01_logic`              | named suspicions and previous bugs, selective                                               |
| `test_02_edge_cases`         | input edge cases, BYOC import rules                                                         |
| `test_03_isolation`          | ownership of state: RNG, config, run folder, logging, plotting imports                      |
| `test_04_failure_modes`      | the pipeline reports instead of aborts, exit codes, `--help`, mdcgen presets, dataset cache |
| `test_05_config_safety`      | the defensive checks for configurations                                                     |
| `test_06_diagnostics`        | the `[CLM-###]` registry safety net, synced with the catalog                                |
| `test_07_text_wizard`        | the wizard's schema                                                                         |
| `test_08_resolved_config`    | running a folder's config reproduces the run                                                |
| `test_09_feature_dependence` | labels that follow a feature: placement, refusals, warnings, the run                        |
| `test_11_metric_health`      | a target metric is delivered within tolerance, or the miss is reported (will be added)      |

The suite needs no network: downloads and mdcgenpy are stubbed where a case needs them,
the suite is currently not shipped in the sdist. Install the package with the test extra and run the suite:

```bash
pip install -e ".[test]"
```

```bash
pytest
```

## CLM Catalogue

62 CLM errors and warnings can be regenerated (reproduced) by this tool.
OUT_DIR defaults to docs/troubleshooting_catalog. Pass a different one to
generate a candidate catalog for comparison before filing an issue. It must be
absent, empty, or a catalog from an earlier run; anything else is refused before
a file is touched. The catalog is built beside it, in `.<OUT_DIR name>.new`, and
its folders replace OUT_DIR's only when every case gives its diagnostic and exit
code; otherwise, or when the run is stopped, OUT_DIR is as it was. Files in OUT_DIR
outside the generated folders are never touched.
--threaded runs the cases in a thread pool, one worker per CPU. Each case has its
own folder, so the catalog and the report are the same as a serial run's.

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

<!--
For an isolated environment for each check, install tox and run it from the
repository root:

```bash
pip install "tox>=4"
tox
```

This runs the tests, Ruff, mypy and Bandit on the current Python interpreter.
Use `tox -e test -- -q` to run only the suite, or `tox -e lint,typing,security`
for static checks. Tests treat warnings and unexpected xfail passes as errors,
capture logs at INFO level, and show a summary of non-passing results.
-->

The `test` and `dev` dependency groups also support tools such as uv:

```bash
uv sync --group dev
uv run pytest
```

The existing `test` and `dev` extras remain available for pip installs.

<!--
CI runs the suite on Python 3.12–3.15, the static analysis above, and a
packaging job that builds the distributions. The packaging job also runs
`tools/check_release_metadata.py`, which checks that both shipped manuals declare the
current package version. 
-->

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
behavior on purpose, so that fixing the defect can invert the assertion. Example:
`proportions` given as a mapping got no coded diagnostic when 0.7.2
refused it with `[CLM-134]`, now `[CLM-141]`, and was replaced by
`test_01_logic::test_proportions_given_as_a_mapping_are_refused`.

**What CI deliberately does not run.** Four stay out of the gate and out of the test folder:

- **Regression against the manual's Test Data table and the article's results
  tables.** Article and manual material, deferred to CLMSynth-GUI, where a
  front-end for research use is being introduced.
- **Property and fuzz tests over the config surface.** Valuable for *finding*
  new uncoded paths, which is what the 0.7.2 robustness work used them for.
- The catalog is self-contained and portable, so any case can
  be reproduced directly:

      cd docs/troubleshooting_catalog
      python -m clmsynth InfeasibleAllocation_15x/CLM-150.yaml --no-viz


## Reproducibility tests (upcoming)

The reproducibility tests must pass for every major release (N.x.x)
and are actionable errors if they fail to validate.

## Data generated for publication (upcoming)

The results of the data generated for the accompanying article (*SoftwareX*, 2026)
are archived in Zenodo (embargoed). The code producing those results is not public yet.
