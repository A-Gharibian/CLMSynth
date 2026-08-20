# Tests modules
### version 0.6.8

A record of test modules are kept here for reference.

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
