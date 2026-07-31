# Changelog

All notable changes to CLMSynth are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-07-31

Wizard and config-renderer polish. No change to the label-generation math.

### Added

- **`generate_config.py` can now emit every documented `clm_label` option.**
  `skew_params`, `concentrated_labels`, `competing_noise`, `target_metric`
  (including `scope`, `tolerance`, `max_iter`) and `centroid_dependence.steepness`
  previously had no template slot, so the payload path could not express features
  the manual documents in full — they could only be added by hand-editing the
  rendered YAML. Each block renders only when the payload carries it, so an
  existing payload produces the same config as before.
- **Render-time warnings for configs the engine will reject**, matching the
  existing `skew_rule`/`data_source` checks: `target_metric` outside
  `single`/`custom` (`[CLM-111]`/`[CLM-114]`), `scope: pair` without `type: mcc`
  *and* `matching_mode: single` (`[CLM-123]`/`[CLM-124]`), and `competing_noise`
  under `random` (`[CLM-115]`).
- **Wizard now asks for `spillover_rule` in `single` mode.** It was a
  `custom`-only question, so a `single` run silently took the default even though
  spillover governs every point the rule does not claim.
- **Wizard now asks for `concentrated_labels`** when `spillover_rule:
  concentrated` is chosen, instead of leaving the engine to fall back to the
  single largest label.

### Fixed

- **`generate_config` raised `NameError` on import under Python 3.11–3.13.**
  The new optional-block renderers annotated `List[str]` without importing
  `List` from `typing`. Python 3.14 defers annotation evaluation (PEP 649) so
  the module imported cleanly there, masking the fault on the only interpreter
  it was exercised with; on 3.11–3.13 annotations are evaluated at definition
  time and `clmsynth-config` failed outright. Caught by static inspection, not
  by running it.
- **The `mdcgen` source could never run.** `fetch_mdcgen_data` did
  `import mdcgenpy` and then reached for `mdcgenpy.clusters.ClusterGenerator`,
  but importing a package does not bind its submodules, so every run failed with
  `module 'mdcgenpy' has no attribute 'clusters'`. Because the access sat inside
  the generic generation-failure handler, it was reported as a data-generation
  error rather than an import problem. Now imports
  `from mdcgenpy.clusters import ClusterGenerator` directly, and the import
  guard reports the underlying `ImportError` so a missing package and a changed
  package layout are distinguishable.
- **The wizard printed a command that fails.** On completion it suggested
  `python main.py <config>`, which raises `ImportError` under the package layout;
  it now prints `python -m clmsynth.main <config>`. The same stale form is
  corrected in the `main.py`, `generate_config.py` and `config_wizard.py`
  docstrings and in `upstream_payload.yaml`, all of which now also name the
  console-script equivalent.
- **The wizard crashed on empty input at the cluster-id prompts.** Pressing Enter
  at `single_match.cluster` or a `competing_noise` cluster raised an uncaught
  `IndexError`; an empty cluster list in a `custom` rule was accepted silently and
  surfaced much later as a confusing `[CLM-150]` (zero capacity). All three
  prompts now re-ask.
- README: a sentence in the `[CLM-309]` limitation had been severed mid-clause,
  and a stray `****` rendered as literal asterisks in the `competing_noise` entry.

## [0.3.0] — 2026-07-31

A documentation-consistency pass ahead of the first public release. No change to
the label-generation math: every `[CLM-###]` code, its trigger, and its message
text behave exactly as in 0.1.0.

### Added

- `clustering_mcc_pair` is now exported from the package's public API
  (`clmsynth.__all__`), alongside `clustering_mcc` and `clustering_ari`. It is
  the 2x2 Matthews phi that `target_metric.scope: pair` inverts in closed form,
  so callers can now verify that result directly rather than reaching into
  `clmsynth.metrics`.
- `maintainers` field in `pyproject.toml`, distinguishing shared authorship of
  the method from sole ownership of the program.
- This changelog.

### Changed

- **`fabricated_data` cluster ids are now integers `0..K-1`**, matching
  `clustbench` and `mdcgen`. They were strings (`"Class_0"`, …), which meant a
  `clm_label` config's `clusters:` / `single_match.cluster` values had to be
  retyped when moving between sources. The conversion happens in the source
  adapter (`fetch_fabricated_data`); `fabricated_generator` still emits readable
  string labels, so its own standalone CSV output is unchanged.
  `test_data_config_offline.yaml` updated accordingly.
- Authorship recorded as Arootin Gharibian and Milos Kudelka; the program is
  written and maintained by Arootin Gharibian alone.
- `pyivm` is now documented consistently as **not implemented yet**.
  `metrics.evaluate_cluster_label_matching` remains a provisional hook that no
  part of the pipeline calls. Previously the README both asserted and denied
  that the dependency had been verified.
- `dataset_sources.py` module docstring now describes all four sources
  (`byoc` was missing) and notes that the registry shape is the extension point.

### Fixed

- Repository URL in `pyproject.toml` corrected to
  `https://github.com/A-Gharibian/CLMSynth`; it previously pointed at an
  unrelated repository.
- `[CLM-309]`'s explanation corrected in the README and the engine comment. The
  probe and output streams differ because they are seeded differently
  (`default_rng(probe_seed)` vs `default_rng(seed)`), not because the label-count
  draw had advanced the run stream — that only happens under
  `skew_rule: dirichlet`, and is now described as the secondary effect it is.
- Stale `dummy` naming replaced with `fabricated_data` (the offline source's
  current name) in the troubleshooting reference and `requirements.txt`.
- Removed pointers to a `Latex/main.tex` that does not exist; the diagnostics
  registry now correctly cites `Manual/troubleshooting.tex`. The accompanying
  article will be linked here once published.
- Raw BibTeX keys in the README replaced with readable prose citations.
- Stale internal codename ("Layer1") removed from
  `solve_alpha_for_target_metric`'s docstring.

### Removed

- The erroneous-configuration catalog. Its logs had been captured from a
  pre-package layout and no longer matched the current engine — stale module
  paths, the old `dummy` source name, and in one case a raw `IndexError` where
  `[CLM-102]` now fires. The manual's reference to it is commented out, not
  deleted, pending a regenerated catalog.
- A duplicate `requirements` file (byte-identical to `requirements.txt`).

## [0.2.0]

Internal, unpublished. Repository and packaging housekeeping; no functional or
user-facing changes.

## [0.1.0]

Initial release state: the CLM label engine, four interchangeable data sources
(`clustbench`, `mdcgen`, `fabricated_data`, `byoc`), the coded `[CLM-###]`
diagnostics registry, the config wizard and template renderer, MCC/ARI
evaluation, and scatter-plot output.
