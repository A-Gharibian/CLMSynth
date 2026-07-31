# Changelog

All notable changes to CLMSynth are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
