# CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.0b1-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.0b1 to 1.0.0, for a description of what the software does *today*,
refer to  **`../README.md`**. For a record of past changes, refer to **`../CHANGELOG.md`.** 

## Conventions

Every version below ships code. A patch release (`0.0.x`) fixes behavior that is already
specified; a minor release (`0.x.0`) ships a capability that did not exist before.

`[CLM-###]` codes are a public contract: never renumbered, never reused. Adding
one is therefore additive, and growing the registry is on its own enough to make
a release a minor rather than a patch, except a fix.

## Article review

Reviewer comments on the accepted article are implemented **between 0.7.0b1 and
0.7.0rc1**. That is the whole of the window: anything arriving after rc1 is cut
goes to 0.7.1 or later, because rc1 is a freeze.

---

## 0.7.0rc1

**The PyPI release candidate.**

### Required before the freeze

- **The README carries the direct API call.** A short section showing the
  direct path, without the YAML or the wizard. The README is the documentation
  to *run* the program; the LaTeX manual in `./docs` is the authoritative manual.

---

## 0.7.0, article published version

Identical to rc1 apart from the version strings and the release date.

### Distribution

**The first published PyPI release**

- **Release workflow**, `.github/workflows/release.yml`, triggered by a `v*` tag:
  build the sdist and wheel with `python -m build`, check with `twine check`,
  install the built wheel into a clean environment.

- **Release preconditions.** The workflow refuses to publish when the tag and the
  declared version disagree. `tools/check_release_metadata.py` runs it in CI.

- **The documented non-extra stays a non-extra.** `mdcgenpy` is only available as
  a git repository and PyPI rejects uploads whose metadata carries direct-URL
  dependencies. This is now a standing packaging constraint, not a to-do.

- **Signed tags are not adopted here.** 0.7.0 through 0.9.0 publish with PEP 740
  attestations.

### Open, carried past this release

- **`tests/` is pruned from the sdist** (`MANIFEST.in`), so a downloaded sdist
  cannot run the suite that `pyproject.toml` points `testpaths` at.
- **The `main.py` split is half-done.** `plot_dataset(...)` exists at 0.7.0b1 as
  `_render_dataset_plots`; the other half of the 0.7.0a note, a `generate_dataset(...)`
  that yields the CSV data and metrics, still does not, and still carries no defect or
  trigger. It stays a shape until the batch wrapper needs it, at which point that
  wrapper's requirements specify it rather than this note.

---

## 0.8.0

### Reachability: the MCC ceiling

- **The closed-form ceiling at the point of asking.**
  `MCC = sqrt(M(M-1) / (K(K-1)))` is arithmetic on two integers *from where `K` comes*:
  `byoc` knows it directly, but for `clustbench` and
  `mdcgen` it must come from a static table (a fetch is what the rule-based
  constraint forbids) or the wizard stays silent.

- **The ceiling while the wizard is running**
  A rule-based wizard can only carry
  the closed-form half (see 0.6.7).

### Added

- **The ceiling computed and reported by the engine.** Two quantities:
  - the *closed-form* ceiling from `M` and `K` alone, available before any
    allocation runs;
  - the *actual reachable* ceiling for the configured rule set, which is the MCC
    achieved at full recall. The global solver already evaluates `alpha = 1.0`,
    `grid = np.linspace(0.0, 1.0, 11)` includes it, so this is computed on every
    solve and currently discarded. `[CLM-306]` reports `best_metric` without the ceiling.
- **The ceiling acted on, not just computed.** Refuse or warn *before* searching
  when the request provably exceeds the closed-form bound, and name the reachable
  value in `[CLM-306]` when the search falls short. 
  - Presenting the closed-form bound at the moment a user is asked for a target is the wizard.

### **Parallel-safe batch execution and performance release**

### Will be Fixed

- **The search grid cannot see inside a narrow feasible band.** Candidate recalls
  come from 11 grid points, so the interval between the last feasible one and the
  true boundary is never explored. The bracket needs widening.

- **`--no-viz`, and the import that has to move with it.** The 0.7.0b1 extraction
  makes skipping the render a one-line guard, but the saving is only partial while
  `main.py` imports `plot_feature_scatter` at module scope. The viz stack costs about
  0.42 s of import on top of what the rest of the package already pulls, and
  `import clmsynth.main` loads matplotlib and seaborn modules whether a
  plot is used. Deferring
  the import into `_render_dataset_plots` is what removes that, and it is not free:
  `conftest.py` and `test_04_failure_modes` patch `clmsynth.main.plot_feature_scatter`
  at nine sites and a deferred import would bypass every one of them. The tests are
  restated first, then the import moves. `config_wizard.py` already refuses this
  dependency on purpose and is the precedent to follow.

- **matplotlib/seaborn plotting is not thread-safe.**
  `plot_feature_scatter` calls `sns.set_theme()` and uses the pyplot
  current-figure stack, both process-global. From six concurrent threads it
  produced 5 of 6 PNGs; one failed inside the function's own handler with "main
  thread is not in main loop", the TkAgg backend then in use requiring
  `plt.subplots()` on the main thread. Interpreter shutdown then logged
  `Tcl_AsyncDelete: async handler deleted by the wrong thread`, Tcl/Tk state
  corruption at the C level, not merely a caught warning.

  *Fix:* `matplotlib.use("Agg")` at import in `visualization.py`. Agg is
  thread-safe for file output and changes nothing on the single-threaded path.

- **Logging is a shared sink the moment there is more than one worker**: no test and 
  three consequences:

  - `main()` installs one root handler via `logging.basicConfig`. Under workers
    that handler is shared, so partial lines interleave and no message can be
    attributed to the worker that emitted it.
  - Messages carry a dataset name only *sometimes*. A `[CLM-304]` from one
    dataset reads identically to one from another, so in a batch the warning
    telling a user their achieved label counts deviate from the configured
    proportions cannot be traced to the dataset it concerns.
  - Per-run log files versus a run/dataset id threaded into the record: still
    open.

### Added

- **A documented batch entry point**, with `03_isolation` extended to cover it.
- **A Sphinx site, deferred here on purpose.**

### Performance

- **The catalog generator runs 54 subprocesses one at a time**, at ~5.5 s
  (error path) to ~7.8 s (full run with plots) each. It was
  serial because `build_run_dir` was check-then-act and two concurrent runs could
  be handed one folder. **0.6.3 closed that**; each case already writes to its own
  `_scratch/CLM-###`, and every registry is an import-time constant, so a process
  pool is safe now.


### Scope note

This is the pipeline layer (`main.py`, `visualization.py`), not the CLM engine.
`generate_clm_labels` is already a pure function of its inputs and its seed.
Python release 3.15 may address lazy loading inherently, the fixes here mostly
concern Python releases `3.15`.

---

## 0.9.0

**Source and generator extensions.**

Both items are scoped to the data-source layer, **not** the CLM engine. The
boundary the project rests on, clusters are fixed, read-only input, must hold:
anything here produces `c(x)` and `X` *before* the engine.

### Added

- **`fabricated_generator`** overhaul: the implementation of the module is from a historical
  code written for a different pipeline, and although feature generation is not a goal of
  this package, but emitting more than one ground-truth column can be useful. 
  **Clustering benchmark** datasets already ship multiple reference labeling
  (`GroundTruth_labels0`, `GroundTruth_labels1`, …), and `DatasetContext` already
  consumes any number of them generically, `build_context` collects every
  `GroundTruth_*` column. The offline fabricator produces one per run. 
  - Extending it to fabricate several labeling derived from different distributions
    of the same synthetic feature space (a radial split, a linear
    combination, a nonlinear boundary) lets the offline source mimic that
    multi-labeling structure without touching `label_context.py` or the engine.
    This is a necessity to explore all possible outcomes for a real-world
    datasets, characterizing a dataset by more than one structural criterion at
    once.

- **SYNLABEL** SYNLABEL derives a noiseless functional
  labeling from a feature space and injects measured noise by resampling features.
  Useful here because `fabricated_generator.py`'s current ground truth is a single
  rule, either a percentile split on `Feature_1` or a distribution, and a SYNLABEL-derived labeling
  would be a principled synthetic ground truth for the offline source. 
  - **Clustering benchmark** has a noise addition function, which also can possibly utilised.

### Fixed

- **dataset_sources.py:38-76 vs 78-112** 
  the `clustbench` registry disagreement.

  SOURCE_METADATA["clustbench"] documents 5 batteries; CLUSTBENCH_DATASETS defines 9. 
  Resolve_selection reads the second, so batteries: "all" resolves to ~223 datasets including 144 g2mg/h2mg 
  while print_battery_info lists only 5, and fetch_clustbench_data:291 logs "not in the recommended list" for four
  batteries.

---

# 1.0.0
## 14 October 2026

**Python 3.15 and release for conda.**

Ships after Python 3.15 is released and tested, with supported Python versions **3.12 – 3.15**.

### Distribution

**The first published conda release.**

- **conda-forge**, All seven runtime dependencies are
  already there, one `noarch: python` build covering every platform and interpreter.

- **Signed tags, from the 1.0 tag onward.** OIDC has PyPI verify
  that an artifact came from this repository's workflow, and publishing that way
  generates PEP 740 attestations automatically. **conda-forge needs neither.**
  The feedstock verifies a `sha256` of the published sdist.

### Added

- **Python 3.15 support.** `requires-python` changed to `>=3.12,<3.16`, classifier
  added, CI matrix extended to four interpreters.

### Changed

- **Stability commitment.** The `[CLM-###]` registry and the public `__all__`
  become interfaces under semantic versioning: no renumbering, no
  behavior change to an existing code, so future runs will be reproducible.

### Resolved

- **`evaluate_cluster_label_matching` decided either way.** Removed in 0.6.8, with
  its optional `pyivm` import.
