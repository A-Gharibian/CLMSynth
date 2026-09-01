# CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.0-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.0 to 0.9.0, for a description of what the software does *today*,
refer to  **`../README.md`**. For a record of past changes, refer to **`../CHANGELOG.md`.** 

## Conventions

Every version below ships code. A patch release (`0.0.x`) fixes behavior that is already
specified; a minor release (`0.x.0`) ships a capability that did not exist before.

`[CLM-###]` codes are a public contract: never renumbered, never reused. Adding
one is therefore additive, and growing the registry is on its own enough to make
a release a minor rather than a patch, except a fix.

## Article review

Reviewer comments on the accompanying article are implemented **between 0.7.0b1 and
0.7.0rc2**.

---

## 0.7.1 and 0.7.2

**Correctness and diagnostics.** Nothing here is a new capability, so the series
stays patch-numbered even where a fix adds a `[CLM-###]` code.

### Will be Fixed, engine

- **Integer label counts must sum to `N`.** `_largest_remainder_counts` assumes its
  remainder is non-negative, while `[CLM-106]` accepts a `proportions` sum within
  `1e-6` of 1. Once `N` times that slack reaches 1 the remainder goes negative and
  the slice that distributes it runs backwards. It first bites at `N = 1000000`;
  the suite runs at `N = 1000` and the largest benchmark used is 6500, which is why
  no test sees it. A dataset of a million rows then receives a label column longer
  than itself.

- **Negative `proportions` are rejected.** A vector such as `[1.5, -0.25, -0.25]`
  sums to 1, passes `[CLM-106]` and `[CLM-121]`, and delivers a labeling in which
  the negative labels never appear. `[CLM-131]` already gates exactly this for
  `skew_params`; explicit `proportions` need the same gate, either as a new code or
  by widening `[CLM-131]`. Taken together with the `[CLM-121]` placement below,
  since both turn on where `proportions` are validated relative to the `perfect`
  override, and under `perfect` a negative vector currently delivers a correct
  bijection. `random` also fails here uncoded, which the same change closes.

- **Label ids must be integers.** `2.0` and `True` satisfy `label == int(label)`,
  pass `[CLM-104]` and `[CLM-118]`, and then reach a numpy index. YAML writes
  `label: 2.0` as a float, so a config file can produce an uncoded `IndexError`.
  `concentrated_labels` already rejects both.

- **Duplicate `competing_noise` entries.** Two entries naming one `(cluster, label)`
  sum their counts but keep only the last entry's `favors`, so half the request is
  placed at the wrong end of the cluster. Either merge them, reject the duplicate,
  or warn.

### Will be Fixed, diagnostics

The labeling is correct in every case below; only the message is wrong or absent.

- **A missing `label` key escapes uncoded.** `single_match` without `cluster` raises
  `[CLM-205]` and an `assignment_matrix` row without `clusters` raises `[CLM-207]`,
  but either without `label` raises a bare `KeyError`. The troubleshooting reference
  documents both sub-keys as covered.

- **`matching_mode: random` returns before three validators**, so `[CLM-128]` and
  `[CLM-129]` never fire there while `[CLM-125]` still does, in a mode that does no
  placement at all.

- **A valueless `assignment_matrix:` with a `target_metric`** raises `TypeError`
  rather than `[CLM-206]`, because the membership test above the solver uses a
  `.get` default that a present-but-null key does not trigger.

- **`[CLM-111]` pre-empts `[CLM-101]`**, so a misspelled `matching_mode` combined
  with a `target_metric` is reported as a target-metric incompatibility.

- **`[CLM-121]` fires under `perfect`**, where `[CLM-302]` states that
  `proportions` are ignored. The two codes give contradictory accounts.

### Will be Fixed, pipeline

Not the CLM engine, and no `[CLM-###]` codes: these describe file layout and
orchestration, which the registry does not cover.

- **A top-level config key present with no value.** `global_settings:`,
  `<source>_suite:` and `label_generation:` each parse to `None`, escape `main()`'s
  handler as an `AttributeError`, and leave behind the empty run folder that
  `discard_run_dir_if_barren` exists to remove.

- **The closing summary can report more unlabelled datasets than processed ones**,
  because the unlabelled counter is incremented before the processed one.

- **`fetch_clustbench_data` can raise `KeyError('labels0')`** when its own length
  filter drops `labels0` while a later labeling survives.

### Verification

- **A development build after the fixes land**, to confirm the suite, the metric
  health checks and the 2000-config fuzz sweep still pass before the series is
  tagged. A regression gate, not new coverage.

---

## 0.7.3 and 0.7.4

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

## 0.7.5 Release version

Will be released to PyPI.

---

## 0.7.6 and 0.7.7

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

- **SYNLABEL** derives a noiseless functional
  labeling from a feature space and injects measured noise by resampling features.
  Useful here because `fabricated_generator.py`'s current ground truth is a single
  rule, either a percentile split on `Feature_1` or a distribution, and a SYNLABEL-derived labeling
  would be a principled synthetic ground truth for the offline source. 
- **Repliclust** is a package that can generate clusters based on prompts.

### Fixed

- **dataset_sources.py:38-76 vs 78-112** 
  the `clustbench` registry disagreement.

  SOURCE_METADATA["clustbench"] documents 5 batteries; CLUSTBENCH_DATASETS defines 9. 
  Resolve_selection reads the second, so batteries: "all" resolves to ~223 datasets including 144 g2mg/h2mg 
  while print_battery_info lists only 5, and fetch_clustbench_data:291 logs "not in the recommended list" for four
  batteries.


## 0.7.8 and 0.7.9

**Python 3.15 support**

### Added

- `requires-python` changed to `>=3.12,<3.16`, classifier
  added, CI matrix extended to four interpreters.
Ships after Python 3.15 is released and tested, with supported Python versions **3.12 – 3.15**.

---

# 0.8.0
## 22 November 2026

### Distribution

**The first published conda release.**

- **conda-forge**, All seven runtime dependencies are
  already there, one `noarch: python` build covering every platform and interpreter.

- **Signed tags, from the 0.8.0 tag onward.**

### Changed

- **Stability commitment.** The `[CLM-###]` registry and the public `__all__`
  become interfaces under semantic versioning: no renumbering, no
  behavior change to an existing code, so future runs will be reproducible.

### Resolved

- **`evaluate_cluster_label_matching` decided either way.** Removed in 0.6.8, with
  its optional `pyivm` import.
---

# 0.9.0 series

Tracking and implementing upon community feedback/issues and cross-device/OS compatibility.
Pipeline integration improvements and CLMSynth-GUI development is planned for this release.

