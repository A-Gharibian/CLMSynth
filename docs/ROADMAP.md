# CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.6.8-brightgreen)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.6.9 to 1.0.0, for a description of what the software does *today*,
refer to  **`../README.md`**. For a record of past changes, refer to **`../CHANGELOG.md`.** 

## Conventions

Every version below ships code. A patch release (`0.0.x`) fixes behavior that is already
specified; a minor release (`0.x.0`) ships a capability that did not exist before.

`[CLM-###]` codes are a public contract: never renumbered, never reused. Adding
one is therefore additive, and growing the registry is on its own enough to make
a release a minor rather than a patch, except a fix.

## 0.6.9

### Reachability: the MCC ceiling

- **The closed-form ceiling at the point of asking.**
  `MCC = sqrt(M(M-1) / (K(K-1)))` is arithmetic on two integers *from where `K` comes*:
  `byoc` knows it directly, but for `clustbench` and
  `mdcgen` it must come from a static table (a fetch is what the rule-based
  constraint forbids) or the wizard stays silent. The engine-side reachable
  ceiling was always 0.7.0's; see there.

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
    solve and currently discarded. `[CLM-306]` reports `best_metric` without ever
    framing it as a ceiling, and nothing reports it at all when the solve
    succeeds.
- **The ceiling acted on, not just computed.** Refuse or warn *before* searching
  when the request provably exceeds the closed-form bound, and name the reachable
  value in `[CLM-306]` when the search falls short. What belongs *here* is the
  engine computing and reporting the number. Presenting the closed-form bound at
  the moment a user is asked for a target is the wizard's, deferred out of 0.6.7
  to the 0.6.8/0.7.0 ceiling work, and bounded by the wizard's rule-based
  constraint.


### To Be Fixed 
- label_generator.py:50
  legacy noise mode crashes on a single-cluster ground truth.
  rng.choice(classes[classes != base[idx]]) gets an empty array when classes has one element, 
  the no-clm_label path (byoc_source.py:108). BYOC rejects single-cluster files, but no other source does.

---

## 0.7.0

### columns that are neither features nor clusters.

BYOC's premise is that a user brings the feature subset their clustering was
computed in, plus the cluster column. In practice, they often have more: an
outcome variable, a second labeling from another method, an identifier. The
intention has always been that those travel with the data without entering the
CLM machinery, as "other tags".

There is no way to say so. `byoc_source` treats **every** non-cluster column as
a feature, so such a column joins the geometry and shifts centroid placement
without anyone noticing. That is also why 0.6.4's import requirements insist all
non-cluster columns are numeric: with no way to declare a passenger, anything
present must be a feature.

- **A `byoc_suite.tag_columns` declaration**, listing columns carried through to
  the output CSV untouched and excluded from the feature set. Unlike the import
  requirements this is a *feature*, not a guard: it accepts input that is
  currently mishandled rather than rejecting input that is wrong.

  Implementation consequences:

  - tag columns are exempt from the numeric requirement, since a tag is usually a
    string, but not from the reserved-name or duplicate-name checks;
  - a name in `tag_columns` that is absent from the file should be an error, not
    a silent no-op, on the same reasoning as `cluster_column`;
  - naming the cluster column as a tag is a contradiction and should say so;
  - the manual's "Known limitation" note under the BYOC import requirements comes
    out when this lands.

### Will be fixed:
#### `B101`, `B404`, `B603`

Recorded with their individual reasons in `[tool.bandit]` in `../pyproject.toml`:
two `assert`s guarding internal allocation invariants that no configuration can
reach, and the wizard's `subprocess.run`, whose argument vector is built from
`sys.executable` and a path the wizard itself just wrote, with no shell and no
user string reaching `argv`.

#### Two contradicting policies for non-numeric columns.
- byoc_source.py:121-127 vs 190-199: 
  validate_import rejects the whole file for any non-numeric feature column; fetch_byoc_data then has a warn-and-drop
  branch that can never fire. The one case where they disagree is boolean columns: is_numeric_dtype(bool) is True
  (passes validation) but select_dtypes([np.number]) excludes bool (gets dropped).

#### Fixes and Modifications during article review will be applied to this release.


---

## 0.8.0, 

**Parallel-safe batch execution and performance release**

### Fixed

- **matplotlib/seaborn plotting is not thread-safe.**
  `plot_feature_scatter` calls `sns.set_theme()` and uses the pyplot
  current-figure stack, both process-global. From six concurrent threads it
  produced 5 of 6 PNGs; one failed inside the function's own handler with "main
  thread is not in main loop", the default TkAgg backend requiring
  `plt.subplots()` on the main thread. Interpreter shutdown then logged
  `Tcl_AsyncDelete: async handler deleted by the wrong thread`, Tcl/Tk state
  corruption at the C level, not merely a caught warning.

  *Fix:* `matplotlib.use("Agg")` at import in `visualization.py`. Agg is
  thread-safe for file output and changes nothing on the single-threaded path.

- **Logging is a shared sink the moment there is more than one worker**, and it
  is the one parallel-safety hazard never characterized at all: no test in any
  category touches it. Three consequences, all invisible until the per-dataset
  loop is no longer sequential:

  - `main()` installs one root handler via `logging.basicConfig`. Under workers
    that handler is shared, so partial lines interleave and no message can be
    attributed to the worker that emitted it.
  - Messages carry a dataset name only *sometimes*. A `[CLM-304]` from one
    dataset reads identically to one from another, so in a batch the warning
    telling a user their achieved label counts deviate from the configured
    proportions cannot be traced to the dataset it concerns.
  - Per-run log files versus a run/dataset id threaded into the record: still
    open, and worth deciding rather than discovering.

  *Extends the assembly point 0.6.6 built rather than replacing it.* That release
  added `cli_logging.py` and put a filter on the formed record for CR/LF
  scrubbing; attribution is another rule at the same seam, which is why the two
  were split this way rather than done at once. Scheduled here because a worker
  id has nothing to identify until workers exist, and the interleaving cannot be
  tested without them.

### Added

- **A documented batch entry point**, with `03_isolation` extended to cover it.
- Possible build for PyPI (which will be released for version 1, see below)

### Performance

- **The catalog generator runs 54 subprocesses one at a time**, at ~5.5 s
  (error path) to ~7.8 s (full run with plots) each. It was
  serial because `build_run_dir` was check-then-act and two concurrent runs could
  be handed one folder. **0.6.3 closed that**; each case already writes to its own
  `_scratch/CLM-###`, and every registry is an import-time constant, so a process
  pool is safe now.


- **`import clmsynth` costs ~3.2 s, and ~2.0 s of that is scikit-learn and scipy
  a run may never use.** Measured with `-X importtime`:

  |                                                                       | cumulative |
  |-----------------------------------------------------------------------|------------|
  | `clmsynth`                                                            | 3.20 s     |
  | ├ `clmsynth.metrics` (`sklearn.metrics` 1.22 + `scipy.optimize` 0.75) | 1.98 s     |
  | └ `pandas`                                                            | 0.89 s     |

  `clm_label_engine` imports `.metrics` at module level to build `_METRIC_FUNCS`,
  and `__init__.py` re-exports the metric functions, so every invocation pays it.
  But metrics are only *used* when `target_metric` is set, or at the end of a
  pipeline run: a config error aborts long before one is computed and still pays
  the full two seconds.

  Deferring those imports into the functions that use them, with a PEP 562
  `__getattr__` in `__init__.py` so the public re-exports stay lazy, removes it.
  Library callers benefit most, `generate_clm_labels` currently waits for
  scikit-learn it may never touch. Caveat: four public names become lazily bound,
  so `dir(clmsynth)` differs before first access. Worth a test asserting they
  still resolve.
#### Backward-compatible approach (eager on Python <3.15, lazy on 3.15+):
`__lazy_modules__ = ["json", "pathlib"]`

`
import json      # Deferred on 3.15+`

`
import pathlib   # Deferred on 3.15+`

`
import sys       # Eager on all versions`


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
anything here produces `c(x)` and `X` *before* the engine ever sees them.
No new dependency, no architectural boundary to defend, just new rules
inside one existing module. The open question is which mathematical properties
ship as defaults versus staying user-configurable.

### Added

- **`fabricated_generator`** overhaul: the implementation of the module is from a historical
  code written for a different pipeline, and although feature generation is not a goal of
  this package, but emitting more than one ground-truth column can be useful. **Clustering benchmark** 
  datasets already ship multiple reference labeling
  (`GroundTruth_labels0`, `GroundTruth_labels1`, …), and `DatasetContext` already
  consumes any number of them generically, `build_context` collects every
  `GroundTruth_*` column. The offline fabricator produces one per run.

  Extending it to emit several labeling derived from different mathematical
  properties of the same synthetic feature space (a radial split, a linear
  combination, a nonlinear boundary) lets the offline source mimic that
  multi-labeling structure without touching `label_context.py` or the engine.
  This is a necessity to explore all possible outcomes for a real-world
  datasets, characterizing a dataset by more than one structural criterion at
  once.

- **SYNLABEL, behind an optional extra.** SYNLABEL derives a noiseless functional
  labeling from a feature space and injects measured noise by resampling features.
  Useful here because `fabricated_generator.py`'s current ground truth is a single
  rule, either a percentile split on `Feature_1` or a distribution, and a SYNLABEL-derived labeling
  would be a principled synthetic ground truth for the offline source. Possible noise addition
  to fabricated data is being explored. Shipped as an opt-in extra rather than a core dependency.

### Fixed

- Changing the naming for Clustering Benchmark from 
 **[clusterbench](https://github.com/clusterbench/clusterbench)** 
 (a Jakarta EE clustering benchmark application with a repository active from 2025-08-24) to `clustering_benchmark` 
 throughout the software.

- **dataset_sources.py:37-59 vs 77-110** 
  the `clustering_benchmark` registry disagreement.

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
- **Release workflow**, triggered by a tag: build the sdist and wheel with
  `python -m build`, check with `twine check`, publish to PyPI via **Trusted
  Publishing** (OIDC) so no long-lived API token lives in repository secrets.

- **Release preconditions.** the release workflow must refuse to
  publish when the tag and the declared version disagree, which is a check that
  cannot exist until there is something to publish to.

- **Signed tags, from the 1.0 tag onward.** OIDC has PyPI verify
  that an artifact came from this repository's workflow, and publishing that way
  generates PEP 740 attestations automatically. Signing tags is the complement:
  it attests the source tree, Trusted Publishing attests the built artifact.

  **conda-forge needs neither.** The feedstock verifies a `sha256` of the
  published sdist.

- **The two documented non-extras stay non-extras.** `mdcgenpy` is only available
  as a git repository and PyPI rejects uploads whose metadata carries direct-URL
  dependencies; `pyivm` pins `numpy<2.0`, which would downgrade this project's
  numpy during resolution. Both remain README install instructions.
  - A fork of `pyivm` working with more recent version of `numpy` can be considered here.

- **Archival.** 0.6.0 is on Zenodo with a DOI; the 1.0 tag should get its own, and
  `CITATION.cff` updated to point at it.

### Added

- **Python 3.15 support.** `requires-python` changed to `>=3.12,<3.16`, classifier
  added, CI matrix extended to five interpreters.

### Changed

- **Stability commitment.** The `[CLM-###]` registry and the public `__all__`
  become interfaces under semantic versioning: no renumbering, no
  behavior change to an existing code without a major bump. Codes are already
  treated this way in practice, `clm_errors.py` states it.

### Resolved

- **`evaluate_cluster_label_matching` decided either way.** It is exported in
  `__all__`, documented as a "provisional, not-yet-implemented" internal-validity
  hook, and never called by the pipeline; `pyivm` is an unlisted optional import
  that cannot be a proper extra because its `numpy<2.0` pin would downgrade the
  project's own numpy. This is the last tag that a decision to include it will be
  considered.

---

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
