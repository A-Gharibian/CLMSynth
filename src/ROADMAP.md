# CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.6.7-brightgreen)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.6.8 to 1.0.0, for a description of what the software does *today*,
refer to  **`README.md`**. For a record of past changes, refer to **`CHANGELOG.md`.** 

## Conventions

Every version below ships code. A patch release (`0.0.x`) fixes behavior that is already
specified; a minor release (`0.x.0`) ships a capability that did not exist before.

`[CLM-###]` codes are a public contract: never renumbered, never reused. Adding
one is therefore additive, and growing the registry is on its own enough to make
a release a minor rather than a patch, except a fix.

---


## 0.6.8

**Staging for [CLMSynth-GUI](https://github.com/A-Gharibian/CLMSynth-GUI).**

Two releases preparing this package to be driven by a separate graphical
front end, which lives in its own repository.

- **A generated control-flow graph of the pipeline and the engine.** The
  pipeline is a fixed sequence, config → fetch → context → label generation →
  engine → metrics → output, and the engine is a sequence too, from label totals
  through matching rules, feasibility, allocation, spillover and placement. Both
  are worth drawing, and the drawing is worth generating from the source rather
  than maintaining by hand. The generator is a separate experimental tool rather
  than part of this package, so it carries no version here.

### Will be fixed

- **spillover_rule**: uniform/concentrated silently drops proportions with no engine-level warning
  (only the wizard warns)
- K-dependent codes 102/125/127 abort the whole batch while 104/105 skip-and-continue.
- missing required keys → bare KeyError re-hit per dataset.


## 0.6.9

### Reachability: the MCC ceiling

- **The closed-form ceiling at the point of asking.**
  `MCC = sqrt(M(M-1) / (K(K-1)))` is arithmetic on two integers and so *could*
  live here, but it is deferred to 0.6.8/0.7.0 with the rest of the ceiling work,
  keeping this release to the schema move and the two guards. *Where `K` comes
  from* travels with it: `byoc` knows it directly, but for `clustbench` and
  `mdcgen` it must come from a static table (a fetch is what the rule-based
  constraint forbids) or the wizard stays silent. The engine-side reachable
  ceiling was always 0.7.0's; see there.

- **The ceiling while the wizard is running**
  A rule-based wizard can only carry
  the closed-form half (see 0.6.7). Whether the reachable half needs a design
  change to be presentable at configuration time, rather than only reported after
  a run, is a question the staging releases answer.

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

---

## 0.8.0, Parallel-safe batch execution and performance

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

### Performance


- **`import clmsynth` costs ~3.2 s, and ~2.0 s of that is scikit-learn and scipy
  a run may never use.** Measured with `-X importtime`:

  | | cumulative |
  |---|---|
  | `clmsynth` | 3.20 s |
  | ├ `clmsynth.metrics` (`sklearn.metrics` 1.22 + `scipy.optimize` 0.75) | 1.98 s |
  | └ `pandas` | 0.89 s |

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

- **The catalog generator runs 54 subprocesses strictly one at a time**, at ~5.5 s
  (error path) to ~7.8 s (full run with plots) each. It was
  serial because `build_run_dir` was check-then-act and two concurrent runs could
  be handed one folder. **0.6.3 closed that**; each case already writes to its own
  `_scratch/CLM-###`, and every registry is an import-time constant, so a process
  pool is safe now. 54 cases over 8 workers is under a minute.

### Scope note

This is the pipeline layer (`main.py`, `visualization.py`), not the CLM engine.
`generate_clm_labels` is already a pure function of its inputs and its seed.

---

## 0.9.0

**Headline: Source and generator extensions.**

Both items are scoped to the data-source layer, **not** the CLM engine. The
boundary the project rests on, clusters are fixed, read-only input, must hold:
anything here produces `c(x)` and `X` *before* the engine ever sees them.
No new dependency, no architectural boundary to defend, just new rules
inside one existing module. The open question is which mathematical properties
ship as defaults versus staying user-configurable.

### Added

- **`fabricated_generator` emitting more than one ground-truth column.** Real
  clustbench datasets already ship multiple reference labeling
  (`GroundTruth_labels0`, `GroundTruth_labels1`, …), and `DatasetContext` already
  consumes any number of them generically, `build_context` collects every
  `GroundTruth_*` column. The offline fabricator produces one per run.

  Extending it to emit several labeling derived from different mathematical
  properties of the same synthetic feature space (a radial split, a linear
  combination, a nonlinear boundary) lets the offline source mimic that
  multi-labeling structure without touching `label_context.py` or the engine at
  all. The plumbing already supports it. This is the territory Gagolewski's
  benchmarking framework and the adjusted internal-validity-measure work
  (Gagolewski, 2022, SoftwareX; Jeon et al., 2025, TPAMI) explore for real
  datasets, characterizing a dataset by more than one structural criterion at
  once, applied here to a synthetic one.

- **SYNLABEL, behind an optional extra.** SYNLABEL derives a noiseless functional
  labeling from a feature space and injects measured noise by resampling features.
  Useful here because `fabricated_generator.py`'s current ground truth is a single
  crude rule, a percentile split on `Feature_1`, and a SYNLABEL-derived labeling
  would be a far more principled synthetic ground truth for the offline source.

  Shipped as an opt-in extra rather than a core dependency, alongside the existing
  `faker` / `mdcgenpy` / `pyivm` precedent, can be
  dropped without a breaking change. Vetting must
  complete before 1.0 either way.

---

## 1.0.0, 14 October 2026

**Headline: Python 3.15 and conda.**

Ships after Python 3.15 is released and tested, with support declared and
verified across **3.11 – 3.15**.

### Distribution

**The first published release.** Everything before this is a source release: a
tag and a GitHub archive, installed with `pip install -e .` from a clone. 1.0
adds **PyPI** (so `pip install clmsynth` works) and **conda**.

- **Release workflow**, triggered by a tag: build the sdist and wheel with
  `python -m build`, check with `twine check`, publish to PyPI via **Trusted
  Publishing** (OIDC) so no long-lived API token lives in repository secrets. A
  TestPyPI dry run precedes the first real upload, and the name `clmsynth` needs
  confirming as available before anything else here matters.

- **Release preconditions.** These were the manual pre-flight checks until
  0.6.5, and CI's packaging job now asserts all three on every push: the version
  agrees across `pyproject.toml`, `CITATION.cff`, `__init__.py` and both shipped
  `.tex` banners; `CHANGELOG.md` carries a dated entry matching it; and the built
  sdist contains what `MANIFEST.in` says it does, verified by unpacking the
  artifact rather than by reading the manifest.

  What 1.0 adds is binding them to the *tag*: the release workflow must refuse to
  publish when the tag and the declared version disagree, which is a check that
  cannot exist until there is something to publish to.

- **Signed tags, from the 1.0 tag onward.** Nothing is signed today, and until
  1.0 nothing needs to be: `user.name` and `user.email` are unverified strings,
  but a source-only release from a solo repository has no consumer for a
  signature. 1.0 is the first release cut for people who will install it without
  reading it, and a release is cut from a tag, so the tag is the thing worth
  attesting.

  **SSH signing, not GPG.** It reuses the key already used to push, with no
  keyring, no expiry management and no revocation story to maintain, and GitHub
  marks it Verified identically. GitHub's *Vigilant mode* stays off until signing
  is consistent, since it retroactively marks every unsigned commit as
  Unverified, and a history that is entirely Unverified communicates nothing.

  **Trusted Publishing matters more, and is already above.** OIDC has PyPI verify
  that an artifact came from this repository's workflow, and publishing that way
  generates PEP 740 attestations automatically. That answers "can I trust this
  package" better than a commit signature does, and with no long-lived key to
  lose. Signing tags is the complement, not the substitute: it attests the source
  tree, Trusted Publishing attests the built artifact.

  **conda-forge needs neither.** The feedstock verifies a `sha256` of the
  published sdist, so signing buys nothing on that channel. Noted so the effort
  is not spent twice.

  Signing *commits* stays optional and is a separate decision: it becomes
  meaningful when more than one person can push, since that is when branch
  protection requiring signed commits has something to protect against.

- **The two documented non-extras stay non-extras.** `mdcgenpy` is only available
  as a git repository and PyPI rejects uploads whose metadata carries direct-URL
  dependencies; `pyivm` pins `numpy<2.0`, which would downgrade this project's
  numpy during resolution. Both remain README install instructions. A published
  package cannot smuggle either in, so the README's optional installation lines
  become load-bearing at 1.0 in a way they are not today.

- **conda-forge**, decided, over a personal `anaconda.org` channel. It is the
  standard route for a scientific package: community review, discoverability, and
  automatic rebuilds when a dependency moves. All seven runtime dependencies are
  already there, and the package is pure Python, so the recipe is `noarch: python`
 ,one build covering every platform and interpreter.

  `environment.yml` already mirrors the pinned versions and stays the
  develop-from-source route regardless.

- **Archival.** 0.6.0 is on Zenodo with a DOI; the 1.0 tag should get its own, and
  `CITATION.cff` updated to point at it.

### Added

- **Python 3.15 support.** `requires-python` widened to `>=3.11,<3.16`, classifier
  added, CI matrix extended to five interpreters.

- **Regression pinning of published numbers.** The manual's Test Data table and
  the article's results tables, asserted at seed 42, so a reported number can
  never drift silently. Deferred here from 0.6.5 deliberately: pinning published
  values belongs with the release that promises they will not move.

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
  project's own numpy. Shipping a documented-as-unimplemented function in a 1.0
  public API is the wrong signal. Either implement it against `pyivm` and solve
  the pin (vendor the three adjusted measures, or depend on a fixed upstream), or
  remove it from `__all__` and keep it internal until it works.

---

## Testing policy

**Ship criterion.** A script ships as pytest only if it is deterministic (no
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
  discovers something new on run 300 is not a gate. They stay research.
