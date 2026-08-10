# CLMSynth Roadmap

Planned work from 0.6.6 to 1.0.0, for a description of what the software does *today*,
refer to  **`README.md`**. For a record of past changes, refer to **`CHANGELOG.md`.** 

## Conventions

Every version below ships code. A patch release (`0.0.x`) fixes behavior that is already
specified; a minor release (`0.x.0`) ships a capability that did not exist before.

`[CLM-###]` codes are a public contract: never renumbered, never reused. Adding
one is therefore additive, and growing the registry is on its own enough to make
a release a minor rather than a patch, except a fix.

---

## 0.6.6

**Carrying columns that are neither features nor clusters.**

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
- a name in `tag_columns` that is absent from the file should be an error, not a
  silent no-op, on the same reasoning as `cluster_column`;
- naming the cluster column as a tag is a contradiction and should say so;
- the manual's "Known limitation" note under the BYOC import requirements comes
  out when this lands.

### Also in this release

- **Path-valued configuration keys get the guard their names already have.**
  0.6.4 refused path-shaped battery and dataset names, on the stated grounds that
  "a config is a shareable artifact here: reproducing someone's results means
  running a YAML you did not write". That reasoning applies equally to the
  *directory* those names are joined to, and `global_settings.output_dir` and
  `byoc_suite.input_dir` are still taken verbatim. The asymmetry is the finding:
  a config guards the stem but not the folder.

  Impact is small and bounded by construction, `build_run_dir` uses
  `mkdir(exist_ok=False)` so no existing file can be overwritten, and the BYOC
  import requirements constrain what a redirected `input_dir` can read. It is
  scheduled as tidiness and consistency rather than as a defect, which is why it
  is here and not a patch release.

  Shape to settle while implementing: whether a guard *rejects* an absolute or
  traversing directory outright, or resolves it and requires containment under
  the working directory. Rejecting is simpler to explain and to test; resolving
  is friendlier to the legitimate case of an `output_dir` on another volume.
  Whichever is chosen, `SECURITY.md`'s "partially guarded today" paragraph is
  rewritten when it lands.

---
## 0.6.7

### Graphical abstract, statistics and CLMSynth-GUI reference

- **A generated control-flow graph of the pipeline and the engine.** The
  pipeline is a fixed sequence, config → fetch → context → label generation →
  engine → metrics → output, and the engine is a sequence too, from label totals
  through matching rules, feasibility, allocation, spillover and placement. Both
  are worth drawing, and the drawing is worth generating from the source rather
  than maintaining by hand. It is a separate experimental tool rather than part
  of this package, so it carries no version here.


---

## 0.7.0

**Reachability: the MCC ceiling.**

Today the structural ceiling is a documented fact and nothing more. The closed
form `MCC = sqrt(M(M-1) / (K(K-1)))`, the ceiling of a balanced `M`-coarsening
over `K` equally sized clusters, verified exact is implemented **nowhere in `src/`**. A
user who asks for `MCC = 0.6` with 4 labels over 20 clusters (ceiling `0.178`)
gets a 40-iteration search, a best-effort labeling and a `[CLM-306]`
non-convergence warning that names neither the ceiling nor the reason.

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
- **The ceiling surfaced in the wizard.**
  Surface both: refuse or warn *before* searching when the request provably
  exceeds the closed-form bound, and name the reachable value in `[CLM-306]` when
  the search falls short. The wizard-side presentation of this is 0.6.7's;
  what belongs here is the engine computing the number in the first place.

  Prior art for the feasibility half exists in the project's unpublished
  research workspace, in a Dirichlet targeting study whose `max_feasible_alpha`
  and `size_matched_rules` compute the largest globally feasible recall from
  capacities and budgets by pure arithmetic, with no generation step. The method
  carries over even though that code does not ship.

---

**A more intuitive wizard.**

`config_wizard.py` asks the right questions, but the question flow is welded to
`input()` and to the order the options happen to appear in the config schema.
The two are worth separating: what is asked, from how it is asked.

### Added

- **A declarative question schema**, one entry per question carrying prompt,
  type, default, help text and a visibility predicate. The `explain=` text 
  is currently unreachable from
  anything but a terminal. Making the flow data rather than control flow makes
  it reorderable, testable without driving stdin, and is the same groundwork any
  future graphical front-end needs.

- **The structural ceiling surfaced at the point of asking.** The wizard
  currently requests a target metric with no idea whether it is achievable, and
  its own explanatory text already warns that "a target can be impossible for
  your data (e.g. MCC=1 with fewer labels than clusters)" without being able to
  say so concretely. The closed form is a pure relationship between `K` and `M`,
  so this needs no data, no fetch and no call into the engine, once both numbers
  are in hand it is one arithmetic expression evaluated inline. Depends on the
  engine-side work in 0.7.0 for the reachable (as opposed to closed-form) bound.

  *Deferred, not blocking:* where `K` comes from for the registry sources. The
  wizard always knows `M`, and already knows `K` for `byoc`, where
  `_peek_cluster_count` reads the cluster column of the first CSV. For
  `clustbench` and `mdcgen` the dataset is named but not yet fetched or
  generated, so `K` has to come from a static table, an on-demand fetch, or the
  wizard stays silent. Settle it when the feature is built; it does not change
  the shape of anything above.

### Scope note

The wizard's guided path already mitigates several uncoded paths at the config
layer, as of 0.6.1: `num_classes` is floored at 2 (closing the `M=1`
`ZeroDivisionError` and the `M=1` single-mode rejection), and `scope: pair` pins
`proportional_to_marginal` spillover (closing a `[CLM-130]` path). Those are
guards in `config_wizard.py` only. Handwritten and library configs bypass them
entirely, which is why the engine still needs its own diagnostics regardless of
what this release does.

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

### Added

- **A documented batch entry point**, with `03_isolation` extended to cover it.

### Performance

  This is the first concrete payoff from the N2 fix.

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
