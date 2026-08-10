# Changelog

All notable changes to CLMSynth are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.6] — 2026-08-11

A patch release.

### Added

- **A run states where it writes, before it writes.** `global_settings.output_dir`
  and `byoc_suite.input_dir` are now resolved to absolute paths and logged once
  per run, before the run folder is created or a single CSV is read.

  A report and **not** a restriction, and the reasoning is recorded
  in `SECURITY.md` so it is not relitigated. 0.6.4 refuses path-shaped battery
  and dataset names because a *name* is not supposed to be a path, so a
  separator in one is a category error. A directory setting **is** a path, so
  there is no category error to detect and every candidate restriction refuses
  something legitimate: scratch space on a cluster, an output volume,
  `../results`. The asymmetry is the correct outcome rather than a gap.

 `build_run_dir` creates a fresh
  timestamped folder with `mkdir(exist_ok=False)`, so no existing file can be
  overwritten. It also stays correct under
  parallel, cluster and pipeline execution, because it reports rather than
  prompts, never raises, once per run rather than once per dataset.

- **A configuration-safety test category**, `tests/test_05_config_safety.py`.
  The stated condition for its return was that the program itself implement a
  measure protecting the machine that runs a configuration. Two now exist, so
  the category does. Authorization, ReDoS, uncontrolled recursion, wall-clock timing.

### Fixed

- **A configuration value containing a newline could forge a log line.**
  Reported by CodeQL as `py/log-injection`. Configuration values reach log
  messages by design: a warning naming an unrecognised `skew_rule` has to quote
  it. A value carrying a newline split one record into what read as two, and the
  second could be shaped to look like a line the program never emitted,
  a fabricated `Pipeline ready. 99 dataset(s) processed.`, for instance. A
  configuration is a shareable artifact here, so the value need not have been
  written by whoever reads the output.

  The new `cli_logging.py` gives the package a single point where a record is
  finished, and `SingleLineFilter` escapes `\r` and `\n` there. Escaped rather
  than stripped: a visible `\n` says a newline was present and was neutralized,
  where deleting it would leave a plausible single line and hide the attempt.

  The filter attaches to the **handlers**, not to the `clmsynth` logger. A
  logger-level filter only sees records logged directly to it, records from
  child loggers reach ancestors through `callHandlers`, which consults ancestor
  handlers and never re-applies ancestor filters.

### Changed

- **Logging configuration belongs to the console scripts, not to the package.**
  `configure_cli_logging()` is called by `clmsynth` and `clmsynth-config`, and by
  nothing on import. A library running inside another process keeps its own
  handlers, format and levels; importing `clmsynth` configures nothing. This also
  settles a divergence: `main` set its own format while `generate_config` took
  `basicConfig`'s default, so the same workflow produced two line shapes.

- **CI actions moved to their Node 24 builds**, `actions/checkout` v4 → v7 and
  `actions/setup-python` v5 → v7.

- **Test taxonomy corrected.** The new measures were first filed under
  `03_isolation` on the strength of a convenient heading. Only the two genuinely
  about ownership of state stayed; the defenses moved to `05`, and
  "a report must not be why a run fails" moved to `04_failure_modes`, whose
  subject it is. Suite is 266 tests across eight modules.

- **`03_isolation`'s two identical pipeline configs are one factory.** Flagged as
  duplicate code by static analysis. Kept local to the module rather than hoisted
  into `conftest.py`: `run_pipeline(...) == 1` is only meaningful while that
  config names exactly one dataset, so a fixture shared across modules would let
  a change made elsewhere silently redefine what these tests check.

## [0.6.5] — 2026-08-10

The test suite becomes part of the repository.

### Added

- **A tracked pytest suite, `tests/`.** 257 tests across seven modules, running
  in well under a minute. It has existed for some time as a local workspace that
  was excluded from version control, which meant a fresh checkout carried no
  tests and CI had nothing to run.

  | module                  | defends                                                                     |
  |-------------------------|-----------------------------------------------------------------------------|
  | `test_smoke`            | the program runs at all and produces the output it should                   |
  | `test_00_contract`      | right input produces right output, the sensitive sweep                      |
  | `test_01_logic`         | named suspicions and shipped-once bugs, the selective pins                  |
  | `test_02_edge_cases`    | the ends of the input range: nothing, one, very many, degenerate, non-ASCII |
  | `test_03_isolation`     | ownership of state: RNG, config, run folder, module registries              |
  | `test_04_failure_modes` | the pipeline degrades rather than aborts, and reports it                    |
  | `test_06_diagnostics`   | the `[CLM-###]` registry safety net, driven entirely by the catalog         |

  The suite needs no network and no optional dependency: `clustbench` fetches
  are patched, and every case runs against the offline `fabricated_data` source
  or against the engine directly. `pytest` is a development dependency declared
  as the `test` extra, not a runtime one, and the suite is not shipped in the
  sdist.

- **Continuous integration that actually checks something.** The workflow was a
  placeholder that installed the package and imported it. It now runs three
  jobs on every push and pull request:

  - **Tests** across Python 3.11, 3.12, 3.13 and 3.14, the full range
    `requires-python` declares, without `fail-fast` so one interpreter failing
    alone is distinguishable from all of them failing.
  - **Lint, typing and security**: `ruff`, `mypy` and `bandit`, all three
    gating. Each was brought to zero findings before being turned on; a gate
    switched on over a non-zero baseline is one people learn to ignore.
  - **Packaging**: builds the sdist and wheel, runs `twine check`, unpacks the
    sdist to verify it contains what `MANIFEST.in` claims, and asserts the
    version agrees across `pyproject.toml`, `__init__.py`, `CITATION.cff` and
    both shipped `.tex` banners, with a dated `CHANGELOG.md` entry to match.
    Those last checks were performed by hand until now.

- **`ROADMAP.md` is published**, for the first time, next to the source it
  describes in `src/`. Planned work from 0.6.6 through 1.0.0.

- **`SECURITY.md`**, stating the threat model and how to report a vulnerability
  privately. CLMSynth is a local single-user CLI and library with no privilege
  boundary between the person supplying input and the person running it, which
  is why a path arriving from `sys.argv` or an interactive prompt is the
  interface rather than a vulnerability. The document says where that reasoning
  stops, a configuration file is a shareable artifact, so configuration values
  are the one input that can come from someone else, and records the standing
  verdict on each accepted scanner finding, including bandit's `B310`, so the
  judgments are not re-derived on every scan.

### Changed

- **The `[tool.bandit]` judgements are recorded in `pyproject.toml`**, with the
  reason for each, in the same form as the `ruff` exemptions added in 0.6.4.
  Four findings (`B101`, `B404`, `B603`, `B310`) were traced by hand and judged
  unexploitable under this package's threat model, a local single-user CLI and
  library with no auth boundary. `B310` in particular is revisited the moment
  URL-scheme restriction is implemented.

- **The test suite is laid out flat.** Each category was a directory containing
  exactly one module of the same name; the directory added a level that held
  nothing. `tests/00_contract/test_00_contract.py` is now
  `tests/test_00_contract.py`, and `smoke_test.py` follows the `test_*` prefix
  every other module already used.

### Fixed

- **A run that produced nothing left a run folder behind claiming it had.** The
  timestamped folder, its `csv/`, `png/` and `txt/` subfolders and the copy of
  the config are all created before the pipeline starts, because the run needs
  somewhere to write the moment the first dataset succeeds. When none did, that
  scaffolding survived: `OUTPUT` accumulated folders that were indistinguishable
  at a glance from successful runs, and anything globbing
  `OUTPUT/*/csv/*.csv` walked over runs that had produced no data.

  Reported against the wizard's `byoc` path, where it is easiest to reach, a
  config naming an input folder with no CSV in it is a reasonable thing to write,
  since the wizard configures a run rather than performing one, but it was never
  specific to `byoc`. Any run processing zero datasets did it: an unknown
  `data_source`, a missing `batteries` key, a source unreachable offline.

  Both failure exits now discard the folder, and only when it holds exactly the
  scaffolding and nothing else. A single written file, expected or not, means the
  folder stays untouched. This also makes `precheck_byoc_matching_ids` truthful:
  it documented itself as aborting "before any output is written", which was
  already false of the filesystem by the time it ran.

- **The sdist was missing the byoc catalog input.** `MANIFEST.in` included
  `docs/**` for `.tex`, `.yaml` and `.log` but not `.csv`, so
  `docs/troubleshooting_catalog/_data/k65_clusters.csv` never shipped, and the
  catalog's `byoc` entries could not be reproduced from a source distribution.
  Now asserted by CI rather than by reading the manifest.

- **`[tool.pytest.ini_options] testpaths` pointed at a folder that no longer
  existed**, a leftover from the staging layout, so a bare `pytest` collected
  nothing at all.

- **Two pinned versions in `requirements.txt` had fallen behind** what the
  project is verified against: `matplotlib` 3.11.0 → 3.11.1 and `pandas`
  3.0.3 → 3.0.5.

## [0.6.4] — 2026-08-09

A solved target metric is now the value actually delivered,
rather than a value the search reached on a stream the output
did not use.

### Added

- **Lint, typing and test configuration now live in `pyproject.toml`.** The
  project previously had no `[tool.ruff]`, `[tool.mypy]` or
  `[tool.pytest.ini_options]` section.

- **BYOC imports are checked against stated requirements.** BYOC is an import
  path, not a generator: the premise is that you clustered a subset of your own
  features and are bringing the result. A file is now rejected, with every
  problem reported in one pass rather than one per attempt, when it is empty, has
  duplicate column names, uses a name the pipeline reserves (`Cohort_Class`,
  `GroundTruth_*`, `Cluster_n`, `Label_n`), has missing values in the cluster
  column or in a feature, holds fewer than two clusters, contains a cluster of
  fewer than three points, or has a non-numeric feature column. The rule regarding
  number of points in a cluster will probably change or deleted in next release.

  The full list, with the reason for each, is in the troubleshooting reference
  under *Uncoded Rejections (data import)*. These deliberately carry no
  `[CLM-###]` code: the coded diagnostics describe the cluster-label matching
  model, while these describe whether the data is a usable clustering at all.

- **A `labels_only_4class` preset for the `fabricated_data` source**, cluster
  ids with no feature columns. The engine has always accepted such a dataset,
  because recall targets, class balance, allocation and spillover never look at
  coordinates, but until now no configuration could produce one, so the
  capability was reachable only from Python.

### Fixed

- **A solved `target_metric` is now the value actually delivered.** Under
  `scope: global` the solver scores candidate recall levels on one fixed random
  stream so that candidates compare fairly. The labeling that was written out,
  however, was generated on the run's own stream, and the two had drifted apart:
  allocation continued a stream that a dirichlet skew may already have consumed
  draws from, which no probe could reproduce. A search could therefore report
  success and deliver something outside tolerance by up to 0.07 on small
  datasets.

  Allocation now draws from its own stream, started from the run seed, and the
  search uses that same stream. The labeling the solver scored is byte-identical
  to the one written. What remains, and is reported by `[CLM-306]` as it always
  was, is the ordinary case of a target the data cannot reach; at small dataset
  sizes the achievable values form a coarse ladder and a tight tolerance can fall
  between two rungs.

  `[CLM-309]`'s message has been rewritten accordingly, it previously explained
  a cause that no longer exists.

- **Battery and dataset names that look like paths are refused.** These names are
  used to build file paths in both directions: `byoc` resolves
  `input_dir/<dataset>.csv` to read, and every source writes
  `csv/<source>__<battery>__<dataset>.csv`. A name containing a separator or
  `..` therefore reached outside both configured folders. The registry sources
  filter their names against a known list and were never exposed; `byoc` trusts
  the configuration's list verbatim, which is the route this closes.

### Changed

- `config_wizard` reports a non-zero exit code from the pipeline run it launches
  instead of discarding it, so a failed run no longer looks like a successful
  one.
- The troubleshooting reference gains a section for rejections that are not
  `[CLM-###]` diagnostics, and both shipped documents now carry a
  machine-readable banner so tooling locates them by marker rather than by
  filename, and can check that the documentation was updated for the release.
- Every `[CLM-###]` code now has a runnable catalog fixture, and the generator
  refuses to run if a registry code has no builder.

## [0.6.3] — 2026-08-08

A correctness release. Every entry below closes a case where the program either
gave an answer that was wrong without saying so, stopped with an error that did
not explain itself, or reported a number it had not actually delivered.

One new capability comes with it: a dataset can now consist of cluster ids with
no features at all.

### Added

- **A labels-only data source.** The CLM engine has always accepted a dataset
  with cluster ids and no feature space, because recall targets, class balance,
  allocation and spillover are all counting problems that never look at
  coordinates. Until now no configuration could produce such a dataset, every
  source emitted at least one feature column, so the capability was reachable
  only by calling the library from Python.

  The `fabricated_data` source gains a `labels_only_4class` preset that emits
  cluster ids and nothing else. Spatial placement is the one feature that does
  need coordinates, so asking for `centroid_dependence` on top of this dataset is
  refused with `[CLM-125]`; the generator logs a warning saying as much when the
  preset is used.

- **Full diagnostic coverage in the troubleshooting catalog.** Every one of the
  45 `[CLM-###]` codes now has a runnable example config, up from 42. The three
  that were missing, `[CLM-125]`, `[CLM-131]` and `[CLM-310]`, are now present,
  and the test suite fails if a future code is added without one.

  The catalog is also self-contained and portable. Configs and logs
  Every path is now relative to the catalog folder, so any case can
  be reproduced directly:

      cd docs/troubleshooting_catalog
      python -m clmsynth.main ValueError_1xx/CLM-150.yaml

### Fixed

- **Out-of-range skew settings silently produced negative class sizes.** When
  class sizes come from a `skew_rule` rather than explicit `proportions`, the
  parameters controlling that rule were never checked. Three settings did not
  fail, they returned. A `geometric` rule with `ratio: -0.5` produced the class
  sizes `[1600, -800, 400, -200]`, and `dominant_minority` with a
  `dominant_share` above 1 or below 0 produced similar. Because those still added
  up to the dataset size, nothing downstream objected and the run completed with
  a labeling nobody had asked for.

  Four more settings crashed with a raw Python error instead of an explanation:
  a `dominant_index` past the last class, `dominant_minority` with only one
  class, and `dirichlet` with an `alpha` of zero or less.

  All of them are now **`[CLM-131]`**, checked before any class sizes are
  computed. The check only applies when the skew rule is actually used, so a
  configuration that supplies explicit `proportions` is never failed for a stale
  skew block it does not read.

- **An empty YAML key crashed instead of taking the default.** Writing
  `skew_params:` or `centroid_dependence:` with nothing after it produces a null
  value in YAML, not an empty block, and the engine passed that null on as if it
  were a set of options. Both now fall back to their documented defaults, which
  is how `target_metric` already treated the same shape.

- **`target_metric` ignored your `tolerance` when using `scope: pair`.** The
  check that compares the delivered result against the requested one was fixed at
  0.01 on that path, so asking for 0.001 quietly got you 0.01, and asking for
  0.05 quietly got you 0.01 as well. `tolerance` now applies to both scopes.
  `max_iter` remains meaningful only for `scope: global`, which is the only one
  that searches.

- **Two runs starting in the same second could overwrite each other.** Run
  folders are named from the clock down to the second, and the pipeline chose a
  free name in one step and created it in another. Two runs sharing an
  `output_dir` and starting within the same second were handed the same folder
  and wrote their CSVs, plots and config copies into it, one silently replacing
  the other. The folder is now claimed at the moment it is chosen. No parallel
  execution was needed to hit this, two terminals were enough.

- **The run summary could report success for datasets that got no labels.** When
  label generation fails for one dataset, that dataset is still written out with
  its features and clusters, and still counts as processed, which is correct,
  but the summary said "10 dataset(s) processed" while some of those files were
  missing the generated label that was the point of the run. The count now
  reports the shortfall separately.

- **Plot failures caused by long Windows paths named the wrong cause.** Windows
  rejects file paths of 260 characters or more, and the plot filenames are the
  longest the pipeline writes, so with a deeply nested `output_dir` the plots
  fail while the CSV and summary succeed. Windows reports this as "No such file
  or directory" for a folder that plainly exists. The message now says how long
  the path is and what the limit is.

### Changed

- **A cluster id that is missing from one dataset no longer aborts a whole
  batch.** `[CLM-104]` and `[CLM-105]` report that a label or cluster id named in
  your configuration does not exist in the data. Unlike every other configuration
  error, that is a statement about one dataset rather than about the
  configuration, under `byoc` each CSV brings its own cluster ids, and nothing
  requires them to match. Previously the first mismatch stopped the run, throwing
  away both the datasets already written and the ones that would have succeeded.

  Now, for `byoc`, every input file's cluster column is read before any work
  begins: if ids are missing anywhere the run is refused immediately, with every
  offending file named, and nothing is written. For the other sources, where ids
  cannot be known without downloading or generating each dataset, the mismatch is
  reported for that dataset and the batch continues. All other configuration
  errors still stop the run.

- **The configuration wizard now asks for `tolerance` under both target-metric
  scopes**, since `scope: pair` reads it as of this release.

- **`validate_matching_ids` is available from the engine** for callers who want
  to check a configuration's cluster and label ids against a dataset before
  running it. The pipeline uses it for the pre-flight check described above.

## [0.6.2] — 2026-08-02 — Public release

The release accompanying the article submission. No change to the
label-generation math; the engine behaves exactly as in 0.6.0.

### Changed

- Documentation revised throughout: the manual, the configuration troubleshooting
  reference, and the README. The troubleshooting reference is now complete.

## [0.6.1] — 2026-08-02

### Changed

- Repository contents restored for ongoing development: this changelog, the
  engine-internals note, and `docs/troubleshooting_catalog/` are tracked again.
  They were deliberately withheld from 0.6.0, which remains the lean tree tagged
  for the archive on Zenodo. No change to the engine or to any documented behavior.

## [0.6.0] — 2026-08-02 - Pre-release

Six diagnostics, each closing a path that previously produced wrong output
silently. Archived on Zenodo: <https://doi.org/10.5281/zenodo.21751081>.

### Fixed

- **`proportions` longer than `num_classes` silently enlarged the label space.**
  The counts array was sized from `proportions` and `_spillover_draws` derived
  `M` from that array rather than from the config, so six proportions under
  `num_classes: 4` wrote labels `4` and `5` into the dataset. Only
  `proportional_to_marginal` failed, and then with an uncoded numpy broadcasting
  error, which is why `uniform` and `concentrated` went unnoticed. Now
  **`[CLM-121]`**, previously reserved as documentation-only.
- **`concentrated_labels` was never validated.** An id outside `0..M-1` reached
  the written dataset; a noninteger was truncated on write; a bare number was
  read by numpy as a *range*, scattering the remainder over that many labels.
  With a `target_metric` set, the solver reported a score "within tolerance"
  computed over a label that did not exist. Now **`[CLM-128]`**, checked before
  the solver runs.
- **A capitalization slip in `centroid_dependence.favors` inverted the spatial
  placement.** `favors == "core"` was tested directly and everything else
  treated as boundary, so `Core`, `CORE` or a typo placed labels on the cluster
  rim with no error. Now **`[CLM-129]`**.
- **`target_metric.scope: pair` was exact only under the default spillover
  rule.** The closed form sizes the target label so that all of it sits inside
  the target cluster; `uniform` spillover delivered `0.347` and `concentrated`
  onto that label delivered `0.000` against a request of `0.765`. Now
  **`[CLM-130]`** rejects the combinations that can move the label out of its
  cluster, and **`[CLM-310]`** verifies the delivered coefficient afterward,
  the counterpart of `[CLM-309]` for the global solver.
- **Several `assignment_matrix` rules naming the same label could overshoot that
  label's budget.** A rule's `recall_target` is a fraction of the label's whole
  budget, so repeated labels add up: two rules at `recall 0.6` on a 200-point
  label claimed 240, and `proportional_to_marginal` spillover cannot compensate
  because its pool clips at zero. Now **`[CLM-153]`**.


## [0.5.0] — 2026-07-31 — research release

The research release, accompanying the submitted article and the Code
Ocean capsule. No change to the label-generation math.

### Added

- **`CITATION.cff`**

### Changed

- **Coded `[CLM-1xx]` configuration errors now abort the run** instead of being
  swallowed per dataset. A malformed config is equally wrong for every dataset,
  so the pipeline previously logged the identical message once per dataset and
  exited having written nothing, reported as an "unexpected error" rather than a
  configuration one. It now fails immediately with the coded message at CRITICAL
  and exit code 2, distinct from exit 1, "ran but produced nothing".
  `InfeasibleAllocationError` (`15x`) is unchanged: it remains a per-dataset
  skip, because another dataset's cluster sizes may well satisfy the same rules.

## [0.4.0] — 2026-07-31

Wizard and config-renderer polish. No change to the label-generation math.

### Added

- **`generate_config.py` can now emit every documented `clm_label` option.**
  `skew_params`, `concentrated_labels`, `competing_noise`, `target_metric`
  (including `scope`, `tolerance`, `max_iter`) and `centroid_dependence.steepness`
  previously had no template slot, so the payload path could not express features
  the manual documents in full, they could only be added by hand-editing the
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
- **The wizard printed a command that fails.** On completion, it suggested
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
- Authorship recorded.
- `pyivm` is now documented consistently as **not implemented yet**.
  `metrics.evaluate_cluster_label_matching` remains a provisional hook that no
  part of the pipeline calls. Previously the README both asserted and denied
  that the dependency had been verified.
- `dataset_sources.py` module docstring now describes all four sources
  (`byoc` was missing) and notes that the registry shape is the extension point.

### Fixed

- Repository URL in `pyproject.toml` corrected.
- `[CLM-309]`'s explanation corrected in the README and the engine comment. The
  probe and output streams differ because they are seeded differently
  (`default_rng(probe_seed)` vs `default_rng(seed)`), not because the label-count
  draw had advanced the run stream, that only happens under
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
  pre-package layout and no longer matched the current engine, stale module
  paths, the old `dummy` source name, and in one case a raw `IndexError` where
  `[CLM-102]` now fires. The manual's reference to it is commented out, not
  deleted, pending a regenerated catalog.

## [0.2.0] — 2026-07-15
Internal, unpublished. Repository and packaging housekeeping; no functional or
user-facing changes.

## [0.1.0] — 2026-07-07
Initial release state: the CLM label engine, four interchangeable data sources
(`clustbench`, `mdcgen`, `fabricated_data`, `byoc`), the coded `[CLM-###]`
diagnostics registry, the config wizard and template renderer, MCC/ARI
evaluation, and scatter-plot output.
