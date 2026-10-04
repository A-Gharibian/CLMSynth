# Changelog

All notable changes to CLMSynth are documented here.

---

## [0.7.2] — 2026-10-04

This patch release is the first release after the availability of the SoftwareX article,
including the SoftwareX fork. To see what issues may arise if the program were
checked by large language models, the whole source code was uploaded to three models:
`GPT-6 Luna` (OpenAI), `Gemini 3.1 Pro` (Google), and `Claude Opus 5.5` (Anthropic),
with prompts crafted to identify issues.

We categorized, triaged, and resolved the identified issues when they were relevant
(prompts and correctness data are available upon request). None of the issues changed the 
output of the core label allocation engine; however, three new diagnostic codes
were added: `[CLM-132]`, `[CLM-133]` and `[CLM-134]`. The fixes were applied by the 
same model that identified them and were checked individually.

### Added

- **`--no-viz` skips rendering and matplotlib import.** `clmsynth cfg.yaml --no-viz`
  writes CSVs and summaries only; `png/` stays empty. `clmsynth.main` no longer
  imports the plotting stack at module scope (h/t Reviewer #2).

### Changed

- **The command line is parsed by `argparse`.** `--help` works. An unrecognized
  argument is now a usage error with exit code 2, where it used to be ignored.

- **A config or payload that is not valid YAML exits 1.** `clmsynth` and
  `clmsynth-config` report the line and column of the mistake, where they used to
  stop with a traceback.

- **An `output_dir` blocked by a file exits 1.** When `output_dir`, or a folder
  above it, is an existing file, `clmsynth` names that file and exits 1 before
  creating anything. On Windows the run used to hang, retrying run-folder names
  forever; other systems stopped with a traceback.

- **Skipped datasets are counted at the end of a run.** The run ends with "N of
  the M resolved dataset(s) were skipped" when any were. An unexpected error names
  its type (`unexpected KeyError: 'cluster'`, where it used to read
  `unexpected error: 'cluster'`).

- **`run_pipeline` reads its config without changing it.** The suite block is read
  with `.get()` and typed as a read-only `Mapping`, so mypy rejects a `.pop()`; it
  used to be copied and emptied. A suite given as a list now stops with a
  traceback instead of a silent exit 2.

- **Three ruff rules replace three tests.** TID251 (no core import of the wizard,
  no stdlib `random`), TID253 (no module-level plotting imports) and NPY002 (no
  global `np.random`). Eleven tests that duplicated another check or could not
  fail left `tests/` (397 tests now). The test files are formatted with
  `ruff format`.

### Fixed

- **An unknown `split_rule` is reported under `custom` whatever the rules' shape.**
  It used to be read only for a rule spanning two or more clusters, so a typo with
  one cluster per rule passed silently. `[CLM-108]` now fires before any
  allocation. Other modes still do not read it. Labels are unchanged.
- **`num_classes` must be an integer.** A quoted `"4"` or a float crashed with an
  uncoded `TypeError`, and `yes`/`true` passed as one class. Both are `[CLM-126]`
  now. Labels are unchanged.

- **BYOC refuses cluster ids that differ only by whitespace.** `A` beside `A ` used
  to become two clusters without a word. An id padded the same way on every row
  is still accepted as written. Labels are unchanged.

- **ruff now lints the package.** The `.gitignore` entry `clmsynth/` also matched
  `src/clmsynth/`, and ruff skips ignored paths, so CI linted only `tests/` and
  `tools/`. The entry is now `/clmsynth/`.

- **An unknown `balance` is refused with `[CLM-132]`.** `balance` was compared only
  against `'balanced'`, so a typo (`Balanced`, `balnced`), `''` or a valueless
  `balance:` was read as `'unbalanced'`. With `proportions` set, `Balanced`
  delivered the proportions it was meant to ignore; without them, `[CLM-203]`
  blamed `skew_rule`. An absent key still means `'balanced'`. `clmsynth-config`
  warns at render time, and no longer warns about `balance`, `proportions` or
  `skew_rule` under `perfect`. Labels are unchanged for every valid config.

- **`perfect` no longer reads or validates the label-sizing keys.** `balance`,
  `proportions`, `skew_rule` and `skew_params` were resolved and checked, then
  discarded, so `[CLM-121]`, `[CLM-106]`, `[CLM-107]`, `[CLM-131]` and `[CLM-203]`
  could stop a `perfect` run that `[CLM-302]` said ignored them. `[CLM-302]` now
  also names `skew_params`. Labels are unchanged.

- **The matching mode is checked before label sizing.** A missing `matching_mode`
  (`[CLM-202]`), an unknown one (`[CLM-101]`) and `perfect` with `M != K`
  (`[CLM-102]`) used to lose to a sizing error such as `[CLM-203]` or `[CLM-121]`.
  A misspelled mode with a `target_metric` now reads `[CLM-101]`, not `[CLM-111]`.
  Labels are unchanged.

- **`[CLM-304]` fires only when the delivered counts differ from their targets.**
  Its `competing_noise` part was predicted before placement, so it fired whenever
  noise was placed. Under `proportional_to_marginal` the noise usually only
  displaces spillover and the counts stay exact; they move only when the noise
  gives a label more points than its target. The warning lists both count vectors
  and is logged once, not once per solver probe. Labels are unchanged.

- **`[CLM-150]` names where the label's size came from.** Under `single`, which has
  no `recall_target` key, it advised a "max feasible recall_target". It now gives
  the label's largest feasible size and the setting behind it (`proportions[3] =
  0.4`, `balance 'balanced'` or the `skew_rule`); under `custom` it also gives the
  feasible `recall_target`, rounded down so the advice itself fits, and a
  `recall_target` above 1 is told it must be at most 1.0. Labels are unchanged.

- **A valueless `assignment_matrix:` with a `target_metric` is `[CLM-206]`.** It
  crashed with `TypeError`, also under `single`, which never reads the matrix;
  without a target it already gave `[CLM-206]`. Under `single` the stray key is now
  ignored. Labels are unchanged.

- **Unusable cluster ids or `coords` are refused with `[CLM-133]`.** Through the
  Python API, missing ids (NaN/None) were merged into one cluster that counted no
  points, so they were never labeled: an uncoded crash, or labels under `random`.
  Ids mixing strings and numbers crashed the sort, and extra `coords` rows were
  silently dropped. 
  - Now it is judged per dataset, like `[CLM-127]`. `byoc` already
    refused blank cluster cells and skipped the dataset; that refusal now carries
    `[CLM-133]` too.

- **A string `batteries` or `datasets` stops the run and names the key.** A value
  other than `"all"` was read letter by letter: `datasets: "mydata"` became `m`,
  `y`, `d`, `a`, `t`, `a`, and `batteries: "fabricated"` matched nothing. It now
  stops like an unset `batteries`.

- **A valueless `seed:` draws a seed and reports it.** A bare
  `label_generation.seed:` crashed on `None + 0`, and a bare suite `seed:` reached
  the generator as `None`, so the data could not be reproduced. One integer is now
  drawn, used and logged ("Set 'label_generation.seed: N' to reproduce this run").
  An absent key is still 42. Labels are unchanged for any seed that is set.

- **Each summary `.txt` records its seeds.** A new Seeds section gives the
  `label_generation` seed and, for `mdcgen` and `fabricated_data`, the suite seed,
  with how each was decided (set, the default 42, or drawn). Every generated label
  lists its own seed (label i uses seed + i). `clustbench` and `byoc` never read
  the suite seed, so it is neither drawn nor listed for them. A drawn seed is no
  longer only in the log. Labels are unchanged.

- **A value of the wrong type is refused with `[CLM-134]`.** Each of these ran as
  something else, with exit 0:

  - `proportions: {0: 0.8, 1: 0.2}` was read by its keys (0 + 1 = 1), passed
    `[CLM-106]`, and gave every point label 1.
  
  - `centroid_dependence.enabled: "false"` (quoted) or `"no"` switched placement
    on.
  
  - `true` matched cluster 1 in `single_match.cluster`, `clusters` and
    `competing_noise.cluster`.
  
  - `true` as a `recall_target` meant 1.0, and as a `max_iter` it meant one
    iteration

  - A NaN, infinite or `true` `tolerance` meant `[CLM-309]`/`[CLM-310]` could
    never fire.
  
  - A negative `max_iter` skipped the search refinement.
  
  - `clusters` that is not a list (a string was read letter by
    letter), and NaN in `proportions`, `recall_target` or an `exponential`
    `steepness` are also refused. Each check runs only where its key is read.

- **Booleans and NaN/inf are refused by the existing codes too.** `share: true`
  is `[CLM-117]`, `target_metric.value: true` or a valueless `value:` is
  `[CLM-113]`, a NaN or infinite `ratio`/`alpha` is `[CLM-131]` (they used to
  crash later), and an infinite `concentrated_labels` entry is `[CLM-128]`. The
  wizard asks again when given `nan` or `inf`. Labels are unchanged.

- **A misspelled `profile` is refused wherever placement can read it.**
  `[CLM-110]` used to fire only when a label was actually placed by profile, so
  `profile: bogus` passed with no rule-claimed label or a `competing_noise` entry
  that placed nothing. Labels are unchanged.

- **A `competing_noise` entry without `cluster` or `label` is `[CLM-207]`.** It
  was a bare `KeyError` that logged only `'cluster'`. It is still a per-dataset
  skip. Labels are unchanged.

## [0.7.1] — 2026-09-17

### Changed

- **`run_pipeline` split into helpers; behavior unchanged.**

### Fixed

- **`[CLM-205]` and `[CLM-207]` catch missing `label`.**
- **The wizard rejects negative `proportions`.**
- **The engine rejects negative `proportions`, `recall_target`, `tolerance`, `steepness`.**
  A plain `ValueError`, not a `[CLM-###]` code.
- **A noninteger `label` is refused by type.** `1.0` and `True` were accepted,
  then failed later as an uncoded `IndexError`.
- **`[CLM-104]` and `[CLM-118]` name the type, not the range.** A label that is a
  column name no longer reads as out of range.

## [0.7.0] — 2026-09-01

First published release on PyPI.

This is the release pinned to the published SoftwareX paper, with one notable
behavioral change of the engine from release 0.6.2 which was the submitted version.
The change affected how the target was reached and reported for MCC, as a result,
the Zenodo repository of the article data reproduction and validation tests was also
updated with clear traceable changes.

## [0.7.0rc2] — 2026-09-01

Documentation corrections.

### Fixed

- **`[CLM-309]`'s documented cause.** The troubleshooting reference from 0.6.4.

- **Two `README.md` Known-limitations bullets** are added in response to reviews.
  The `scope: pair` description overstated the delivered value: The solve is
  exact; the delivered pair MCC is measured afterward and reported as achieved.

- Additions **In response to reviewer comments.**

## [0.7.0rc1] — 2026-08-30

Testing release on `test.pypi.org`.

## [0.7.0b1] — 2026-08-30

### Added

- **BYOC `tag_columns`.** Columns listed are carried to the output CSV,
  so an outcome variable, a second labeling or an identifier can travel with the data.
- **`MissingConfigKey` is exported at the package root.**
- Tests for the shown-figure and catalog file-target fixes.

### Changed

- **`clustering_mcc` aligns to maximise R_K, not accuracy.** The Hungarian step
  now runs on `n*C - outer(t, p)`, which is the same single assignment call and
  returns `max` over matchings rather than R_K read off the accuracy-optimal one.
  Balanced runs with `M <= K` are bit-identical; imbalanced and surplus-label
  configurations move, and the value can no longer come out negative.
- **The recall solver uses SciPy.** It is now
  `brentq`, with a bounded minimization when the grid shows the target is not settled.
- **`_loadtxt_url` refuses any scheme but http(s).** `base_url` comes from a
  config, and a config is a shareable artifact.
- **Plotting and `run_pipeline` now separate.** Behavior is
  unchanged;  `_render_dataset_plots` can be skipped now.

### Removed

- **Python 3.11.** `requires-python` is now `>=3.12`. numpy and scipy both moved
  to `>=3.12` under SPEC 0, so `requirements.txt` could not be installed on 3.11.

### Fixed

- **`[CLM-302]` fired even when nothing was ignored.** `matching_mode: perfect`
  warned that `proportions`, `balance` and `skew_rule` were being ignored on every
  run, including configs that set none of them. It now warns only when at least
  one is present. Labels are unchanged; the code and its message are unchanged.

- Config generator, one root cause across five sites:
  - A valueless `assignment_matrix:` renders `[]`, so `[CLM-206]` fires.
  - A valueless `centroid_enabled:` keeps the documented default of true.
  - Scalar keys render their default instead of the literal `None`.
  - A bare `batteries: "g2mg"` is one name, not four characters.
  - `target_metric` renders every key, so a typo stays visible.
- `[CLM-104]` is checked before any dataset-dependent guard.
- A malformed config value aborts cleanly instead of escaping as a traceback.
- A valueless `datasets:` no longer crashes the run.
- Wizard corrections:
  - Perfect mode respects the label-count cap it bypassed.
  - `concentrated_labels` is capped like every sibling label question.
  - The name-clash guard renames the file, not its parent folder.
  - A bound message no longer prints `None` for an absent bound.
  - The registry seed prompt goes through the question schema.
- Non-numeric BYOC columns have one policy, not two disagreeing ones.
- An interactively shown figure is no longer closed immediately.
- The catalog generator refuses a file target with a clear message.


## [0.6.9] — 2026-08-23

**Config and Wizard corrections.**

### Fixed

- Wizard and config generator improvements:
  - The wizard's `num_classes` upper bound matches the engine.
  - Catalog generator refuses to delete a non-catalog target.
  - Valueless config keys render their documented default.
- A figure-creation failure returns `False` like every other.
- `[CLM-104]` is reported once, not per dataset.


## [0.6.8] — 2026-08-20

A patch release.

### Added

- **`[CLM-201]`–`[CLM-209]`: the missing-key band is coded `KeyError`s.`MissingConfigKey`
  subclasses, next to the `1xx` `ValueError` the bands separate an
  *absent* key from a *wrong* key.

- **`target_metric.probe_seed` is renderable.** 
  (when `[CLM-309]` sets it, the config renderer can now emit it).

### Fixed

- **Which failures are per dataset, corrected in both directions.** The rule is
  whether a code judges the *configuration* or *this dataset*. [CLM-102] (M == K), [CLM-119]
  (a competing_noise cluster this dataset does not have), [CLM-125] (no features) and [CLM-127]
  (K over the cap) are statements about the dataset
  and now skip it and continue, where they used to
  abort a batch that later datasets could have satisfied. `[CLM-104]` moves the
  other way: its bound is `num_classes`, which no dataset can change, so it is a
  configuration error and now aborts like the rest of the band. `[CLM-105]` is
  unchanged, and the BYOC precheck is unchanged.

- **A bare `concentrated_labels:` key no longer crashes uncoded.**
  `dict.get(key, default)` returns the default only when the key is *absent*,
  and YAML parses a valueless key to `None`, so `spillover_rule: concentrated`
  with an empty `concentrated_labels:` reached `rng.choice(None, ...)`.

- **`clmsynth-config` exits non-zero when it writes nothing.** A failed write was
  logged and then reported success to the shell. It also now writes UTF-8.

- **`clmsynth`, `clmsynth-config`, `clmsynth-wizard` cancels with `KeyboardInterrupt`**.

- **The wizard's last two free-text prompts re-ask.** The BYOC filename list and
  the dataset picker were the only prompts bypassing `questions.SCHEMA`.

- `plot_feature_scatter` saves the figure it built rather than whatever pyplot
  considers current.

### Removed

- **`evaluate_cluster_label_matching`.** `pyivm` is scheduled for release 1.0.0,
  it is pinned to `numpy<2.0`, which may prevent implementation, removed until decided.

### Staging for CLMSynth-GUI

CLMSynth-GUI development started here; the goal is to be able to configure the clusters,
cluster-label matching and allocation in an accessible graphical user interface.
CLMSynth-GUI will be a separate package and will not include the CLMSynth core.

## [0.6.7] — 2026-08-12

**The wizard, in isolation.**

Scope is one file plus a sibling schema module: `config_wizard.py` and the new
`questions.py`.
The wizard's guided path already mitigates several uncoded paths at the config
layer, as of 0.6.1: `num_classes` is floored at 2 (closing the `M=1`
`ZeroDivisionError` and the `M=1` single-mode rejection), and `scope: pair` pins
`proportional_to_marginal` spillover (closing a `[CLM-130]` path). Those are
guards in `config_wizard.py` only. Handwritten and library configs bypass them
entirely, which is why the engine still needs its own diagnostics regardless of
what the wizard does.
The reusable half is not the wizard but the **question schema**.

### Added

- **A declarative question schema** (`questions.py`), one `Question` per prompt
  carrying its wording, help text, default, range, choices and a `visible_when`
  predicate. Making the flow data rather than control flow makes it reorderable,
  testable without driving stdin, and reusable by the upcoming `help` command.

  **Question ranges belong in the schema.** The engine's `[CLM-131]` guard exists
  because `skew_params` values did not crash but *returned*: a negative `ratio`,
  and `dominant_share`/`dominant_index` out of range, produced label counts that
  still summed to `N`, so largest-remainder rounding was satisfied and nothing
  downstream objected. A schema entry carrying a range makes those un-enterable at
  the point of asking, which is arithmetic on values already in hand. It is the
  same move as the existing `num_classes` floor of 2, generalized.

- **A wizard-only floor on a target metric.** MCC is defined on
  `[-1, 1]`, and a negative *target* is possible, but the promise of the program
  is not to solve for one. So the wizard floors the target at
  `0.0` while the engine **can solve**.

- **A path-length warning where `output_dir` is asked for.** Plot writes fail on
  Windows past the 260-character `MAX_PATH` limit; the engine names that cause in
  its failure message as of 0.6.4, but only after a run has already produced
  partial output. The `MAX_PATH`
  literal is duplicated rather than imported from `visualization.py`, which pulls
  matplotlib and seaborn.

### Fixed

-  dirichlet `alpha` is floored *strictly* (`> 0`, matching the engine's
  divide-by-zero guard, where the wizard previously accepted `0`), and
  `dominant_index` gains its `0..M-1` upper cap at the point of asking.

---
## [0.6.6] — 2026-08-11

A patch release.

### Added

- **A run states where it writes, before it writes.** `global_settings.output_dir`
  and `byoc_suite.input_dir` are now resolved to absolute paths and logged once
  per run, before the run folder is created or a CSV is read.

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
  prompts once per run rather than once per dataset.

- **A configuration-safety test category**, `tests/test_05_config_safety.py`.
  The stated condition for its return was that the program itself implement a
  measure protecting the machine that runs a configuration.

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

- **A tracked pytest suite, `tests/`.** 257 tests across seven modules‽ ``Refer to more recent changes.``

- **Continuous integration** The workflow was a
  placeholder that installed the package and imported it. It now runs three
  jobs on every push request:

  - **Tests** across Python 3.11, 3.12, 3.13 and 3.14, the full range
    `requires-python` declares, without `fail-fast` so one interpreter failing
    alone is distinguishable from all of them failing.
  - **Lint, typing and security**: `ruff`, `mypy` and `bandit`, all three
    gating.
  - **Packaging**: builds the sdist and wheel.

- **`ROADMAP.md` is published**,
  Planned work from 0.6.6 through 1.0.0‽ ``Refer to more recent changes.``

- **`SECURITY.md`**, CLMSynth is a local single-user CLI and library with no privilege
  boundary between the person supplying input and the person running it. Configuration
  file is shareable, and its values can come from someone else, which is the only internal
  source of vulnerability. Security of online clustering data are not covered by this project.
  The file also records each accepted scanner finding, including bandit's `B310`.

### Fixed

- **A run that produced nothing left a run folder**
  Both failure exits now discard the folder, and only when it holds exactly the
  scaffolding and nothing else. A single written file, expected or not, means the
  folder stays untouched. This also makes `precheck_byoc_matching_ids` truthful:
  it documented itself as aborting "before any output is written".

- **The sdist was missing the byoc catalog input.** `MANIFEST.in` included
  `docs/**` for `.tex`, `.yaml` and `.log` but not `.csv`;
   now asserted by CI rather than by reading the manifest.


### Changed

- **The `[tool.bandit]` judgements are recorded in `pyproject.toml`**, with the
  reason for each, in the same form as the `ruff` exemptions added in 0.6.4.
  Four findings (`B101`, `B404`, `B603`, `B310`) were traced by hand and judged
  unexploitable under this package's threat model, a local single-user CLI and
  library with no auth boundary. `B310` in particular is revisited the moment
  URL-scheme restriction is implemented.

- **The test suite is laid out flat.** Each category was a directory containing
  exactly one module of the same name; the directory added a level that held
  nothing. `tests/00_contract/test_00_contract.py` is transferred to
  `tests/test_00_contract.py`, and `smoke_test.py` follows the `test_*` prefix.


## [0.6.4] — 2026-08-09

A solved target metric is now the value actually delivered,
rather than a value the search reached on a stream the output
did not use.

### Added

- **Lint, typing and test configuration now live in `pyproject.toml`.** The
  project previously had no `[tool.ruff]`, `[tool.mypy]` or
  `[tool.pytest.ini_options]` section.

- **BYOC imports are checked against stated requirements.** BYOC is an import
  path, not a generator: the premise is that the user clustered a subset of 
  features and are importing the result. A file is now rejected, with every
  problem reported in one pass rather than one per attempt, when it is empty, has
  duplicate column names, uses a name the pipeline reserves (`Cohort_Class`,
  `GroundTruth_*`, `Cluster_n`, `Label_n`), has missing values in the cluster
  column or in a feature, holds fewer than two clusters, contains a cluster of
  fewer than three points, or has a non-numeric feature column.   
  The full list, with the reason for each, is in the troubleshooting reference
  under *Uncoded Rejections (data import)*. These deliberately carry no
  `[CLM-###]` code: the coded diagnostics describe the cluster-label matching
  engine, while these describe whether the data is a usable clustering at all.

  - The rule regarding number of points in a cluster will change based on feedback.

- **A `labels_only_4class` preset for the `fabricated_data` source**, cluster
  ids with no feature columns. The engine has always accepted such a dataset,
  because recall targets, class balance, allocation and spillover never look at
  coordinates, but until now no configuration could produce one, so the
  capability was reachable only from Python.

### Fixed

- **A solved `target_metric` is now the value actually delivered.** Under
  `scope: global` the solver scores candidate recall levels on one fixed random
  stream so that candidates compare fairly. The labeling that was written out,
  however, was generated on the run's own stream, and the two could drift apart:
  allocation continued a stream that a dirichlet skew may already have consumed
  draws from, which no probe could reproduce. A search could therefore report
  success and deliver something outside tolerance by up to 0.07 on small
  datasets. This has changed one validation test from version 0.6.2 which is documented
  in the Zenodo repository.

  Allocation now draws from its own stream, started from the run seed, and the
  search uses that same stream. `[CLM-306]` reports the ordinary case of a target
  the data cannot reach; 
  - at small dataset sizes the achievable values form a coarse ladder and a tight tolerance
    can fall between two rungs and is a known limitation (refer to README).

    `[CLM-309]`'s message has been rewritten accordingly, it previously explained
    a cause that no longer applies.

- **Battery and dataset names that look like paths are refused.** These names are
  used to build file paths in both directions: `byoc` resolves
  `input_dir/<dataset>.csv` to read, and every source writes
  `csv/<source>__<battery>__<dataset>.csv`. A name containing a separator or
  `..` therefore reached outside both configured folders. The registry sources
  filter their names against a known list and were never exposed; `byoc` trusts
  the configuration's list verbatim, which is the route this closes.

### Changed

- `config_wizard` reports a non-zero exit code from the pipeline run it launches
  instead of discarding it.
-  A section for rejections that are not 
  `[CLM-###]` diagnostics is added to the troubleshooting reference. 
- Every `[CLM-###]` code now has a runnable catalog fixture, and the generator
  refuses to run if a registry code has no builder.

## [0.6.3] — 2026-08-08

A correctness release. Every entry below closes a case where the program either
gave an answer that was wrong without warning, stopped with an error that did
not explain, or reported a number it had not actually delivered. Plus one new
feature: a dataset can now consist of cluster ids with no features.

### Added

- **A labels-only data source.** The CLM engine can accept a dataset
  with cluster ids and no feature space, because recall targets, class balance,
  allocation and spillover are all counting problems that never look at
  features. No configuration could produce such a dataset, every
  source emitted at least one feature column, so the capability was reachable
  only by calling the library from Python.

  The `fabricated_data` source has a `labels_only_4class` preset that emits
  cluster ids and nothing else. Spatial placement is the one feature that does
  need coordinates, so asking for `centroid_dependence` on top of this dataset is
  refused with `[CLM-125]`; the generator logs a warning saying as much when the
  preset is used.

- **Full diagnostic coverage in the troubleshooting catalog.** Every one of the
  45 `[CLM-###]` codes now has a runnable example config, up from 42. The three
  that were missing, `[CLM-125]`, `[CLM-131]` and `[CLM-310]`, are now present,
  and the test suite fails if a future code is added without a catalogue config.

### Fixed

- **Out-of-range skew settings silently produced negative class sizes.** When
  class sizes come from a `skew_rule` rather than explicit `proportions`, the
  parameters controlling that rule were never checked. Three settings did not
  fail, they returned. A `geometric` rule with `ratio: -0.5` produced the class
  sizes `[1600, -800, 400, -200]`, and `dominant_minority` with a
  `dominant_share` above 1 or below 0 produced similar. Because those still added
  up to the dataset size, nothing downstream objected and the run completed with
  a labeling that was not expected.

  Four more settings crashed with a raw Python error instead of an explanation:
  a `dominant_index` past the last class, `dominant_minority` with only one
  class, and `dirichlet` with an `alpha` of zero or less.

  All of them are now **`[CLM-131]`**, checked before any class sizes are
  computed. The check only applies when the skew rule is used, so a
  configuration that supplies explicit `proportions` ignores the skew.

- **An empty YAML key crashed instead of taking the default.** Writing
  `skew_params:` or `centroid_dependence:` with nothing after it produces a null
  value in YAML, not an empty block, and the engine passed that null on as if it
  were a set of options. Both now fall back to their documented defaults.

- **`target_metric` ignored `tolerance` when using `scope: pair`.** The
  check that compares the delivered result against the requested one was fixed at
  0.01 on that path, so asking for 0.001 quietly got you 0.01, and asking for
  0.05 quietly got you 0.01 as well. `tolerance` now applies to both scopes.
  `max_iter` remains meaningful only for `scope: global`.

- **Concurrency failure.** Fixed by allocating folder before run starts.

- **The run summary could report success for datasets that got no labels.**

- **Plot failures caused by long Windows paths.** Windows
  rejects file paths of 260 characters or more, so the plots can fail to write
  while the CSV and summary landed.

### Changed

- **A cluster id that is missing from one dataset no longer aborts a whole
  batch.** `[CLM-104]` and `[CLM-105]` report that a label or cluster id named in
  a configuration does not exist in the data. Unlike every other configuration
  error, that is a statement about one dataset rather than about the
  configuration, under `byoc` each CSV brings its own cluster ids, and nothing
  requires them to match. Now, for `byoc`, every input file's cluster column is read:
  if ids are missing anywhere the run is refused immediately, with every
  offending file named, and nothing is written. For the other sources, where ids
  cannot be known without downloading or generating each dataset, the mismatch is
  reported for that dataset and the batch continues.

- **The configuration wizard now asks for `tolerance` under both target-metric
  scopes**, since `scope: pair` reads it as of this release.

- **`validate_matching_ids` is available from the engine** for callers who want
  to check a configuration's cluster and label ids against a dataset before
  running it. The pipeline uses it for the pre-flight check described above.

## [0.6.2] — 2026-08-02 — Public release

The release accompanying the article submission. the engine behaves exactly as in 0.6.0.

## [0.6.1] — 2026-08-02

### Changed

- Repository contents restored for ongoing development: 
  They were deliberately withheld from 0.6.0 for the archive on Zenodo.


## [0.6.0] — 2026-08-02 - Pre-release

Six diagnostics, archived on Zenodo: <https://doi.org/10.5281/zenodo.21751081>.

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
Ocean capsule. No change to the label-generation core.

### Added

- **`CITATION.cff`**

### Changed

- **Coded `[CLM-1xx]` configuration errors abort the run** 
  `InfeasibleAllocationError` (`15x`) is unchanged: it remains a per-dataset
  skip, because another dataset's cluster sizes may well satisfy the same rules.

## [0.4.0] — 2026-07-31

Wizard and config-renderer. No change to the label-generation core.

### Added

- **`generate_config.py` can now emit every documented `clm_label` option.**
  `skew_params`, `concentrated_labels`, `competing_noise`, `target_metric`
  (including `scope`, `tolerance`, `max_iter`) and `centroid_dependence.steepness`
  previously had no template slot, so the payload path could not express features
  the manual documents in full, they could only be added by hand-editing the
  rendered YAML. Each block renders only when the payload carries it, so an
  existing payload produces the same config as before.
- **Render-time warnings for configs the engine will reject**. Matching the
  existing `skew_rule`/`data_source` checks: `target_metric` outside
  `single`/`custom` (`[CLM-111]`/`[CLM-114]`), `scope: pair` without `type: mcc`
  *and* `matching_mode: single` (`[CLM-123]`/`[CLM-124]`), and `competing_noise`
  under `random` (`[CLM-115]`).
- **Wizard now asks for `spillover_rule` in `single` mode.** It was a
  `custom`-only question, so a `single` run silently took the default even though
  spillover governs every point the rule does not claim.
- **Wizard now asks for `concentrated_labels`** when `spillover_rule:
  concentrated` is chosen.

### Fixed

- **`generate_config` raised `NameError` on import under Python 3.11–3.13.**
  The new optional-block renderers annotated `List[str]` without importing
  `List` from `typing`. Python 3.14 defers annotation evaluation (PEP 649).
- **The `mdcgen` source.** 
  `fetch_mdcgen_data` did
  `import mdcgenpy` and then reached for `mdcgenpy.clusters.ClusterGenerator`,
   every run failed with
  `module 'mdcgenpy' has no attribute 'clusters'`. Fixed import to
  `from mdcgenpy.clusters import ClusterGenerator` directly, and the import
  guard reports the underlying `ImportError`.
- **The wizard printed a command that fails.** 
  On completion, it suggested
  `python main.py <config>`, which raises `ImportError` under the package layout;
  it now prints `python -m clmsynth.main <config>`.
- **The wizard crashed on empty input at the cluster-id prompts.**
  Parsing `single_match.cluster` or a `competing_noise` cluster raised an uncaught
  `IndexError`; an empty cluster list in a `custom` rule was accepted silently and
  surfaced much later as a confusing `[CLM-150]` (zero capacity).

## [0.3.0] — 2026-07-31

### Added

- `clustering_mcc_pair` is now exported from the package's public API
  (`clmsynth.__all__`), alongside `clustering_mcc` and `clustering_ari`. It is
  the 2x2 Matthews phi that `target_metric.scope: pair` inverts in closed form,
  so callers can now verify that result directly rather than reaching into
  `clmsynth.metrics`.
- `maintainers` field in `pyproject.toml`, distinguishing shared authorship of
  the method from sole ownership of the program.
- Added *CHANGELOG.md

### Fixed

- Repository URL in `pyproject.toml` corrected.
- `[CLM-309]`'s explanation corrected in the README and the engine comment. The
  probe and output streams differ because they are seeded differently
  (`default_rng(probe_seed)` vs `default_rng(seed)`), not because the label-count
  draw had advanced the run stream, that only happens under `skew_rule: dirichlet`,
  and is now described as the secondary effect it is.

### Changed

- **`fabricated_data` cluster ids are now integers `0..K-1`**, matching
  `clustbench` and `mdcgen`. They were strings (`"Class_0"`, …), which meant a
  `clm_label` config's `clusters:` / `single_match.cluster` values had to be
  retyped when moving between sources. The conversion happens in the source
  adapter (`fetch_fabricated_data`); `fabricated_generator` still emits readable
  string labels, so its own standalone CSV output is unchanged.
  `test_data_config_offline.yaml` updated accordingly.
- Authorship recorded.
  `metrics.evaluate_cluster_label_matching` remains a provisional hook that no
  part of the pipeline calls. Previously the README both asserted and denied
  that the dependency had been verified.
- `dataset_sources.py` module docstring now describes all four sources
  (`byoc` was missing) and notes that the registry shape is the extension point.


### Removed

- The erroneous-configuration catalog. Its logs had been captured from a
  pre-package layout and no longer matched the current engine, stale module
  paths, the old `dummy` source name, and in one case a raw `IndexError` where
  `[CLM-102]` now fires.

## [0.2.0] — 2026-07-15
Internal, unpublished. Repository and packaging housekeeping; no functional or
user-facing changes.

## [0.1.0] — 2026-07-07
Initial release state: the CLM label engine, four interchangeable data sources
(`clustbench`, `mdcgen`, `fabricated_data`, `byoc`), the coded `[CLM-###]`
diagnostics registry, the config wizard and template renderer, MCC/ARI
evaluation, and scatter-plot output.

---

*The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).*