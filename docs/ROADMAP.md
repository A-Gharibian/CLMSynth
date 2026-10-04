# CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.2-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.3 to 0.9.0. Current behavior is in **`../README.md`**.
Past changes are in **`../CHANGELOG.md`**.

## Conventions

Every version below ships code. A patch release (`0.0.x`) fixes specified behavior.
A minor release (`0.x.0`) ships a new capability.

`[CLM-###]` codes are never renumbered or reused. A new code makes a release
minor, except for a fix.

---

## 0.7.3

### Reachability of MCC ceiling

- **The ceiling while the wizard is running.** The ceiling
  `MCC=sqrt(M(M-1)/(K(K-1)))` needs `K` before any fetch. `byoc` knows `K`;
  `clustbench` need a static table. Generative data such as `mdcgen` can report
   only at the end if target fails to reach. *[feature]*

### Will be Added

- **The engine computes and reports the ceiling.** The closed-form ceiling needs only
  `M` and `K`. The reachable ceiling is the MCC at full recall. The solver already
  evaluates `alpha = 1.0` and discards it. *[feature]*
- **The solved value is returned, not only logged.** `generate_clm_labels` returns
  only the label Series. *[feature]*
- **The ceiling acted on, not just computed.** Refuse or warn before searching past
  the closed-form bound. Name the reachable value in `[CLM-306]`. The wizard shows
  the bound when asking for a target. *[feature]*
- **`scope: pair` with `type: ari`.** Pair ARI inverts in closed form, as a
  quadratic. `[CLM-123]` refuses it today. `[CLM-307]` would need the ARI floor. *[feature]*

## 0.7.4

### Parallel-safe batch execution and performance release

### Will be Fixed

- **The search grid misses a narrow feasible band.** Widen the bracket beyond the
  grid points. *[issue]*

- **Integer label counts must sum to `N`.** The `[CLM-106]` tolerance can make the
  rounding remainder negative. Large datasets then get too many labels. *[issue]*

- **Duplicate `competing_noise` entries.** Duplicates sum their counts but keep the
  last `favors`. *[issue]*

- **matplotlib/seaborn plotting is not thread-safe.** `plot_feature_scatter` uses
  process-global pyplot and seaborn state. *Fix:* `matplotlib.use("Agg")` at import
  in `visualization.py`. *[preventive]*

- **Logging is a shared sink with several workers.** Worker records interleave and
  cannot be attributed. Messages do not always name their dataset. Per-run log
  files or a run id: still open. *[preventive]*

### Will be Added

- **A documented batch entry point.** Extend `03_isolation` to cover it. *[feature]*
- **A Sphinx site.** *[feature]*

### Performance

- **The catalog generator runs its subprocesses serially.** Each case has its own
  folder, so pooling is safe. *[improvement]*

### Scope note

Python 3.15 lazy imports may make some fixes unnecessary.

## 0.7.5 Release version

Will be released to PyPI.

---

## 0.7.6 and 0.7.7

**Source and generator extensions.** Data-source work, not engine work. Clusters stay
fixed, read-only input to the engine.

### Will be Added

- **Repliclust.** It generates clusters from prompts. *[feature]*

### Will be Fixed

- **The `clustbench` registries disagree.** `SOURCE_METADATA` and
  `CLUSTBENCH_DATASETS` list different batteries. `batteries: all` uses the larger
  one. *[improvement]*

## 0.7.8 and 0.7.9

**Python 3.15 support**

- `requires-python` becomes `>=3.12,<3.16`. Add the 3.15 classifier and CI
  interpreter. Ships once Python 3.15 is released and tested. *[feature]*

- Relax exact pins to `>=` in `requirements.txt` and
  `environment.yml`. conda-forge discourages exact version pins. *[improvement]*

---

# 0.8.0
## 22 November 2026

### Distribution

**conda release**

- **conda-forge** `noarch: python` build. *[feature]*

- **Signed tags, from the 0.8.0 tag onward.** *[improvement]*

### Changed

- **Stability commitment.** The `[CLM-###]` registry and `__all__` follow semantic
  versioning. Existing codes never change behavior, so runs stay reproducible. *[improvement]*

---

# 0.9.0 series

Community feedback, issues and cross-platform compatibility. Pipeline integration
and a CLMSynth GUI are planned.

## Unplanned

- **Integer network flow.** Rounding happens at three independent sites. Their
  errors do not cancel. A transportation formulation gives integer counts in one
  exact solve. *[improvement]*

- **Automatic assignment for a target metric.** An explicit opt-in such as
  `assignment_matrix: auto` would let the engine choose the label-to-cluster
  pairing (label i to cluster i, or matched by size) and solve the target over
  it. An empty or valueless matrix stays an error: the pairing changes the
  result, so it is not guessed. *[feature]*

- **Label opt-out from spillover.** `scope: pair` resizes the target label and pins
  its recall. *[feature]*

- **NCA as a `target_metric` type.** clustbench scores with NCA by default. NCA is
  asymmetric and has no known closed form. It needs a `_METRIC_FUNCS` entry and a
  global solve. *[feature]*
