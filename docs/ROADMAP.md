  # CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.7-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.7 to 0.9.0. Current behavior is in **`../README.md`**.
Past changes are in **`../CHANGELOG.md`**.

## Conventions

A patch release (`0.0.x`) fixes specified behavior.
A minor release (`0.x.0`) ships a new capability.
A major release uploads validation tests to the Zenodo repository.
`[CLM-###]` codes are never renumbered or reused.

---

## 0.7.8 and 0.7.9

- **A Sphinx site.** *[feature]*

- **clustbench local cache.** Every run downloads its datasets from GitHub as of 0.7.6

- **Reading the diagnostic codes from Python.** A way to collect the `[CLM-###]` codes
  of the warnings a call logged, so that a script can check them after
  `generate_clm_labels`.

- **One source of truth for settings** Every setting's default, the
  condition it applies under and its allowed values come from one table that the
  engine, the saved run config and the wizard all read. *[improvement]*


---

# 0.8.0

Will be submitted to the conda-forge review queue on 22 November 2026.

### Distribution

**conda release**

- **conda-forge** `noarch: python` build. *[feature]*

- **Signed tags, from the 0.8.0 tag onward.** *[improvement]*


### Changed

- **Stability commitment.** The `[CLM-###]` registry and `__all__` follow semantic
  versioning. Existing codes never change behavior, so runs stay reproducible.

---

# 0.9.0 series

Community feedback, issues and cross-platform compatibility. 
A CLMSynth-GUI is planned, which will use compiled C binaries for near real-time rendering.

## Unplanned

- **Automatic assignment for a target metric.** An explicit opt-in such as
  `assignment_matrix: auto` would let the engine choose the label-to-cluster
  pairing (label i to cluster i, or matched by size) and solve the target over
  it. An empty or valueless matrix stays an error: the pairing changes the
  result, so it is not guessed. *[feature]*

- **Label opt-out from spillover.** `scope: pair` resizes the target label and pins
  its recall. *[feature]*

- **NCA, SMI, and AAA as a `target_metric` type.** clustbench scores with NCA by default.
  NCA is asymmetric and has no known closed form. It needs a `_METRIC_FUNCS` entry and a
  global solve. *[feature]*

- **Fewer repeated tries in the `target_metric` search.** Each recall level gives every
  label a whole number of points, so nearby levels give the same labeling, and once
  the search has narrowed to a few of them it scores them again; on small clusters
  most of its tries are repeats. scipy's step tolerance, `xtol` for `brentq` and
  `xatol` for `minimize_scalar`, set to one point of the largest label stops it there.
  The labels stay the same when every label has the same count. When counts differ,
  two labeling can lie closer than one step, and the larger step changes which
  levels scipy tries, so some configs can get different labels. *[improvement]*

- **Python 3.15 features, once 3.15 is the minimum.** `frozendict` for the read-only
  registries, closed `TypedDict` so a type checker flags a misspelled key in a config
  written in Python, and sampling profiler (PEP 799) for the target-metric search and large runs.
  *[improvement]*

- **`clmsynth-config` in a pipeline.** Rendered configs piped straight into runs,
  e.g. calibration datasets at MCC 0 to 1 in steps of 0.1. *[feature]*

- **Integer network flow.** Since 0.7.5 every count is rounded exactly from the
  written values, and the rules' counts are rounded together with their clusters as
  one integral flow, so `[CLM-150]` and `[CLM-151]` refuse only when no rounding of
  the rules' counts fits. The label counts are still rounded first, alone:
  `proportions: [0.5025, 0.4975]` of 200 points gives 101 + 99, and `single_match`
  refuses label 0 in a 100-point cluster (`[CLM-150]`), though 100 + 100 rounds the
  same proportions and fits. A transportation formulation over the label counts, the
  rules and the clusters gives integer counts in one exact solve. `competing_noise`
  and spillover stay outside it: they take what the matching leaves. *[improvement]*
