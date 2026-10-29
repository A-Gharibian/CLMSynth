  # CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.5rc1-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.6 to 0.9.0. Current behavior is in **`../README.md`**.
Past changes are in **`../CHANGELOG.md`**.

## Conventions

A patch release (`0.0.x`) fixes specified behavior.
A minor release (`0.x.0`) ships a new capability.
A major release always uploads validation tests to the Zenodo repository.
`[CLM-###]` codes are never renumbered or reused.

---

## 0.7.6 and 0.7.7

**Source and generator extensions.** Data-source additions, plus an engine hook that reads
features and noise-dimension. Clusters stay fixed, read-only input to the engine.

### Will be Added

- **Repliclust.** It generates clusters from prompts. *[feature]*

- **`mdcgen` presets pass the parameters mdcgenpy reads.** mdcgenpy's names are
  `n_samples`, `n_feats`, `compactness_factor` and `outliers`; the presets pass `m`,
  `n`, `cp` and `out`, which it ignores, so a preset does not get the
  size, dimension or outliers (labelled -1) count it names.  *[feature]*

- **Noise dimensions from `mdcgen`.** New presets set mdcgenpy's `n_noise`. The
  listed columns become uniform noise for every point, or for chosen clusters only.
  mdcgenpy validates `add_noise` but never applies it, so it is not used. The output
  records, per feature, whether it carries cluster structure. `clustbench` and `byoc`
  report it as unknown. *[feature]*

- **clustbench has no local cache.** Every run downloads its datasets from GitHub,
   Planned as a cache folder beside the run folders, `<output_dir>/_cache/` or a `cache_dir` key defaulting to it.
  Concurrent tasks fill it at once. Entries are keyed by the
  pinned data version, so changing `base_url` never serves stale data.
  *[improvement]*

- **Labels that follow a feature.** `centroid_dependence` scores points by their
  distance to the cluster center. A chosen feature column or a projection can supply
  the score instead. With `profile: step`, a label then takes one side of a threshold
  on that feature within each cluster. The label count sets the threshold, so it
  differs per cluster. The output also records, per feature, whether it carries label
  structure. A feature with neither role is noise. Roles are recorded after labeling,
  since a noise column used as the score carries label structure. A projection reports
  its weights. *[feature]*

- **Reading the diagnostic codes from Python.** a logging handler
  that collects the `[CLM-###]` codes a call logged, e.g. to check for `[CLM-306]`
  or `[CLM-309]` after `generate_clm_labels`. *[improvement]*

---

## 0.7.8 and 0.7.9

- **A Sphinx site.** *[feature]*

- **Dependency pins** exact pins to `>=` in `requirements.txt` and
  `environment.yml`. *[improvement]*

- **One source of truth for settings, before 0.8.0.** Every setting's default, the
  condition it applies under and its allowed values come from one table that the
  engine, the saved run config and the wizard all read. *[improvement]*

- **A community registry entry.** CLMSynth in the Research Software Directory or
  on WikiData, a one-time submission. *[improvement]*

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
