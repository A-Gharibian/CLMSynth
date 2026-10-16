  # CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.4-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.5 to 0.9.0. Current behavior is in **`../README.md`**.
Past changes are in **`../CHANGELOG.md`**.

## Conventions

A patch release (`0.0.x`) fixes specified behavior.
A minor release (`0.x.0`) ships a new capability.
A major release always uploads validation tests to the Zenodo repository.
`[CLM-###]` codes are never renumbered or reused.

---

## 0.7.5 Release version

Will be released to PyPI. Harmonised questions, diagnostics, tests and manual.

### Will be Fixed

- **The search grid misses a narrow feasible band.** Widen the bracket beyond the
  grid points. *[issue]*

- **Integer label counts must sum to `N`.** The `[CLM-106]` tolerance lets explicit
  `proportions` sum to within 1e-6 of 1. Above about a million points, the counts
  then sum to more than `N`, or to less: 80 too many at 100 million points for a sum
  of 1.0000008, 7 too few for 0.9999999. *[issue]*

- **The `clustbench` registries disagree.** `SOURCE_METADATA` and
  `CLUSTBENCH_DATASETS` list different batteries. `batteries: all` uses the larger
  one. *[improvement]*

### Will be Added

- **Plots drawn after the batch.** Workers write the CSVs with plots off, and the
  plots are drawn from the run folder afterwards, or not at all. *[feature]*

- **The size ceiling acted on, not just reported.** Refuse or warn before searching
  past the size ceiling. *[feature]*

---

## 0.7.6 and 0.7.7

**Source and generator extensions.** Data-source work, plus one engine hook that reads
features. Clusters stay fixed, read-only input to the engine.

### Will be Added

- **Repliclust.** It generates clusters from prompts. *[feature]*

- **Noise dimensions from `mdcgen`.** New presets set mdcgenpy's `n_noise`. The
  listed columns become uniform noise for every point, or for chosen clusters only.
  mdcgenpy validates `add_noise` but never applies it, so it is not used. The output
  records, per feature, whether it carries cluster structure. `clustbench` and `byoc`
  report it as unknown. *[feature]*

- **Labels that follow a feature.** `centroid_dependence` scores points by their
  distance to the cluster center. A chosen feature column or a projection can supply
  the score instead. With `profile: step`, a label then takes one side of a threshold
  on that feature within each cluster. The label count sets the threshold, so it
  differs per cluster. The output also records, per feature, whether it carries label
  structure. A feature with neither role is noise. Roles are recorded after labeling,
  since a noise column used as the score carries label structure. A projection reports
  its weights. *[feature]*

## 0.7.8 and 0.7.9

- **A Sphinx site.** *[feature]*

**Dependency pins**

- Relax exact pins to `>=` in `requirements.txt` and
  `environment.yml`. conda-forge discourages exact version pins. *[improvement]*

---

# 0.8.0

Will be submitted to the conda-forge review queue on 22 November 2026.

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
and a CLMSynth GUI are planned, which will use compiled C binaries for real-time resndering.

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

- **NCA and AAA as a `target_metric` type.** clustbench scores with NCA by default. NCA is
  asymmetric and has no known closed form. It needs a `_METRIC_FUNCS` entry and a
  global solve. *[feature]*
