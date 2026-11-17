  # CLMSynth Roadmap

[![Version](https://img.shields.io/badge/version-0.7.8-blue)](https://github.com/A-Gharibian/CLMSynth/releases)

Planned work from 0.7.8 to 0.9.0. Current behavior is in **`../README.md`**.
Past changes are in **`../CHANGELOG.md`**.

## Conventions

A patch release (`0.0.x`) fixes specified behavior.
A minor release (`0.x.0`) ships a new capability.
A major release uploads validation tests to the Zenodo repository.
`[CLM-###]` codes are never renumbered or reused.

---

## 0.7.9

- **A site.** *[feature]*

- Fable 5.2, Gemini 4 Carbon and GPT-6 Sol/Astra code review improvements. *[improvement]*

- SPEC0
---

# 0.8.0

Will be submitted to the conda-forge review queue on 22 November 2026.

### Distribution

- **conda-forge** `noarch: python` build. *[feature]*

- **Signed tags, and stability commitment.** The `[CLM-###]` registry and `__all__` follow semantic
  versioning.

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
  most of its tries are repeats.

- **Python 3.15 and Pandas 3.x features, once the minimum.** `frozendict` for the read-only
  registries, closed `TypedDict` so a type checker flags a misspelled key in a config
  written in Python, and sampling profiler (PEP 799) for the target-metric search and large runs. str type for Pandas. *[improvement]*

- **`clmsynth-config` in a pipeline.** Rendered configs piped straight into runs,
  e.g. calibration datasets at MCC 0 to 1 in steps of 0.1. *[feature]*

- **Integer network flow.** Since 0.7.5 every count is rounded from the
  written values, and the rules' counts are rounded together with their clusters as
  one integral flow, so `[CLM-150]` and `[CLM-151]` refuse only when no rounding of
  the rules' counts fits. The label counts are still rounded first. A transportation formulation over the label counts, the
  rules and the clusters gives integer counts in one exact solve. `competing_noise`
  and spillover stay outside it: they take what the matching leaves. *[improvement]*