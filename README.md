# Cluster–label matched dataset synthesizer


[![PyPI](https://img.shields.io/pypi/v/clmsynth)](https://pypi.org/project/clmsynth/)
[![Release](https://img.shields.io/github/v/release/A-Gharibian/CLMSynth?include_prereleases&sort=semver)](https://github.com/A-Gharibian/CLMSynth/releases)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/A-Gharibian/CLMSynth/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://github.com/A-Gharibian/CLMSynth/blob/main/LICENSE.txt)
[![CI](https://img.shields.io/github/actions/workflow/status/A-Gharibian/CLMSynth/ci.yml?branch=main&label=CI)](https://github.com/A-Gharibian/CLMSynth/actions/workflows/ci.yml)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blueviolet)](https://github.com/A-Gharibian/CLMSynth/blob/main/CITATION.cff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21759538.svg)](https://doi.org/10.5281/zenodo.21759538)

CLMSynth generates synthetic label columns on top of existing or generated clusters, 
with mathematically controlled agreement
(recall, class balance, spatial placement, or a solved target metric) 
against the ground-truth clusters.

A generated dataset carries the original features, the ground-truth cluster IDs, and one or more synthetic
labels whose relationship to those clusters is characterized by user-defined configuration.

## Install

Requires Python 3.12 or newer.

```bash
pip install clmsynth
```

That pulls the dependencies (compatible version ranges) and installs the console
scripts `clmsynth`, `clmsynth-config`, and `clmsynth-wizard`.

**Start with the wizard.** `clmsynth-wizard` asks a question per setting and
writes a runnable config, which is the entry point that needs no files from the
repository. `clmsynth-config` renders a config from `upstream_payload.yaml`
instead, and that file ships in the source distribution and the repository, not
in the wheel.

Working on CLMSynth itself, from a clone:

```bash
pip install -e ".[test]"
```

To pin the exact versions used for verification rather than the compatible
ranges, install those first:

```bash
pip install -r requirements.txt
```

[//]: # (Or as a conda environment &#40;mirrors the same versions&#41;:)

[//]: # ()
[//]: # (```bash)

[//]: # (conda env create -f environment.yml)

[//]: # (conda activate clmsynth)


Optional, depending on which source/utility you use:

```bash
pip install faker                                    # only used by the fabricated_data source
pip install git+https://github.com/CN-TU/mdcgenpy    # only needed for data_source: "mdcgen"
```

## Use directly 

**CLMSynth** integrates with both downstream and upstream python pipelines,
and can run without a config or YAML file (`load_config` is the only thing that reads YAML).
For example, you can run: `your algorithm → CLMSynth → measure → tune the
algorithm`; The engine is one pure function of two arrays, a dict and a seed.
The `dict` below (from the `clm_label` documented in the manual) runs directly:


```python
import numpy as np
from clmsynth import generate_clm_labels, clustering_mcc

clusters = np.repeat([0, 1, 2], 200)                        # an arbitrary ground truth,  shape (n,N)
X = np.repeat([[0., 0.], [5., 0.], [0., 5.]], 200, axis=0)  # a feature space, shape (n, d)
# configuration that can also be supplied through a YAML file:
cfg = {
    "num_classes": 3,
    "matching_mode": "custom",
    "assignment_matrix": [{"clusters": [k], "label": k} for k in range(3)],
    "target_metric": {"type": "mcc", "value": 0.6, "tolerance": 0.01},
}

for target in (0.2, 0.4, 0.6, 0.8):
    cfg["target_metric"]["value"] = target
    labels = generate_clm_labels(clusters, X, cfg, seed=42)   # pandas.Series, one entry per point
    achieved = clustering_mcc(clusters, labels)               # 0.1925, 0.4025, 0.6075, 0.8025
    ...                                                       # pass the dataset to an algorithm.
```

For several labels attached to one dataset rather than one label per call, the route
is `build_context` → `generate_additional_labels` → `DatasetContext.to_dataframe`,
which ends in a frame of features, `Cluster_n` and `Label_n` columns. A fetcher's
frame is only a `DataFrame` carrying a `GroundTruth_*` column, so you can build a
local array without going through the datasource route.

### Diagnostics
Every error and warning the engine raises carries a `[CLM-###]` code
(`1xx` config `ValueError`, `15x` `InfeasibleAllocationError`, `2xx` `Key Error ` and `3xx` warnings),
defined once in `clm_errors.py`. The full catalogue is available in the CLMSynth
User Manual under the *Configuration Troubleshooting Reference* section.

An invalid configuration 1xx error aborts the whole run for every dataset except:
[CLM-102/105/125/127], which are raised against each dataset's own cluster count,
cluster ids or feature columns, so a failure of configuration for one dataset does not apply to a batch of datasets.
For byoc, [CLM-104] and [CLM-105] are additionally checked across the whole batch before
any work begins, so an id mismatch refuses the run before a file is written and names
every erroneous dataset; everything else is reported per dataset and the run continues.

**Diagnostics are either exceptions, or logged (but not returned).** 
For example, requesting `mcc: 0.9` against clusters of 300/200/100 returns an ordinary `Series` of the right
length with the right label counts but with the achieved MCC of `0.5196`, which is
same solution if `0.6` was requested as a target. The achieved target is reported in the result logs and plots.

| band                                | is           | how you read the code                             |
|-------------------------------------|--------------|---------------------------------------------------|
| `[CLM-1xx]` `[CLM-15x]` `[CLM-2xx]` | an exception | `e.code`: `except ValueError as e: e.code == 102` |
| `[CLM-3xx]`                         | a warning    | match `[CLM-###]` in the message text             |

For troubleshooting the warnings and exceptions, refer to the troubleshooting section of the user manual.

## Quick start

**The wizard:** run `python -m clmsynth.config_wizard`. It asks a question for every setting,
writes the config YAML,
and can run the pipeline. Works for every source, including user-provided (`byoc`) data.
No YAML editing required.
CLMSynth is fully functional from a config file
alone; the wizard helps to *create* one.

1. How to generate a config (edit `upstream_payload.yaml`, or pass your own file):
   ```bash
   python -m clmsynth.generate_config                    # reads upstream_payload.yaml, writes test_data_config.yaml
   python -m clmsynth.generate_config my_payload.yaml    # or an explicit payload (and optionally an output path)
   ```

2. Run the pipeline:
   ```bash
   python -m clmsynth.main                   # reads test_data_config.yaml
   python -m clmsynth.main my_config.yaml    # or an explicit config path
   ```

## Output

Each run creates a folder:

```
OUTPUT/{DDMMYY}_{Source}_{HHMMSS}/
├── {config}.yaml                              # config used
├── csv/{source}__{battery}__{dataset}.csv     # features + Cluster_n + Label_n
├── png/{source}__{battery}__{dataset}__Cluster_0.png
├── png/{source}__{battery}__{dataset}__Label_0.png   # one per generated label
└── txt/{source}__{battery}__{dataset}.txt     # config + MCC/ARI (as shown on plots)
```

- `{Source}` is the human-facing generator name: `clustbench` → **Gagolewski's framework**,
  `mdcgen` → **MDCGen**, `fabricated_data` → **Fabricated**, `byoc` → Bring Your Own Clusters.
- The base folder is `global_settings.output_dir` (default `OUTPUT`); a numeric
  suffix is appended.
- **Column naming:** ground-truth class labeling such as `Cluster_0`, `Cluster_1`, …
  (by position, `source_labeling: labels0` surfaces as `Cluster_0`); generated
  labels become `Label_0`, `Label_1`, … (0-indexed, one per `n_labels`).
- The MCC/ARI printed in each plot subtitle and in the `.txt` summary are
  computed from the output CSV.

### Data sources

- **`clustbench`**, geometries downloaded from Gagolewski's **A Framework for Benchmarking Clustering Algorithms**
  [[1]](#references), whose Python API is imported as `clustbench` and which supplies this source's config key. 
  Every available reference labeling (`labels0`, `labels1`, …) is fetched.
- **`mdcgen`**, fully synthetic geometries via `mdcgenpy` [[2]](#references).
  Use when specific properties (dimensionality, overlap, outliers) is needed that the clustering-benchmarks datasets
  may not cover.
- **`fabricated_data`**, offline fallback. Cluster IDs are integers `0..K-1`, as in the other generated sources;
  the generator's readable class names are mapped to codes on import. The `labels_only_4class` preset emits 
  **cluster ids with no feature columns at all**, a valid CLM run, since recall, balance, allocation and spillover
  do not need feature values.
- **`byoc`**, bring-your-own-clusters: your own CSV with feature columns and one cluster-id column (see below).

### Bring-your-own-clusters (`byoc`)

Point the pipeline at your own CSV, feature columns with **exactly one** cluster-id column, and it generates CLM labels 
against *your* clusters:

```yaml
global_settings:
  data_source: "byoc"
  output_dir: "OUTPUT"

byoc_suite:
  batteries: ["local"]          # fixed label, kept for pipeline uniformity
  input_dir: "INPUT"            # folder holding your CSV(s)
  datasets: ["my_clusters"]     # file STEMS (no ".csv"); one output per file
  cluster_column: "group"       # the single ground-truth cluster column
  tag_columns: ["outcome"]      # optional: carried through, never features
  standardize: false            # optional: min-max rescale features to [0,1] at import
  seed: 42

label_generation:
  # ... identical to any other source ...
  clm_label:
    num_classes: 2
    matching_mode: "perfect"
```

- Every column other than `cluster_column` becomes a feature unless `tag_columns` names it
  (original names are kept, so plot axes show them as imported). Features must be numeric;
   a non-numeric column that is not declared a tag **rejects the file**.
- **`tag_columns`** lists columns that travel with the data without entering the CLM matching: an outcome variable,
  a second labeling from another method, an identifier. They are copied to the output CSV unchanged, are excluded from
  the geometry and from `standardize`, and need not be numeric.
- One `cluster_column` is required, the run is **rejected** (logged, dataset skipped) if it is missing or names
  more than one column.
- `standardize: true` min-max rescales the features to `[0, 1]` at import, applied once, before both the
   geometry/centroid math and the written CSV. Off by default.
- Cluster ids may be integers or strings; if you use `single_match`/`assignment_matrix`, match that id type.

### `matching_mode` reference

- `perfect`, fixed cluster↔label bijection; label counts are forced to the paired cluster sizes. 
   Requires `num_classes == K`. Proportions/balance/skew_rule are ignored.
- `single`, routes label `l*`'s point budget into cluster `k*`. 
   **Note:** the current implementation places up to `recall_target × m_{l*}` points of `l*` into `k*`,
   so it requires `|k*| ≥ m_{l*}` (see Known limitations).
- `random`, labels drawn from the resolved proportions, ignoring cluster structure entirely.
- `custom`, one or more explicit `assignment_matrix` rules, each routing a `recall_target` 
   fraction of one label's points into a set of clusters. Supports surjective (many clusters → one label),
   partial, and overlapping alignments. Unclaimed cluster capacity follows `spillover_rule`.

### Target-metric solving

`target_metric` (only under `single`/`custom`) has two scopes. In both, the
solver varies only the recall level; every setting above (`split_rule`,
`spillover_rule`, `competing_noise`, `proportions`) is held fixed across
probes, so noise structure changes the solved recall rather than being
applied after it.

- **`scope: "global"` (default)**, targets the whole-partition metric. A single
  global recall level is solved (coarse grid scan → bisection) so the achieved 
  MCC or ARI meets the requested value within tolerance. If the target exceeds
  what the geometry/proportions can
  reach, the solver returns its closest feasible value and logs a
  non-convergence warning. Works for `type: mcc` and `type: ari`.
- **`scope: "pair"` (`type: mcc`, `single` mode only)**, targets the `2×2` MCC
  of the `single_match` cluster/label pair (that cluster vs. the rest against
  that label vs. the rest). This inverts in **closed form**, so it is an
  exact solve, and the delivered pair MCC is measured afterward and reported as achieved
  (refer to the article for more information).
  The whole-partition `R_K` and ARI reported on the plots then serve as
  independent views of the same labeling.

> **What metric is being solved.** The global MCC is the multiclass Gorodkin's
> `R_K` that `clustering_mcc` computes (permutation-invariant via Hungarian
> matching), while the pair MCC is the `2×2` Matthews φ that `clustering_mcc_pair`
> computes for a single cluster/label pair. The global `R_K` and ARI between an
> `M`-label and a `K`-cluster partition have no closed form, so their solver is
> numerical; which is why `scope: pair` is exact. For that reason the pair (binary)
> measure is the natural choice for
> single-label to single-cluster matching, whereas the scikit-learn multiclass
> implementation (`R_K`) is meant for multi-label cases. The pair MCC can still be
> requested against a target cluster inside a custom-distribution configuration to
> check one label's precision and recall (rather than its specificity). If you are
> only interested in a single target label and cluster, prefer
> `matching_mode: single` with `scope: pair` over a custom distribution coupled
> with a binary MCC solver [[3]](#references). 

## Known limitations

- **`single` mode is budget-into-`k*`, not drain-`k*`-into-`l*`.** It tries to
  place label `l*`'s full budget `m_{l*}` inside cluster `k*`, so without a
  `target_metric` it raises `InfeasibleAllocationError` whenever `|k*| < m_{l*}`,
  e.g. pointing `single_match` at the *smallest* cluster with a large budget.
- **Target metric can be unreachable (structural ceiling)**, but the engine does not yet
  compute or report the ceiling value. *(planned for 0.7.5)*
- **Proportions are only possible with `spillover_rule: proportional_to_marginal`.**
  `uniform`/`concentrated` deliberately do not preserve the target label counts.
  - **`competing_noise` also breaks proportions**, Each entry converts
    leftover points of one cluster into one specific competing label,
    so achieved label counts are no longer held to `proportions`,
    and the achieved MCC/ARI differs from random-spillover noise.
  - **`balance: balanced` also ignores `proportions`** (enforces uniform 1/M) and warns.
- **Reachable `scope: pair` values are a coarse ladder near the bottom of the
  range.** The target label is sized to an integer number of points, so only a
  discrete set of pair-MCC values is reachable. `[CLM-307]` warns only when a request falls
  *outside* `[phi_min, 1]`; it says nothing about one falling between intervals inside
  that range, which `[CLM-310]` reports instead.
- **The target-metric search grid cannot see inside a narrow feasible band.**
  Candidate recalls are bracketed on 11 grid points (step 0.1); only feasible
  ones are usable, so the interval between the last feasible grid point and the
  true feasibility boundary is never explored. *(planned for 0.7.5)*
- **`plot_feature_scatter` is not thread-safe.** *(planned for 0.7.5)*

## References

1. Gagolewski, M. (2022). A framework for benchmarking clustering algorithms.
   *SoftwareX*, 20, 101270. <https://doi.org/10.1016/j.softx.2022.101270>
2. Iglesias, F., Zseby, T., Ferreira, D., et al. (2019). MDCGen:
   Multidimensional Dataset Generator for Clustering. *Journal of
   Classification*, 36, 599–618. <https://doi.org/10.1007/s00357-019-9312-3>
3. Gorodkin, J. (2004). Comparing two K-category assignments by a K-category
   correlation coefficient. *Computational Biology and Chemistry*, 28(5–6),
   367–374.

