# Cluster–label matched dataset synthesizer


[![Release](https://img.shields.io/github/v/release/A-Gharibian/CLMSynth?include_prereleases&sort=semver)](https://github.com/A-Gharibian/CLMSynth/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE.txt)
[![CI](https://img.shields.io/github/actions/workflow/status/A-Gharibian/CLMSynth/ci.yml?branch=main&label=CI)](https://github.com/A-Gharibian/CLMSynth/actions/workflows/ci.yml)

[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blueviolet)](CITATION.cff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21751081.svg)](https://doi.org/10.5281/zenodo.21751081)

Generates synthetic label columns on top of existing or generated clusters, 
with mathematically controlled agreement
(recall, class balance, spatial placement, or a solved target metric) 
against the ground-truth clusters.

A generated dataset therefore carries three things:
the original features, the ground-truth cluster IDs, and one or more synthetic
labels whose relationship to those clusters is characterized by user-defined
configuration.

## Install

Requires Python 3.11 or newer. **Install the package before running**:

```bash
pip install -e .
```

That pulls the dependencies (compatible version ranges) and installs the console
scripts `clmsynth`, `clmsynth-config`, and `clmsynth-wizard`. Use `pip install .`
instead of `-e` for a non-editable install.

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

## Quick start

**The wizard:** run `python -m clmsynth.config_wizard`. It asks a question for every setting,
writes the config YAML,
and can run the pipeline. Works for every source, including user-provided (`byoc`) data.
No YAML editing required.
CLMSynth is fully functional from a config file
alone; the wizard only *creates* one, for a user who would rather answer
questions than write YAML. Deleting `config_wizard.py` leaves a working program,
so core never imports it and the dependency runs one way only (wizard → core), a
property a test now pins.

1. How to generate a config (edit `upstream_payload.yaml`, or pass your own payload file):
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

Each run creates a time-stamped folder:

```
OUTPUT/{DDMMYY}_{Source}_{HHMMSS}/
├── {config}.yaml                              # exact config used, copied in
├── csv/{source}__{battery}__{dataset}.csv     # features + Cluster_n + Label_n
├── png/{source}__{battery}__{dataset}__Cluster_0.png
├── png/{source}__{battery}__{dataset}__Label_0.png   # one per generated label
└── txt/{source}__{battery}__{dataset}.txt     # config + MCC/ARI (as shown on plots)
```

- `{Source}` is the human-facing generator name: `clustbench` → **Gagolewski**,
  `mdcgen` → **MDCGen**, `fabricated_data` → **Fabricated**, `byoc` → Bring Your Own Clusters.
- The base folder is `global_settings.output_dir` (default `OUTPUT`); a numeric
  suffix is appended if two runs land in the same second.
- **A run that writes no dataset leaves no folder.** The run folder is created
  before the pipeline starts, so it exists the moment the first dataset
  succeeds; if none does, it is removed on the way out rather than left looking
  like a completed run. Only a folder holding nothing but the config copy and
  three empty subfolders is removed, anything written keeps its folder.
- **Column naming:** ground-truth class labeling such as `Cluster_0`, `Cluster_1`, …
  (by position, `source_labeling: labels0` surfaces as `Cluster_0`); generated
  labels become `Label_0`, `Label_1`, … (0-indexed, one per `n_labels`).
- The MCC/ARI printed in each plot subtitle and in the `.txt` summary are
  computed from the written CSV columns themselves (single source of truth).
- If a run fails (no data provided, wrong shape of data, etc.), only the config is retained
  and the rest of the folders are removed.

### Data sources

- **`clustbench`**, real, fixed geometries downloaded from Gagolewski's **clustering-benchmarks** framework[^gagolewski], whose Python API is imported as `clustbench` and which supplies this source's config key. Every available reference labeling (`labels0`, `labels1`, …) is fetched. Use for reproducible research results.
- **`mdcgen`**, fully synthetic geometries via `mdcgenpy`[^mdcgen], seeded for reproducibility. Use when specific properties (dimensionality, overlap, outliers) is needed that the clustering-benchmarks datasets may not cover.
- **`fabricated_data`**, offline fallback. Cluster IDs are integers `0..K-1`, as in the other generated sources; the generator's readable class names are mapped to codes on import. The `labels_only_4class` preset emits **cluster ids with no feature columns at all**, a valid CLM run, since recall, balance, allocation and spillover never read coordinates. Spatial placement does, so combining that preset with `centroid_dependence` is refused with `[CLM-125]`.
- **`byoc`**, bring-your-own-clusters: your own CSV with feature columns and one cluster-id column (see below).

### Bring-your-own-clusters (`byoc`)

Point the pipeline at your own CSV, feature columns plus **exactly one** cluster-id column, and it generates CLM labels 
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
  standardize: false            # optional: min-max rescale features to [0,1] at import
  seed: 42

label_generation:
  # ... identical to any other source ...
  clm_label:
    num_classes: 2
    matching_mode: "perfect"
```

- Every **numeric** column other than `cluster_column` becomes a feature (original names are kept, so plot axes show *your* names); non-numeric columns are ignored with a warning.
- Exactly one `cluster_column` is required, the run is **rejected** (logged, dataset skipped) if it is missing or names more than one column.
- `standardize: true` min-max rescales the features to `[0, 1]` at import, applied once, before both the geometry/centroid math and the written CSV. Off by default.
- Cluster ids may be integers or strings; if you use `single_match`/`assignment_matrix`, match that id type.

### `matching_mode` reference

- `perfect`, fixed cluster↔label bijection; label counts are forced to the paired cluster sizes. Requires `num_classes == K`. Proportions/balance/skew_rule are ignored (logged).
- `single`, routes label `l*`'s point budget into cluster `k*`. **Note:** the current implementation places up to `recall_target × m_{l*}` points of `l*` into `k*`, so it requires `|k*| ≥ m_{l*}` (see Known limitations).
- `random`, labels drawn from the resolved proportions, ignoring cluster structure entirely.
- `custom`, one or more explicit `assignment_matrix` rules, each routing a `recall_target` fraction of one label's points into a set of clusters. Supports surjective (many clusters → one label), partial, and overlapping alignments. Unclaimed cluster capacity follows `spillover_rule`.

### Target-metric solving

`target_metric` (only under `single`/`custom`) has two scopes. In both, the
solver varies only the recall level; every setting above (`split_rule`,
`spillover_rule`, `competing_noise`, `proportions`) is held fixed across
probes, so noise structure changes the solved recall rather than being
applied after it.

- **`scope: "global"` (default)**, targets the whole-partition metric. A single
  global recall level is solved (coarse grid scan → bisection, common random
  numbers across probes) so the achieved MCC or ARI meets the requested value
  within tolerance. If the target exceeds what the geometry/proportions can
  reach, the solver returns its closest feasible value and logs a
  non-convergence warning rather than crashing or fabricating a hit. Works for
  `type: mcc` and `type: ari`.
- **`scope: "pair"` (`type: mcc`, `single` mode only)**, targets the `2×2` MCC
  of the `single_match` cluster/label pair (that cluster vs. the rest against
  that label vs. the rest). This inverts in **closed form**, so it is hit
  exactly and instantly with no search: the label is sized to sit entirely
  inside its cluster (a single-dominant subset) so the pair MCC equals the
  request. The whole-partition `R_K` and ARI reported on the plots then serve as
  independent (chance-adjusted) views of the same labeling.

> **What metric is being solved.** The global MCC is the multiclass Gorodkin's
> `R_K`[^gorodkin] that `clustering_mcc` computes (permutation-invariant via Hungarian
> matching), while the pair MCC is the `2×2` Matthews φ that `clustering_mcc_pair`
> computes for a single cluster/label pair. The global `R_K` and ARI between an
> `M`-label and a `K`-cluster partition have no closed form, so their solver is
> numerical; the single-pair MCC does have one, which is why `scope: pair` is
> exact. For that reason the pair (binary) measure is the natural choice for
> single-label to single-cluster matching, whereas the scikit-learn multiclass
> implementation (`R_K`) is meant for multi-label cases. The pair MCC can still be
> requested against a target cluster inside a custom-distribution configuration to
> check one label's precision and recall (rather than its specificity). If you are
> only interested in a single target label and cluster, prefer
> `matching_mode: single` with `scope: pair` over a custom distribution coupled
> with a binary MCC solver.

### Diagnostics

Every error and warning the engine raises carries a `[CLM-###]` code
(`1xx` config `ValueError`, `15x` `InfeasibleAllocationError`, `3xx` warnings),
defined once in `clm_errors.py`. The full catalogue is available in the CLMSynth
User Manual under the Troubleshooting appendix.

A coded 1xx error aborts the whole run: a malformed config is equally wrong for every dataset.
The exceptions are [CLM-102/105/125/127], judged against each dataset's own cluster count,
cluster ids or feature columns, so a failure there says nothing about the next dataset.
For byoc, [CLM-104] and [CLM-105] are additionally checked across the whole batch before
any work begins, so an id mismatch refuses the run before a file is written and names
every erroneous dataset; everything else is reported per dataset and the run continues.

## Known limitations

- **`single` mode is budget-into-`k*`, not drain-`k*`-into-`l*`.** It tries to
  place label `l*`'s full budget `m_{l*}` inside cluster `k*`, so it raises
  `InfeasibleAllocationError` whenever `|k*| < m_{l*}`, e.g. pointing
  `single_match` at the *smallest* cluster with a large label budget.
- **Target metric can be unreachable (structural ceiling).** MCC/ARI between an
  `M`-label partition and a `K`-cluster partition is structurally bounded when
  `M < K`. For `K` equally sized clusters, the ceiling of a balanced `M`-coarsening
  has the closed form **`MCC = sqrt(M(M-1) / (K(K-1)))`**.
  For unequal clusters, or a specific rule set, the reachable ceiling is the MCC
  the rules achieve at full recall (`alpha = 1`). A chosen skew (e.g.
  `dominant_minority`) constrains it further via fixed label sizes. When the
  target exceeds the ceiling, the solver reports best-effort + a `[CLM-306]`
  non-convergence warning. *(planned for 0.7.0)*
- **Proportions are only honored with `spillover_rule: proportional_to_marginal`.**
  `uniform`/`concentrated` deliberately do not preserve the target label counts:
  when either rule leaves the delivered counts off
  their target, it warns `[CLM-304]`, the same code `competing_noise` raises,
  because it is the same fact. The warning is checked against the counts actually written,
  so a rule set that leaves no unclaimed points should not warn.
- **`concentrated_labels` must be a *list* of existing label ids.** Under
  `spillover_rule: concentrated` the value is drawn from directly and written
  into the label column, so it is validated up front (`[CLM-128]`): every entry
  must be an integer in `0..num_classes-1`. Two forms are rejected rather than
  guessed at, because both used to corrupt the output silently: a bare number
  (`concentrated_labels: 99`) is read by numpy as a *range*, scattering the
  remainder over that many labels instead of concentrating it, and a
  noninteger (`[1.5]`) is truncated to `[1]` when written to the integer label
  column. Validation runs *before* the `target_metric` solver, so a solved score
  can never be reported for a labeling that contains an undeclared label. Omit
  the key to take the default, the single largest label.
- **`competing_noise` also breaks proportions, by design.** Each entry converts
  leftover points of one cluster into one specific competing label (placed
  boundary/core/random), so achieved label counts deviate from `proportions`,
  and the achieved MCC/ARI differs from random-spillover noise, that contrast
  is the feature's purpose. Only valid under `single`/`custom`; a warned no-op
  under `perfect` (no leftover capacity); rejected under `random`.
- **`balance: balanced` ignores `proportions`** (enforces uniform 1/M) and warns.
- **A solved `target_metric` can still miss, and the achieved value is the
  authoritative one.** The generator measures what it actually writes rather than
  what the search believed, and raises **`[CLM-309]`** when that value falls
  outside `tolerance`. Two causes remain: the target is beyond what the geometry
  and proportions can reach, in which case `[CLM-306]` reports it and is the real
  explanation; or at small `N` the achievable values form a coarse ladder and a
  tight tolerance falls between two rungs. `scope: pair` has its own, sharper
  version of that ladder, below.
- **Reachable `scope: pair` values are a coarse ladder near the bottom of the
  range.** The target label is sized to an integer number of points, so only a
  discrete set of pair-MCC values is reachable, and the rungs widen sharply as the
  target approaches `phi_min`. With `N = 800` and a 200-point cluster, one point
  gives `0.0613` and two give `0.0867`, a gap of `0.025`, wider than the default
  `tolerance`. A target between two rungs is unreachable, so the closest rung is
  delivered and `[CLM-310]` reports the miss. `[CLM-307]` only clamps targets
  *outside* `[phi_min, 1]`; it says nothing about one falling between rungs inside
  that range. Inherent to the closed form, not a defect.
- **The target-metric search grid cannot see inside a narrow feasible band.**
  Candidate recalls are bracketed on 11 grid points (step 0.1); only feasible
  ones are usable, so the interval between the last feasible grid point and the
  true feasibility boundary is never explored. *(planned for 0.7.0)*
- **`plot_feature_scatter` is not thread-safe.** matplotlib's default (TkAgg)
  backend requires plotting on the main thread; `main.py` only calls it
  sequentially. *(planned for 0.8.0)*

[^gorodkin]: Gorodkin, J. (2004). Comparing two K-category assignments by a
    K-category correlation coefficient. *Computational Biology and Chemistry*,
    28(5–6), 367–374.

[^gagolewski]: Gagolewski, M. (2022). A framework for benchmarking clustering
    algorithms. *SoftwareX*, 20, 101270.
    <https://doi.org/10.1016/j.softx.2022.101270>

[^mdcgen]: Iglesias, F., Zseby, T., Ferreira, D., et al. (2019). MDCGen:
    Multidimensional Dataset Generator for Clustering. *Journal of
    Classification*, 36, 599–618. <https://doi.org/10.1007/s00357-019-9312-3>

