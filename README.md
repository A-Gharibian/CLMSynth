# CLM Benchmark Data Synthesizer

Generates synthetic label columns on top of existing or generated cluster
geometries, with mathematically controlled agreement (recall, class balance,
spatial placement, or a solved target metric) against the ground-truth
clusters, per the Cluster-Label Matching (CLM) specification.

A generated dataset therefore carries three things, strictly row-aligned:
the original features, the ground-truth cluster IDs, and one or more synthetic
labels whose relationship to those clusters is characterized by user-defined
configuration, rather than measured after the execution and synthesis.

## Project layout

| File                               | Role                                                                                                                                                                                                                                                                                                                                                                                                     |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `clmsynth/main.py`                 | Entry point. Reads a config YAML (path optional; defaults to `test_data_config.yaml`), runs the pipeline, and packages each run into a self-contained output folder.                                                                                                                                                                                                                                     |
| `clmsynth/dataset_sources.py`      | Four interchangeable data sources: `clustbench` (Gagolewski benchmark downloads), `mdcgen` (synthetic, via `mdcgenpy`), `fabricated_data` (offline fallback, no deps), and Bring your own clusters (BYOC) as a comma-delimited file.                                                                                                                                                                     |
| `clmsynth/fabricated_generator.py` | Engineered-feature generator with perfect-separation labels, used by the `fabricated_data` source.                                                                                                                                                                                                                                                                                                       |
| `clmsynth/byoc_source.py`          | Bring-your-own-clusters source: reads a user CSV (feature columns + exactly one cluster-id column), with optional min-max standardization on import.                                                                                                                                                                                                                                                     |
| `clmsynth/label_context.py`        | `DatasetContext`, holds features, every ground-truth labeling, and every generated label; rejects any misaligned column.                                                                                                                                                                                                                                                                                 |
| `clmsynth/label_generator.py`      | Orchestrates label generation: calls the CLM engine per `n_labels`, or falls back to simple noise-flipping if no `clm_label` config is given.                                                                                                                                                                                                                                                            |
| `clmsynth/clm_label_engine.py`     | The CLM label-assignment math: proportions/skew, matching modes, recall targets, feasibility-checked allocation, spillover, structured competing noise, spatial (centroid) placement, and the global target-metric solver.                                                                                                                                                                               |
| `clmsynth/clm_errors.py`           | Diagnostics: message templates keyed by a `[CLM-###]` code (1xx `ValueError`, 15x `InfeasibleAllocationError`, 3xx warnings) plus helpers. `InfeasibleAllocationError` is defined here and re-exported from the engine.                                                                                                                                                                                  |
| `clmsynth/metrics.py`              | Standalone evaluation. Three external measures compare two labelings row by row without looking at the features: `clustering_mcc` (Hungarian-matched multiclass MCC / Gorodkin R_K), `clustering_mcc_pair` (the 2x2 Matthews phi of one cluster against one label, the quantity `target_metric.scope: pair` targets), and `clustering_ari` (adjusted Rand index). `evaluate_cluster_label_matching` is a **provisional, not-yet-implemented** internal-validity hook (see Known limitations); the pipeline never calls it. |
| `clmsynth/visualization.py`        | Scatter-plot rendering for any two features, colored by a chosen label column, annotated with MCC/ARI and the generating config.                                                                                                                                                                                                                                                                         |
| `clmsynth/config_template.py`      | The YAML template string used to render a config.                                                                                                                                                                                                                                                                                                                                                        |
| `clmsynth/generate_config.py`      | Renders `config_template.py` into a runnable config YAML from an upstream payload file (default `upstream_payload.yaml`).                                                                                                                                                                                                                                                                                |
| `upstream_payload.yaml`            | Example upstream payload: the minimal facts `generate_config.py` renders into the full config.                                                                                                                                                                                                                                                                                                           |
| `clmsynth/config_wizard.py`        | Interactive CLI wizard: asks and explains every option in plain language, then writes a config YAML (all sources, including `byoc`) and can launch the run.                                                                                                                                                                                                                                              |
| `test_data_config.yaml`            | The default config `main.py` reads. Generated by `generate_config.py`, or hand-edited, or passed explicitly as `python -m clmsynth.main my_config.yaml`.                                                                                                                                                                                                                                                 |

## Install

Requires Python 3.11 or newer. Exact dependency versions are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or as a conda environment (mirrors the same versions):

```bash
conda env create -f environment.yml
conda activate clmsynth
```

Or as an installable package (compatible version ranges, console scripts
`clmsynth`, `clmsynth-config`, `clmsynth-wizard`):

```bash
pip install .
```

Optional, depending on which source/utility you use:
```bash
pip install faker                                    # only used by the fabricated_data source, has a fallback if absent
pip install git+https://github.com/CN-TU/mdcgenpy    # only needed for data_source: "mdcgen"
```
> **`pyivm` is not implemented yet.** `metrics.evaluate_cluster_label_matching`
> ships as a provisional hook against the adjusted internal-validity measures,
> but the dependency has not been adopted or verified and nothing in the
> pipeline calls it. Treat that function as unsupported until a later release.

## Quick start

**The wizard:** run `python -m clmsynth.config_wizard`. It asks a question for every setting, writes the config YAML,
and can run the pipeline. Works for every source, including user-provided (`byoc`) data. No YAML editing required.

Or configure it yourself:

1. Generate a config (edit `upstream_payload.yaml`, or pass your own payload file):
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
- **Column naming:** ground-truth class labeling such as `Cluster_0`, `Cluster_1`, …
  (by position, `source_labeling: labels0` surfaces as `Cluster_0`); generated
  labels become `Label_0`, `Label_1`, … (0-indexed, one per `n_labels`).
- The MCC/ARI printed in each plot subtitle and in the `.txt` summary are
  computed from the written CSV columns themselves (single source of truth).

## Config schema

```yaml
global_settings:
  data_source: "clustbench"        # clustbench | mdcgen | fabricated_data | byoc
  output_dir: "OUTPUT"             # base folder for the timestamped run folders

clustbench_suite:                  # key must be "{data_source}_suite"
  batteries: ["sipu"]               # "all" or a list
  datasets: ["unbalance"]           # "all" or a list
  seed: 42                          # only used by mdcgen/fabricated_data; ignored by clustbench

label_generation:
  n_labels: 1                       # produces Label_0, Label_1, ...
  source_labeling: "labels0"        # which ground-truth labeling to key CLM math off
  noise: 0.15                       # fallback only, if clm_label is omitted
  seed: 42

  clm_label:
    # 1. CARDINALITY & BALANCE ------------------------------------------------
    num_classes: 3                  # M, the number of labels (1-64; see Known limitations)
    balance: "unbalanced"           # balanced -> uniform 1/M (proportions ignored, warns)
                                     # unbalanced -> proportions below, else skew_rule
    proportions: [0.5, 0.3, 0.2]    # used directly when balance == "unbalanced"
    skew_rule: "geometric"          # fallback when unbalanced AND no proportions given:
                                     #   geometric          p_i ~ ratio^i
                                     #   dominant_minority  one dominant class, uniform rest
                                     #   dirichlet          Dirichlet(alpha), drawn once per seed
    skew_params: {ratio: 0.5, dominant_index: 0, dominant_share: 0.6, alpha: 1.0}

    # 2. MATCHING MODE --------------------------------------------------------
    matching_mode: "custom"         # perfect | single | random | custom
    single_match: {cluster: 4, label: 0}       # required if matching_mode == single
    assignment_matrix:                          # required if matching_mode == custom
      - {label: 0, clusters: [1, 2], recall_target: 0.8}
      - {label: 1, clusters: [3],    recall_target: 0.5}
      - {label: 2, clusters: [3],    recall_target: 0.3}

    # 3. ALLOCATION -----------------------------------------------------------
    split_rule: "proportional_to_size"          # proportional_to_size | equal
    spillover_rule: "proportional_to_marginal"  # proportional_to_marginal | uniform | concentrated
                                     # NOTE: only proportional_to_marginal makes the achieved
                                     # label counts honor 'proportions'. uniform spreads leftover
                                     # points evenly; concentrated dumps them into one label.

    # 4. STRUCTURED COMPETING NOISE (optional; single/custom only) ------------
    competing_noise:
      - {cluster: 1, label: 2, share: 1.0, favors: "boundary"}
                                     # converts `share` of ONE cluster's UNCLAIMED
                                     # (leftover) points into a specific competing
                                     # label, placed at that cluster's boundary/
                                     # core/random. Withdrawn from the leftover pool
                                     # BEFORE spillover_rule above fills the rest.
                                     # Deliberately bypasses 'proportions' (like
                                     # uniform/concentrated) and changes the achieved
                                     # MCC/ARI; that structured-vs-random contrast
                                     # is its point.

    # 5. TARGET METRIC (optional; single/custom only) -------------------------
    target_metric: {type: "mcc", value: 0.6, tolerance: 0.01, max_iter: 40}
                                     # solves one global recall level so the achieved
                                     # mcc/ari meets 'value'; recall_target may be omitted
                                     # from the rules when this is present.
                                     # scope (mcc only): "global" (default) targets the
                                     # whole-partition multiclass MCC by numerical search;
                                     # "pair" (single mode only) targets the 2x2 MCC of the
                                     # single_match cluster/label and is solved exactly and
                                     # instantly, sizing that label to sit inside its cluster.
    # target_metric: {type: "mcc", scope: "pair", value: 0.6}   # exact single-pair MCC

    # 6. SPATIAL PLACEMENT (optional) -----------------------------------------
    centroid_dependence:
      enabled: true
      profile: "linear"             # linear | exponential | step
      favors: "core"                # core | boundary
      steepness: 3.0                # exponential only
```

### Data sources

- **`clustbench`**, real, fixed geometries downloaded from Gagolewski's benchmark suite (v1.1.0). Every available reference labeling (`labels0`, `labels1`, …) is fetched. Use for reproducible research results.
- **`mdcgen`**, fully synthetic geometries via `mdcgenpy`, seeded for reproducibility. Use when you need geometric properties (dimensionality, overlap, outliers) the fixed clustbench datasets don't cover.
- **`fabricated_data`**, no network, no extra dependencies. Offline fallback. Cluster IDs are integers `0..K-1`, as in the other generated sources; the generator's readable class names are mapped to codes on import.
- **`byoc`**, bring-your-own-clusters: your own CSV with feature columns and exactly one cluster-id column (see below).

### Bring-your-own-clusters (`byoc`)

Point the pipeline at your own CSV, feature columns plus **exactly one** cluster-id column, and it generates CLM labels against *your* clusters:

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
- `custom`, one or more explicit `assignment_matrix` rules, each routing a `recall_target` fraction of one label's budget into a set of clusters. Supports surjective (many clusters → one label), partial, and overlapping alignments. Unclaimed cluster capacity follows `spillover_rule`.

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

> **What metric is being solved.** The global MCC is the multiclass Gorodkin
> `R_K` that `clustering_mcc` computes (permutation-invariant via Hungarian
> matching); the pair MCC is the `2×2` Matthews φ that `clustering_mcc_pair`
> computes for one cluster/label pair. The global `R_K`/ARI between an
> `M`-label and a `K`-cluster partition has no closed form, so its solver is
> numerical; the single-pair MCC does, which is why `scope: pair` is exact.

### Diagnostics

Every error and warning the engine raises carries a `[CLM-###]` code
(`1xx` config `ValueError`, `15x` `InfeasibleAllocationError`, `3xx` warnings),
defined once in `clm_errors.py`. The full catalogue is available in the CLMSynth
User Manual under the Troubleshooting appendix.

## Known limitations

- **`single` mode is budget-into-`k*`, not drain-`k*`-into-`l*`.** It tries to
  place label `l*`'s full budget `m_{l*}` inside cluster `k*`, so it raises
  `InfeasibleAllocationError` whenever `|k*| < m_{l*}`, e.g. pointing
  `single_match` at the *smallest* cluster with a large label budget.
- **Target metric can be unreachable (structural ceiling).** MCC/ARI between an
  `M`-label partition and a `K`-cluster partition is structurally bounded when
  `M < K`. For `K` equally sized clusters, the ceiling of a balanced `M`-coarsening
  has the closed form **`MCC = sqrt(M(M-1) / (K(K-1)))`** (verified exact, e.g.
  4 labels over 20 clusters cap MCC at `0.178`).
  For unequal clusters, or a specific rule set, the reachable ceiling is the MCC
  the rules achieve at full recall (`alpha = 1`). A chosen skew (e.g.
  `dominant_minority`) constrains it further via fixed label sizes. When the
  target exceeds the ceiling, the solver reports best-effort + a `[CLM-306]`
  non-convergence warning.
- **Proportions are only honored with `spillover_rule: proportional_to_marginal`.**
  `uniform`/`concentrated` deliberately do not preserve the target label counts.
- **`competing_noise` also breaks proportions, by design.** Each entry converts
  leftover points of one cluster into one specific competing label (placed
  boundary/core/random), so achieved label counts deviate from `proportions`,
  and the achieved MCC/ARI differs from random-spillover noise, that contrast
  is the feature's purpose. Only valid under `single`/`custom`; a warned no-op
  under `perfect` (no leftover capacity); rejected under `random`.
- **`balance: balanced` ignores `proportions`** (enforces uniform 1/M) and warns.
- **`skew_rule: dirichlet`** is stochastic but reproducible: it draws once from
  the run seed, so a fixed seed yields fixed proportions.
- **`pyivm` is not implemented yet.** `evaluate_cluster_label_matching` is unsupported until a later release (see Install).
- **A solved `target_metric` is verified against the delivered labeling, and can
  miss.** `solve_alpha_for_target_metric` scores candidate recalls on a fixed
  probe stream (`default_rng(probe_seed)`, `probe_seed` defaulting to 0) so that
  candidates compare fairly, but the labeling that is written out is generated on
  the run's own stream (`default_rng(seed)`), so a search that converged
  internally can still deliver a labeling outside tolerance. The generator now
  measures what it actually writes and raises **`[CLM-309]`** when that value
  falls outside tolerance, *treat the achieved value as authoritative, not the
  requested one*. Magnitude is governed by `N`: ≤0.009 at `N=3000`, ≤0.007 at
  `N=8000`, but up to 0.07 at `N`≈120–280. Does not affect `scope: pair`
  (closed-form, no search).
- **The target-metric search grid cannot see inside a narrow feasible band.**
  Candidate recalls are bracketed on 11 grid points (step 0.1); only feasible
  ones are usable, so the interval between the last feasible grid point and the
  true feasibility boundary is never explored.
- **`plot_feature_scatter` is not thread-safe.** matplotlib's default (TkAgg)
  backend requires plotting on the main thread; `main.py` only calls it
  sequentially.

## Possible future additions

Two extensions as candidates:

- **SYNLABEL, scoped to `fabricated_generator.py`, not the CLM engine.**
  SYNLABEL derives a noiseless functional labeling from a feature space and
  injects measured noise by resampling features, useful here because
  `fabricated_generator.py`'s current ground truth is a single crude rule
  (a percentile split on `Feature_1`). A SYNLABEL-derived labeling would be a
  more principled synthetic ground truth for the offline source. The
  boundary matters: SYNLABEL must produce `c(x)`/`X` *before* the CLM engine
  ever sees them, staying strictly in the data-source layer (alongside
  `clustbench`/`mdcgen`/`byoc`).
- **`fabricated_generator.py` emitting more than one ground-truth column.**
  Real `clustbench` datasets already ship multiple reference labelings
  (`GroundTruth_labels0`, `GroundTruth_labels1`, ...), and `DatasetContext`
  already consumes any number of them generically (`build_context` collects
  every `GroundTruth_*` column). The offline fabricator currently produces
  exactly one. Extending it to emit several labelings derived from different
  mathematical properties of the same synthetic feature space, e.g. a radial
  split, a linear combination, a nonlinear boundary, would let the offline
  source mimic that same multi-labeling structure without touching
  `label_context.py` or the label engine at all; the plumbing already
  supports it. This is the same territory Gagolewski's benchmarking
  framework and adjusted internal-validity-measure work (Gagolewski, 2022,
  *SoftwareX*; Jeon et al., 2025, *TPAMI*) explore for real datasets, characterizing a
  dataset by more than one structural criterion at once, applied here to a
  synthetic one.
