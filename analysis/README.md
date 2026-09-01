# Downstream Analysis (`analysis/`)

This module implements the analysis stage described in **Sections 2.6, 2.7, and
3.8** of the paper — the steps that run *after* per-image feature extraction
(`pipeline.py`) and turn per-plant, per-date feature outputs into the temporal
trends, statistical tests, and multivariate summaries reported in the Results
(Figures 3h, 8, 10, and 11).

It is a separate, pipeline-agnostic package: it consumes a tidy feature table
(one row per plant × imaging date, one column per feature) rather than being
wired directly into `PlantPipeline.run()`, so it can be pointed at output from
this pipeline, a re-run with different settings, or an externally produced
feature matrix (e.g. the maize cold-stress case study).

```
Feature extraction (pipeline.py)
        │  writes: output_folder/YYYY_MM_DD/plantN/{vegetation_indices,texture,morphology}/*.json
        ▼
analysis.collect.collect_feature_matrix()      ── flattens to one tidy feature-matrix table
        │
        ▼
analysis.feature_matrix                        ── temporal aggregation, distribution descriptors,
        │                                          384 → 240 vegetation-index reduction (Sec. 2.7)
        ▼
analysis.statistics                             ── control-mean t-tests, robust median/MAD z-tests,
        │                                          Benjamini–Hochberg FDR, group-level LMM (Sec. 2.6/2.7)
        │
analysis.multivariate                           ── PCA / LDA / hierarchical clustering (Sec. 3.5, 3.8)
        │
analysis.temporal                               ── per-plant trajectories, mean-difference-from-control
        ▼
analysis.case_studies                           ── Case Study 1 (sorghum) / Case Study 2 (maize) drivers
```

---

## Modules

| File | Paper section | Contents |
|------|---------------|----------|
| `collect.py` | 3.7 | Walk a `PlantPipeline` `output_folder` and flatten its per-plant JSON outputs into one tidy feature-matrix CSV (the "863-dimensional CSV feature matrix"). |
| `feature_matrix.py` | 2.6, 2.7 | `distribution_descriptors` (mean, std, min, max, median, IQR, skewness, kurtosis, Shannon entropy); `build_temporal_matrix` (aggregate by plant × date); `reduce_vegetation_index_features` (the 384 → 240 vegetation-index reduction). |
| `statistics.py` | 2.6, 2.7 | `benjamini_hochberg` FDR correction; `control_mean_ttest` and `robust_z` (median/MAD); `rank_individuals_by_divergence` (individual-level analysis, e.g. Plant47); `group_level_lmm` (`statsmodels` MixedLM with plant-identity random intercept, e.g. Group 2). |
| `multivariate.py` | 3.5, 3.8 | `run_pca`, `run_lda`, `hierarchical_clustering` on the phenotyping feature space. |
| `temporal.py` | 3.5, 3.8.1 | `plant_trajectory`, `mean_difference_from_control`, and matching plot helpers (Figure 8a / 10a style). |
| `case_studies.py` | 2.7, 3.8 | `run_sorghum_treatment_analysis` (Case Study 1) and `run_maize_coldstress_analysis` (Case Study 2) — end-to-end drivers that call the modules above and write result CSVs/plots. |
| `run_analysis.py` | — | CLI: `python -m analysis.run_analysis ...` |

---

## Input tables

### Feature matrix

A tidy table with `plant_id` (or `image_id`) and `date` identifier columns plus
one column per feature. Produce it with:

```python
from analysis import collect_feature_matrix

features = collect_feature_matrix("output/sorghum_run", save_csv="features.csv")
```

which reads every `output_folder/YYYY_MM_DD/plantN/{vegetation_indices,texture,morphology}/*.json`
file the pipeline wrote and names columns `veg__<INDEX>__<stat>`,
`texture__<band>__<descriptor>__<stat>`, and `morph__<trait>`. Vegetation-index
columns must use the `veg__` prefix for `reduce_vegetation_index_features` and
the case-study drivers to find them.

If you already have a feature CSV (e.g. from a non-multispectral imaging
system, as in the maize cold-stress case study), just load it with
`pandas.read_csv` — the rest of the package only assumes the tidy shape above.

### Design table

A small CSV you provide describing the experimental groups, joined to the
feature matrix on `plant_id` (or `image_id`):

```csv
plant_id,group
plant1,NT
plant2,NT
plant7,G1
...
```

For the maize case study, `design` instead maps `image_id` to `genotype` and
`stage` (treatment stage: `before`, `10C`, `4C`, `recovery_10C`, `recovery_4C`).

---

## Example: Case Study 1 (sorghum treatment-level comparison, Sec. 3.8.1)

```python
import pandas as pd
from analysis import collect_feature_matrix, run_sorghum_treatment_analysis

features = collect_feature_matrix("output/sorghum_run")
design = pd.read_csv("sorghum_design.csv")  # plant_id -> group (G1..G7, NT)

results = run_sorghum_treatment_analysis(
    features, design,
    output_dir="analysis_out/sorghum",
    control_label="NT",
)

print(results["individual_table"].sort_values("fdr_significant", ascending=False).head())
print(results["group_counts"])
```

Writes `individual_divergence_summary.csv`, `group_lmm_contrasts.csv`,
`group_significance_counts.csv`, and `group_level_feature_significance.png` to
`output_dir`.

## Example: Case Study 2 (maize cold-stress PCA, Sec. 3.8.2)

```python
import pandas as pd
from analysis import run_maize_coldstress_analysis

features = pd.read_csv("maize_features.csv")  # image_id + veg__* RGB vegetation-index columns
design = pd.read_csv("maize_design.csv")      # image_id -> genotype, stage

results = run_maize_coldstress_analysis(features, design, output_dir="analysis_out/maize")
print(results["pca"].explained_variance_ratio)
print(results["pca"].top_loadings(component=0, n=10))
```

Writes `pca_scores.csv`, `pca_loadings.csv`, and `pca_score_plot.png`.

## Command line

```bash
# Build the feature matrix only
python -m analysis.run_analysis --output-folder output/sorghum_run \
    --save-features analysis_out/features.csv

# Sorghum case study
python -m analysis.run_analysis --output-folder output/sorghum_run \
    --case-study sorghum --design sorghum_design.csv --control-label NT \
    --results-dir analysis_out/sorghum

# Maize case study
python -m analysis.run_analysis --features maize_features.csv \
    --case-study maize --design maize_design.csv \
    --results-dir analysis_out/maize
```

---

## Notes on fidelity to the paper

* `reduce_vegetation_index_features` reproduces the exact 384 → 240 reduction
  in Section 2.7 (48 indices × the 8 statistics emitted by
  `features/vegetation.py`: `mean, std, min, max, median, q25, q75,
  nan_fraction` → drop `min`, `max`, `nan_fraction`, then drop zero-variance
  and >50%-missing columns).
* `group_level_lmm` fits `value ~ C(group, Treatment(control)) + imaging_day`
  with plant identity as a random intercept via `statsmodels.formula.api.mixedlm`,
  matching the model specification in Section 2.7, and applies
  Benjamini–Hochberg correction *within each treatment group* across features.
* `rank_individuals_by_divergence` combines a control-mean t-test with a robust
  median/MAD z-test per plant and feature, matching the two complementary tests
  described for the individual-level analysis (Plant47 / Plant32 example).

These modules implement the *methodology* as specified in the paper. They were
not fit to the original study data, so re-running them on your own pipeline
output will not exactly reproduce the paper's published numbers (e.g. the
specific plant/group counts in Section 3.8.1) unless run on the same dataset
and design table used there.
