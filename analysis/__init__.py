"""
Downstream temporal and statistical analysis (Section 2.6, 2.7 of the paper).

This package picks up where the image-processing pipeline (``pipeline.py``)
leaves off: it takes per-plant, per-date feature outputs and performs the
temporal aggregation, statistical testing, and multivariate analysis reported
in the Results (Section 3.5, 3.8) and used to generate Figures 3h, 8, 10, and
11.

Typical usage::

    from analysis import collect_feature_matrix, build_temporal_matrix
    from analysis import run_sorghum_treatment_analysis, run_maize_coldstress_analysis

    features = collect_feature_matrix("output/sorghum_run")
    design = pd.read_csv("sorghum_treatment_design.csv")  # plant_id -> group
    results = run_sorghum_treatment_analysis(features, design, output_dir="analysis_out")

See ``analysis/README.md`` for the expected input tables and a full example,
and ``analysis/run_analysis.py`` for a command-line entry point.
"""

from .collect import collect_feature_matrix
from .feature_matrix import (
    VEGETATION_DROP_STATS,
    VEGETATION_SUMMARY_STATS,
    build_temporal_matrix,
    distribution_descriptors,
    long_form,
    reduce_vegetation_index_features,
)
from .statistics import (
    IndividualDivergence,
    benjamini_hochberg,
    control_mean_ttest,
    group_level_lmm,
    group_significance_counts,
    individual_summary_table,
    rank_individuals_by_divergence,
    robust_z,
    time_normalized_auc,
)
from .multivariate import (
    ClusteringResult,
    LDAResult,
    PCAResult,
    hierarchical_clustering,
    run_lda,
    run_pca,
)
from .temporal import (
    mean_difference_from_control,
    plant_trajectory,
    plot_mean_difference,
    plot_trajectory,
)
from .case_studies import (
    plot_control_relative_trajectory,
    run_maize_coldstress_analysis,
    run_sorghum_treatment_analysis,
)

__all__ = [
    "collect_feature_matrix",
    "VEGETATION_DROP_STATS",
    "VEGETATION_SUMMARY_STATS",
    "build_temporal_matrix",
    "distribution_descriptors",
    "long_form",
    "reduce_vegetation_index_features",
    "IndividualDivergence",
    "benjamini_hochberg",
    "control_mean_ttest",
    "group_level_lmm",
    "group_significance_counts",
    "individual_summary_table",
    "rank_individuals_by_divergence",
    "robust_z",
    "time_normalized_auc",
    "ClusteringResult",
    "LDAResult",
    "PCAResult",
    "hierarchical_clustering",
    "run_lda",
    "run_pca",
    "mean_difference_from_control",
    "plant_trajectory",
    "plot_mean_difference",
    "plot_trajectory",
    "plot_control_relative_trajectory",
    "run_maize_coldstress_analysis",
    "run_sorghum_treatment_analysis",
]
