"""
Case-study drivers.

Wires together ``analysis.feature_matrix``, ``analysis.statistics``, and
``analysis.multivariate`` to reproduce the two analyses described in Sections
2.7 and 3.8 of the paper:

* :func:`run_sorghum_treatment_analysis` — Case Study 1, the LEEB-mutagenized
  sorghum treatment-level comparison (Section 3.8.1, Figure 10);
* :func:`run_maize_coldstress_analysis` — Case Study 2, the CERCA maize cold-
  stress PCA (Section 3.8.2, Figure 11).

Both take the tidy feature matrix produced by ``analysis.collect`` (or your own
equivalent table) plus a "design" table describing the experimental groups, and
write result CSVs / figures to ``output_dir`` when provided.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    import matplotlib.pyplot as plt

from .feature_matrix import build_temporal_matrix, reduce_vegetation_index_features
from .multivariate import run_pca
from .statistics import (
    group_level_lmm,
    group_significance_counts,
    individual_summary_table,
    rank_individuals_by_divergence,
)
from .temporal import mean_difference_from_control, plot_mean_difference

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Case Study 1 — mutagenized sorghum treatment-level comparison
# --------------------------------------------------------------------------- #
def run_sorghum_treatment_analysis(
    feature_df: pd.DataFrame,
    design: pd.DataFrame,
    output_dir: Optional[str] = None,
    control_label: str = "NT",
    alpha: float = 0.05,
    id_col: str = "plant_id",
    date_col: str = "date",
    group_col: str = "group",
) -> Dict[str, object]:
    """Reproduce the Case Study 1 workflow (Section 2.7 / 3.8.1).

    ``feature_df`` must contain ``plant_id`` / ``date`` columns plus the 48
    vegetation-index x 8-statistic columns produced by
    ``analysis.collect.collect_feature_matrix`` (named ``veg__<INDEX>__<stat>``).
    ``design`` must map ``plant_id`` -> ``group`` (``G1``...``G7`` and the
    control group, e.g. ``NT``).

    Returns a dict with the reduced feature matrix, the individual-level
    divergence ranking, and the group-level LMM contrasts — the same two
    complementary analyses reported in the paper (Plant47 / Group2 example).
    """
    veg_cols = [c for c in feature_df.columns if c.startswith("veg__")]
    if not veg_cols:
        raise ValueError("feature_df has no 'veg__*' vegetation-index columns")

    reduced = reduce_vegetation_index_features(feature_df[[id_col, date_col] + veg_cols])
    feature_cols = [c for c in reduced.columns if c not in (id_col, date_col)]
    matrix = build_temporal_matrix(reduced, id_col=id_col, date_col=date_col, feature_cols=feature_cols)

    individual_results = rank_individuals_by_divergence(
        matrix, design, feature_cols,
        control_label=control_label, id_col=id_col, date_col=date_col,
        group_col=group_col, alpha=alpha,
    )
    individual_table = individual_summary_table(individual_results)

    lmm_results = group_level_lmm(
        matrix, design, feature_cols,
        control_label=control_label, id_col=id_col, date_col=date_col,
        group_col=group_col, alpha=alpha,
    )
    group_counts = group_significance_counts(lmm_results)

    logger.info(
        "Sorghum case study: %d retained vegetation-index features, "
        "%d plants ranked, %d group-level contrasts fit",
        len(feature_cols), len(individual_results), len(lmm_results),
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        individual_table.to_csv(out / "individual_divergence_summary.csv", index=False)
        lmm_results.to_csv(out / "group_lmm_contrasts.csv", index=False)
        group_counts.to_csv(out / "group_significance_counts.csv", index=False)
        _plot_group_significance(group_counts, save_path=out / "group_level_feature_significance.png")

    return {
        "feature_matrix": matrix,
        "feature_cols": feature_cols,
        "individual_results": individual_results,
        "individual_table": individual_table,
        "lmm_results": lmm_results,
        "group_counts": group_counts,
    }


def _plot_group_significance(group_counts: pd.DataFrame, save_path=None):
    """Bar chart of raw- vs. FDR-significant features per group (Figure 10b style)."""
    if group_counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(group_counts))
    width = 0.35
    ax.bar(x - width / 2, group_counts["raw_significant"], width, label="raw p<0.05", color="#7fb2e5")
    ax.bar(x + width / 2, group_counts["fdr_significant"], width, label="FDR-BH", color="#c1443c")
    ax.set_xticks(x)
    ax.set_xticklabels(group_counts["group"])
    ax.set_ylabel("VI features differing from control")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved group-significance plot to %s", save_path)
    plt.close(fig)
    return fig


def plot_control_relative_trajectory(
    matrix: pd.DataFrame,
    design: pd.DataFrame,
    feature: str,
    control_label: str = "NT",
    output_dir: Optional[str] = None,
    **kwargs,
):
    """Convenience wrapper around ``analysis.temporal`` for a single feature
    (e.g. NDVI mean difference from NT, Figure 10a)."""
    diff = mean_difference_from_control(matrix, design, feature, control_label=control_label, **kwargs)
    save_path = None
    if output_dir:
        save_path = Path(output_dir) / f"{feature}_mean_diff_from_{control_label}.png"
    plot_mean_difference(diff, feature, save_path=save_path)
    return diff


# --------------------------------------------------------------------------- #
# Case Study 2 — CERCA maize cold-stress PCA
# --------------------------------------------------------------------------- #
def run_maize_coldstress_analysis(
    feature_df: pd.DataFrame,
    design: pd.DataFrame,
    output_dir: Optional[str] = None,
    id_col: str = "image_id",
    genotype_col: str = "genotype",
    stage_col: str = "stage",
    n_components: int = 2,
    feature_prefix: str = "veg__",
) -> Dict[str, object]:
    """Reproduce the Case Study 2 workflow (Section 2.7 / 3.8.2).

    ``feature_df`` has one row per image (``id_col``) and the RGB-derived
    vegetation-index feature columns (default prefix ``veg__``, e.g. ExG,
    ExR, GLI, MGRVI, ExGR, NGRDI, VARI and their summary statistics).
    ``design`` maps ``image_id`` -> genotype and treatment stage (before /
    10C / 4C / recovery-10C / recovery-4C).

    Returns the PCA result plus a merged score table (genotype + stage) ready
    for the score-plot / scree-plot / loadings visualization in Figure 11b.
    """
    feature_cols = [c for c in feature_df.columns if c.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"feature_df has no columns starting with '{feature_prefix}'")

    matrix = feature_df.set_index(id_col)
    pca_result = run_pca(matrix, feature_cols, n_components=n_components, standardize=True)

    scores = pca_result.scores.merge(
        design.set_index(id_col)[[genotype_col, stage_col]], left_index=True, right_index=True, how="left"
    )

    logger.info(
        "Maize cold-stress PCA: PC1 %.1f%%, PC2 %.1f%% explained variance "
        "(n=%d images, %d features)",
        100 * pca_result.explained_variance_ratio[0],
        100 * pca_result.explained_variance_ratio[1] if len(pca_result.explained_variance_ratio) > 1 else 0.0,
        matrix.shape[0], len(feature_cols),
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "pca_scores.csv")
        pca_result.loadings.to_csv(out / "pca_loadings.csv")
        _plot_pca_score(scores, pca_result.explained_variance_ratio,
                         genotype_col, stage_col, save_path=out / "pca_score_plot.png")

    return {"pca": pca_result, "scores": scores, "feature_cols": feature_cols}


def _plot_pca_score(scores: pd.DataFrame, evr: np.ndarray, genotype_col: str,
                     stage_col: str, save_path=None):
    """PC1-vs-PC2 score plot colored by genotype, marker-shaped by stage (Figure 11b style)."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    stage_to_marker = {s: markers[i % len(markers)] for i, s in enumerate(sorted(scores[stage_col].dropna().unique()))}
    genotypes = sorted(scores[genotype_col].dropna().unique())
    cmap = plt.get_cmap("tab10")
    genotype_to_color = {g: cmap(i % 10) for i, g in enumerate(genotypes)}

    for (genotype, stage), gdf in scores.groupby([genotype_col, stage_col]):
        ax.scatter(
            gdf["PC1"], gdf["PC2"],
            color=genotype_to_color.get(genotype, "gray"),
            marker=stage_to_marker.get(stage, "o"),
            label=f"{genotype} / {stage}", s=40, edgecolor="black", linewidth=0.3,
        )

    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% variance)")
    if len(evr) > 1:
        ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% variance)")
    ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved PCA score plot to %s", save_path)
    plt.close(fig)
    return fig
