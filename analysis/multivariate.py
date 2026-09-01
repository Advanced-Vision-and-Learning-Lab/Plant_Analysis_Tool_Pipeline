"""
Multivariate analysis of the extracted phenotypic feature space.

Implements the unsupervised and supervised multivariate methods referenced in
the paper:

* principal component analysis (PCA) of the full high-dimensional feature set
  (Sections 3.5 and 3.8.2, Figures 8b and 11b);
* linear discriminant analysis (LDA) for supervised multivariate visualization
  (Figure 3h);
* hierarchical clustering for unsupervised grouping (Figure 3h).

All routines take a plain feature matrix (``n_samples x n_features``) plus the
list of feature names, and standardize features by default.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _prepare(matrix: pd.DataFrame, feature_cols: Sequence[str], standardize: bool):
    X = matrix[list(feature_cols)].to_numpy(dtype=np.float64)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])
    if standardize:
        X = StandardScaler().fit_transform(X)
    return X


@dataclass
class PCAResult:
    scores: pd.DataFrame
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame = field(repr=False)
    model: PCA = field(repr=False)

    def top_loadings(self, component: int = 0, n: int = 10) -> pd.Series:
        col = self.loadings.columns[component]
        return self.loadings[col].abs().sort_values(ascending=False).head(n)


def run_pca(
    matrix: pd.DataFrame,
    feature_cols: Sequence[str],
    n_components: int = 2,
    standardize: bool = True,
    whiten: bool = False,
) -> PCAResult:
    """PCA on the phenotyping feature matrix.

    Mirrors the analysis behind Figures 8b and 11b: features are standardized,
    then projected onto the leading components. Returns component scores per
    sample, the explained-variance ratio, and the feature loadings.
    """
    X = _prepare(matrix, feature_cols, standardize)
    n_components = int(min(n_components, X.shape[0], X.shape[1]))
    model = PCA(n_components=n_components, whiten=whiten, random_state=0)
    scores = model.fit_transform(X)

    comp_names = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=comp_names, index=matrix.index)
    loadings_df = pd.DataFrame(
        model.components_.T, index=list(feature_cols), columns=comp_names
    )
    logger.info(
        "PCA: %d components explain %.1f%% of variance",
        n_components, 100.0 * model.explained_variance_ratio_.sum(),
    )
    return PCAResult(scores_df, model.explained_variance_ratio_, loadings_df, model)


@dataclass
class LDAResult:
    scores: pd.DataFrame
    explained_variance_ratio: np.ndarray
    model: LinearDiscriminantAnalysis = field(repr=False)


def run_lda(
    matrix: pd.DataFrame,
    labels: Sequence,
    feature_cols: Sequence[str],
    n_components: Optional[int] = None,
    standardize: bool = True,
) -> LDAResult:
    """Supervised LDA projection for multivariate visualization (Figure 3h)."""
    X = _prepare(matrix, feature_cols, standardize)
    y = np.asarray(labels)
    max_comp = min(len(np.unique(y)) - 1, X.shape[1])
    n_components = max_comp if n_components is None else int(min(n_components, max_comp))

    model = LinearDiscriminantAnalysis(n_components=n_components)
    scores = model.fit_transform(X, y)
    comp_names = [f"LD{i + 1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=comp_names, index=matrix.index)
    scores_df["label"] = y
    evr = getattr(model, "explained_variance_ratio_", np.array([]))
    return LDAResult(scores_df, evr, model)


@dataclass
class ClusteringResult:
    linkage_matrix: np.ndarray = field(repr=False)
    labels: pd.Series
    n_clusters: int


def hierarchical_clustering(
    matrix: pd.DataFrame,
    feature_cols: Sequence[str],
    method: str = "ward",
    metric: str = "euclidean",
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    standardize: bool = True,
) -> ClusteringResult:
    """Agglomerative hierarchical clustering for unsupervised grouping (Figure 3h).

    Returns the SciPy linkage matrix (for dendrogram rendering) and a flat
    cluster assignment per sample. Provide either ``n_clusters`` or
    ``distance_threshold``.
    """
    X = _prepare(matrix, feature_cols, standardize)
    Z = linkage(X, method=method, metric=metric)

    if n_clusters is not None:
        flat = fcluster(Z, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        flat = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        flat = fcluster(Z, t=2, criterion="maxclust")

    labels = pd.Series(flat, index=matrix.index, name="cluster")
    return ClusteringResult(Z, labels, int(labels.nunique()))
