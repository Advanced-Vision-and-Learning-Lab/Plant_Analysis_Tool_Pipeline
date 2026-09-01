"""
Temporal feature-matrix construction and per-distribution descriptors.

Implements the aggregation described in Section 2.6 ("Temporal and Statistical
Analysis") of the paper: features extracted from segmented plant regions are
aggregated by plant identity and imaging date to build temporal feature
matrices, and pixel-level reflectance distributions are summarized with
central-tendency and higher-order descriptors.

This module is intentionally independent of the image-processing pipeline: it
consumes a tidy feature table (one row per plant x imaging date, one column per
feature) such as the one produced by ``analysis.collect.collect_feature_matrix``.
"""

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Eight per-index summary statistics emitted by ``features/vegetation.py``
# (mean, std, min, max, median, 25th pct, 75th pct, undefined-pixel fraction).
VEGETATION_SUMMARY_STATS: List[str] = [
    "mean", "std", "min", "max", "median", "q25", "q75", "nan_fraction",
]

# Statistics removed during the 384 -> 240 vegetation-index reduction
# (Section 2.7): the two per-index extrema and the undefined-pixel fraction.
VEGETATION_DROP_STATS: List[str] = ["min", "max", "nan_fraction"]


def distribution_descriptors(values: Sequence[float]) -> Dict[str, float]:
    """Summary descriptors for a single pixel-level distribution.

    Returns the descriptors listed in Section 2.6: mean, standard deviation,
    minimum, maximum, median, interquartile range (75th - 25th percentile),
    skewness, kurtosis, and Shannon entropy. These capture both central
    tendency and higher-order variability in the spectral response within a
    plant region.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        keys = [
            "mean", "std", "min", "max", "median", "iqr",
            "skewness", "kurtosis", "shannon_entropy",
        ]
        return {k: float("nan") for k in keys}

    q25, q75 = np.percentile(arr, [25, 75])
    # Shannon entropy of the intensity histogram (nats).
    hist, _ = np.histogram(arr, bins=256)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    shannon_entropy = float(-(p * np.log(p)).sum())

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "skewness": float(stats.skew(arr, bias=False)) if arr.size > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(arr, bias=False)) if arr.size > 3 else 0.0,
        "shannon_entropy": shannon_entropy,
    }


def build_temporal_matrix(
    df: pd.DataFrame,
    id_col: str = "plant_id",
    date_col: str = "date",
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a sorted temporal feature matrix indexed by (plant, date).

    Rows are (plant identity, imaging date) pairs; columns are image-derived
    features. Plant identity is assumed to have been established upstream by the
    SAM2Long temporal tracker (Section 2.4.2). Duplicate (plant, date) rows —
    e.g. multiple frames of the same plant on the same day — are averaged.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if work[date_col].isna().any():
        logger.warning("Dropping %d rows with unparseable dates", int(work[date_col].isna().sum()))
        work = work.dropna(subset=[date_col])

    if feature_cols is None:
        feature_cols = [
            c for c in work.columns
            if c not in {id_col, date_col} and pd.api.types.is_numeric_dtype(work[c])
        ]
    feature_cols = list(feature_cols)

    matrix = (
        work.groupby([id_col, date_col], as_index=True)[feature_cols]
        .mean()
        .sort_index()
    )
    return matrix


def reduce_vegetation_index_features(
    df: pd.DataFrame,
    id_cols: Sequence[str] = ("plant_id", "date"),
    missing_threshold: float = 0.50,
) -> pd.DataFrame:
    """Apply the 384 -> 240 vegetation-index feature reduction of Section 2.7.

    Starting from ``48 indices x 8 summary statistics`` columns named
    ``<INDEX>__<stat>`` (or ``veg__<INDEX>__<stat>``), this:

    1. removes the two per-index extrema (``min``, ``max``);
    2. removes the per-index undefined-pixel fraction (``nan_fraction``);
    3. removes any remaining zero-variance features;
    4. drops features whose missing-value fraction exceeds ``missing_threshold``
       (no feature exceeded 50% in the paper).
    """
    id_cols = [c for c in id_cols if c in df.columns]
    feat = df.drop(columns=id_cols)

    def _stat_of(col: str) -> str:
        return col.rsplit("__", 1)[-1]

    keep = [c for c in feat.columns if _stat_of(c) not in VEGETATION_DROP_STATS]
    dropped_extrema = [c for c in feat.columns if c not in keep]
    feat = feat[keep]

    # Missing-value threshold.
    miss_frac = feat.isna().mean()
    over_thresh = miss_frac[miss_frac > missing_threshold].index.tolist()
    feat = feat.drop(columns=over_thresh)

    # Zero-variance removal (after filling for the variance check only).
    variances = feat.var(numeric_only=True, skipna=True).fillna(0.0)
    zero_var = variances[variances <= 0.0].index.tolist()
    feat = feat.drop(columns=zero_var)

    logger.info(
        "Vegetation-index reduction: %d -> %d features "
        "(dropped %d extrema/undefined, %d over missing threshold, %d zero-variance)",
        df.shape[1] - len(id_cols), feat.shape[1],
        len(dropped_extrema), len(over_thresh), len(zero_var),
    )
    return pd.concat([df[id_cols].reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


def long_form(
    matrix: pd.DataFrame,
    id_col: str = "plant_id",
    date_col: str = "date",
    value_name: str = "value",
    var_name: str = "feature",
) -> pd.DataFrame:
    """Melt a (plant, date) x feature matrix into long form for LMM fitting."""
    flat = matrix.reset_index()
    return flat.melt(id_vars=[id_col, date_col], var_name=var_name, value_name=value_name)
