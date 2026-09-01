"""
Treatment-level and individual-level statistical analysis.

Implements the tests described in Sections 2.6 and 2.7 of the paper:

* control-mean t-tests and robust median/MAD z-tests of each plant against the
  non-treated control, ranked by divergence using temporal-mean and
  time-normalized area-under-the-curve summaries;
* group-level linear mixed-effects models (LMM) with treatment group and linear
  imaging day as fixed effects and plant identity as a random intercept;
* Benjamini-Hochberg false-discovery-rate correction applied across features.

All routines operate on the tidy temporal feature matrix produced by
``analysis.feature_matrix``.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:  # statsmodels is only needed for the group-level LMM
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    _HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAVE_STATSMODELS = False

logger = logging.getLogger(__name__)

_MAD_TO_SIGMA = 1.4826  # consistency constant for the normal distribution


# --------------------------------------------------------------------------- #
# Multiple-testing correction
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995).

    Falls back to a NumPy implementation when statsmodels is unavailable so the
    individual-level analysis still runs without the optional dependency.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    finite = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    p_adj = np.full(p.shape, np.nan)

    if finite.sum() == 0:
        return {"reject": reject, "pvalue_adj": p_adj}

    if _HAVE_STATSMODELS:
        rej, adj, _, _ = multipletests(p[finite], alpha=alpha, method="fdr_bh")
        reject[finite] = rej
        p_adj[finite] = adj
        return {"reject": reject, "pvalue_adj": p_adj}

    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    m = ranked.size
    adj_sorted = ranked * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adj = np.empty(m)
    adj[order] = adj_sorted
    p_adj[finite] = adj
    reject[finite] = adj <= alpha
    return {"reject": reject, "pvalue_adj": p_adj}


# --------------------------------------------------------------------------- #
# Individual-level divergence from the non-treated control
# --------------------------------------------------------------------------- #
def time_normalized_auc(times: Sequence[float], values: Sequence[float]) -> float:
    """Trapezoidal AUC over time rescaled to a unit interval.

    Used as a per-feature summary of an individual plant's trajectory when
    ranking divergence from the control (Section 2.7).
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < 2:
        return float("nan")
    order = np.argsort(t)
    t, v = t[order], v[order]
    span = t[-1] - t[0]
    if span <= 0:
        return float(np.mean(v))
    return float(np.trapz(v, x=(t - t[0]) / span))


def control_mean_ttest(plant_values: Sequence[float], control_mean: float) -> Dict[str, float]:
    """One-sample t-test of a plant's repeated measurements against the control mean."""
    v = np.asarray(plant_values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2 or not np.isfinite(control_mean):
        return {"t": float("nan"), "pvalue": float("nan"), "effect_size": float("nan")}
    res = stats.ttest_1samp(v, popmean=control_mean)
    sd = np.std(v, ddof=1)
    effect = (np.mean(v) - control_mean) / sd if sd > 0 else 0.0
    return {"t": float(res.statistic), "pvalue": float(res.pvalue), "effect_size": float(effect)}


def robust_z(plant_value: float, control_values: Sequence[float]) -> float:
    """Robust median/MAD z-score of a plant summary against the control sample."""
    c = np.asarray(control_values, dtype=np.float64)
    c = c[np.isfinite(c)]
    if c.size == 0 or not np.isfinite(plant_value):
        return float("nan")
    med = np.median(c)
    mad = np.median(np.abs(c - med))
    if mad <= 0:
        sd = np.std(c)
        return float((plant_value - med) / sd) if sd > 0 else 0.0
    return float((plant_value - med) / (_MAD_TO_SIGMA * mad))


@dataclass
class IndividualDivergence:
    """Per-plant divergence result across the retained feature set."""

    plant_id: str
    group: str
    n_raw_significant: int
    n_fdr_significant: int
    mean_abs_effect_size: float
    mean_abs_robust_z: float
    per_feature: pd.DataFrame = field(repr=False)


def rank_individuals_by_divergence(
    matrix: pd.DataFrame,
    design: pd.DataFrame,
    feature_cols: Sequence[str],
    control_label: str = "NT",
    id_col: str = "plant_id",
    date_col: str = "date",
    group_col: str = "group",
    alpha: float = 0.05,
) -> List[IndividualDivergence]:
    """Rank individual plants by divergence from the non-treated control.

    For every plant and retained feature the routine computes:

    * a temporal-mean and a time-normalized AUC summary of the trajectory;
    * a control-mean t-test (plant timepoints vs. control mean);
    * a robust median/MAD z-test of the plant's temporal mean against the
      distribution of control-plant temporal means.

    Benjamini-Hochberg FDR correction is then applied across the feature set
    *for each plant*. Results are returned sorted by the number of
    FDR-significant features (descending), matching the reporting in
    Section 3.8.1.
    """
    flat = matrix.reset_index()
    flat = flat.merge(design[[id_col, group_col]].drop_duplicates(), on=id_col, how="left")

    times = pd.to_datetime(flat[date_col])
    flat["_t_days"] = (times - times.min()).dt.total_seconds() / 86400.0

    control_ids = design.loc[design[group_col] == control_label, id_col].unique()
    ctrl_rows = flat[flat[id_col].isin(control_ids)]

    # Control reference: per-feature mean over all control timepoints, and the
    # per-control-plant temporal means (for the robust z null distribution).
    ctrl_mean = ctrl_rows[list(feature_cols)].mean()
    ctrl_plant_means = ctrl_rows.groupby(id_col)[list(feature_cols)].mean()

    results: List[IndividualDivergence] = []
    for plant_id, pdf in flat.groupby(id_col):
        if plant_id in control_ids:
            continue
        group = pdf[group_col].iloc[0] if group_col in pdf else "NA"

        rows = []
        for feat in feature_cols:
            series = pdf[["_t_days", feat]].dropna()
            temporal_mean = float(series[feat].mean()) if not series.empty else float("nan")
            auc = time_normalized_auc(series["_t_days"], series[feat])
            tt = control_mean_ttest(series[feat], ctrl_mean.get(feat, float("nan")))
            rz = robust_z(temporal_mean, ctrl_plant_means[feat])
            rows.append({
                "feature": feat,
                "temporal_mean": temporal_mean,
                "auc": auc,
                "t": tt["t"],
                "pvalue": tt["pvalue"],
                "effect_size": tt["effect_size"],
                "robust_z": rz,
            })

        per_feat = pd.DataFrame(rows)
        bh = benjamini_hochberg(per_feat["pvalue"].to_numpy(), alpha=alpha)
        per_feat["pvalue_adj"] = bh["pvalue_adj"]
        per_feat["fdr_significant"] = bh["reject"]
        per_feat["raw_significant"] = per_feat["pvalue"] < alpha

        results.append(IndividualDivergence(
            plant_id=str(plant_id),
            group=str(group),
            n_raw_significant=int(per_feat["raw_significant"].sum()),
            n_fdr_significant=int(per_feat["fdr_significant"].sum()),
            mean_abs_effect_size=float(per_feat["effect_size"].abs().mean()),
            mean_abs_robust_z=float(per_feat["robust_z"].abs().mean()),
            per_feature=per_feat,
        ))

    results.sort(key=lambda r: (r.n_fdr_significant, r.n_raw_significant), reverse=True)
    return results


def individual_summary_table(results: Sequence[IndividualDivergence]) -> pd.DataFrame:
    """Flatten :func:`rank_individuals_by_divergence` output into one table."""
    return pd.DataFrame([
        {
            "plant_id": r.plant_id,
            "group": r.group,
            "raw_significant": r.n_raw_significant,
            "fdr_significant": r.n_fdr_significant,
            "mean_abs_effect_size": r.mean_abs_effect_size,
            "mean_abs_robust_z": r.mean_abs_robust_z,
        }
        for r in results
    ])


# --------------------------------------------------------------------------- #
# Group-level linear mixed-effects models
# --------------------------------------------------------------------------- #
def group_level_lmm(
    matrix: pd.DataFrame,
    design: pd.DataFrame,
    feature_cols: Sequence[str],
    control_label: str = "NT",
    id_col: str = "plant_id",
    date_col: str = "date",
    group_col: str = "group",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit one LMM per feature and return treatment-vs-control contrasts.

    Model (Section 2.7), fit independently for each retained feature::

        value ~ C(group, Treatment(control)) + imaging_day
        random intercept: plant identity

    The treatment-group fixed-effect coefficients are the treatment-versus-
    control contrasts. p-values are corrected with Benjamini-Hochberg *within
    each treatment group* across the feature set. The returned frame has one row
    per (treatment group, feature).
    """
    if not _HAVE_STATSMODELS:
        raise ImportError(
            "group_level_lmm requires statsmodels (pip install statsmodels==0.14.2)"
        )

    flat = matrix.reset_index().merge(
        design[[id_col, group_col]].drop_duplicates(), on=id_col, how="left"
    )
    times = pd.to_datetime(flat[date_col])
    flat["imaging_day"] = (times - times.min()).dt.total_seconds() / 86400.0
    flat[group_col] = flat[group_col].astype("category")

    treatment_groups = [g for g in flat[group_col].cat.categories if g != control_label]

    records: List[dict] = []
    for feat in feature_cols:
        sub = flat[[id_col, group_col, "imaging_day", feat]].dropna()
        sub = sub.rename(columns={feat: "value"})
        if sub["group"].nunique() < 2 or sub["value"].nunique() <= 1:
            continue
        formula = f"value ~ C({group_col}, Treatment(reference={control_label!r})) + imaging_day"
        try:
            model = smf.mixedlm(formula, data=sub, groups=sub[id_col])
            fit = model.fit(method="lbfgs", reml=True, disp=False)
        except Exception as exc:  # singular fits, non-convergence, ...
            logger.debug("LMM failed for %s: %s", feat, exc)
            continue

        for g in treatment_groups:
            key = [k for k in fit.params.index if f"T.{g}]" in k]
            if not key:
                continue
            name = key[0]
            records.append({
                "group": g,
                "feature": feat,
                "coef": float(fit.params[name]),
                "std_err": float(fit.bse[name]),
                "pvalue": float(fit.pvalues[name]),
            })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    out["pvalue_adj"] = np.nan
    out["fdr_significant"] = False
    out["raw_significant"] = out["pvalue"] < alpha
    for g, idx in out.groupby("group").groups.items():
        bh = benjamini_hochberg(out.loc[idx, "pvalue"].to_numpy(), alpha=alpha)
        out.loc[idx, "pvalue_adj"] = bh["pvalue_adj"]
        out.loc[idx, "fdr_significant"] = bh["reject"]
    return out.sort_values(["group", "pvalue"]).reset_index(drop=True)


def group_significance_counts(lmm_results: pd.DataFrame) -> pd.DataFrame:
    """Per-group counts of raw- and FDR-significant features (Figure 10b)."""
    if lmm_results.empty:
        return lmm_results
    return (
        lmm_results.groupby("group")
        .agg(raw_significant=("raw_significant", "sum"),
             fdr_significant=("fdr_significant", "sum"),
             n_features=("feature", "count"))
        .reset_index()
        .sort_values("fdr_significant", ascending=False)
    )
