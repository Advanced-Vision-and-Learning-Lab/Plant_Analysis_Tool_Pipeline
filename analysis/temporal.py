"""
Temporal trajectory helpers and plots.

Implements the longitudinal summaries shown in the paper's figures:

* per-plant feature trajectories over imaging date (Figure 8a, NDVI example);
* group-mean difference from the non-treated control over time, with a
  variability band (Figure 3h / Figure 10a);

Plot helpers use matplotlib with the non-interactive ``Agg`` backend, matching
the rest of the pipeline's output code.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plant_trajectory(
    matrix: pd.DataFrame,
    plant_id: str,
    feature: str,
    id_col: str = "plant_id",
    date_col: str = "date",
) -> pd.Series:
    """Return one plant's trajectory for a single feature, indexed by date."""
    flat = matrix.reset_index()
    sub = flat[flat[id_col] == plant_id].sort_values(date_col)
    return pd.Series(sub[feature].to_numpy(), index=pd.to_datetime(sub[date_col]), name=feature)


def mean_difference_from_control(
    matrix: pd.DataFrame,
    design: pd.DataFrame,
    feature: str,
    control_label: str = "NT",
    id_col: str = "plant_id",
    date_col: str = "date",
    group_col: str = "group",
) -> pd.DataFrame:
    """Per-date, per-group mean difference from the control mean.

    Reproduces the summary behind Figure 3h / Figure 10a ("GOSAVI mean
    difference from the non-treated (NT) control over time"): for each imaging
    date and treatment group, the group mean is subtracted from the concurrent
    control-group mean, together with the standard error of the group mean for
    a variability band.
    """
    flat = matrix.reset_index().merge(
        design[[id_col, group_col]].drop_duplicates(), on=id_col, how="left"
    )
    flat[date_col] = pd.to_datetime(flat[date_col])

    ctrl = (
        flat[flat[group_col] == control_label]
        .groupby(date_col)[feature]
        .mean()
        .rename("control_mean")
    )

    grp = flat[flat[group_col] != control_label].groupby([group_col, date_col])[feature]
    stats = grp.agg(group_mean="mean", group_sem=lambda s: s.std(ddof=1) / np.sqrt(max(len(s), 1)))
    stats = stats.join(ctrl, on=date_col)
    stats["mean_diff_from_control"] = stats["group_mean"] - stats["control_mean"]
    return stats.reset_index()


def plot_trajectory(
    series: pd.Series,
    feature_name: str,
    save_path: Optional[str] = None,
    highlight_label: Optional[str] = None,
):
    """Line plot of a single trajectory (Figure 8a style)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(series.index, series.values, marker="o", color="#3b6fa0")
    ax.set_xlabel("Date")
    ax.set_ylabel(feature_name)
    if highlight_label:
        ax.set_title(highlight_label)
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved trajectory plot to %s", save_path)
    plt.close(fig)
    return fig


def plot_mean_difference(
    diff_df: pd.DataFrame,
    feature: str,
    group_col: str = "group",
    date_col: str = "date",
    save_path: Optional[str] = None,
):
    """Mean-difference-from-control plot with per-group SEM bands (Figure 10a style)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, gdf in diff_df.groupby(group_col):
        gdf = gdf.sort_values(date_col)
        ax.plot(gdf[date_col], gdf["mean_diff_from_control"], label=f"{group} - NT")
        ax.fill_between(
            gdf[date_col],
            gdf["mean_diff_from_control"] - gdf["group_sem"],
            gdf["mean_diff_from_control"] + gdf["group_sem"],
            alpha=0.15,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"{feature} (difference from NT)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved mean-difference plot to %s", save_path)
    plt.close(fig)
    return fig
