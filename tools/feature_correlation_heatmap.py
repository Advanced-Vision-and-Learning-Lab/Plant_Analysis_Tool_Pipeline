"""
Create correlation heatmap image(s) from pipeline-saved feature statistics.

This script scans the pipeline output folder for per-plant JSON statistics saved by
`sorghum_pipeline/output/manager.py`:

- texture/*/texture_statistics.json
- vegetation_indices/vegetation_statistics.json
- morphology/traits.json

It flattens all numeric values into a single feature table (rows = plants, columns = features),
computes a correlation matrix, and saves a correlation heatmap as a PNG.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires pandas to run.") from e

import matplotlib

# Use a non-GUI backend (safe for headless servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ScanConfig:
    include_texture: bool = True
    include_vegetation: bool = True
    include_morphology: bool = True


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _is_number(x: Any) -> bool:
    # bool is a subclass of int; treat it as non-numeric for features.
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def _flatten_numeric(d: Any, prefix: str) -> Dict[str, float]:
    """
    Flatten nested dicts and keep only finite numeric leaves.
    Keys are joined with '.'.
    """
    out: Dict[str, float] = {}

    if isinstance(d, dict):
        for k, v in d.items():
            k_str = str(k)
            child_prefix = f"{prefix}.{k_str}" if prefix else k_str
            out.update(_flatten_numeric(v, child_prefix))
        return out

    if _is_number(d):
        if prefix:
            out[prefix] = float(d)
        return out

    return out


def _discover_plant_dirs(output_root: Path) -> List[Path]:
    # Expected layout: output_root/YYYY_MM_DD/plantX/...
    # We'll treat any directory that contains one of the expected JSON files as a plant dir.
    candidates: List[Path] = []
    for p in output_root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "texture").exists() or (p / "vegetation_indices").exists() or (p / "morphology").exists():
            candidates.append(p)
    # Deduplicate and keep only likely leaf plant directories (heuristic: contains metadata.json OR subfolders)
    uniq = sorted(set(candidates))
    return uniq


def _extract_features_from_plant_dir(plant_dir: Path, scan: ScanConfig) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Returns:
      - features: flattened numeric feature dict
      - meta: useful identifiers (strings)
    """
    features: Dict[str, float] = {}

    # Basic metadata from path if possible: .../<date>/<plant>/
    meta: Dict[str, str] = {}
    try:
        meta["plant_dir"] = str(plant_dir)
        meta["plant"] = plant_dir.name
        meta["date"] = plant_dir.parent.name
    except Exception:
        pass

    # Texture stats: output_root/date/plant/texture/<band>/texture_statistics.json
    if scan.include_texture:
        texture_root = plant_dir / "texture"
        if texture_root.exists():
            for band_dir in texture_root.iterdir():
                if not band_dir.is_dir():
                    continue
                stats_path = band_dir / "texture_statistics.json"
                stats = _load_json(stats_path)
                if not isinstance(stats, dict):
                    continue
                band = band_dir.name
                flat = _flatten_numeric(stats, prefix=f"texture.{band}")
                features.update(flat)

    # Vegetation stats: output_root/date/plant/vegetation_indices/vegetation_statistics.json
    if scan.include_vegetation:
        veg_path = plant_dir / "vegetation_indices" / "vegetation_statistics.json"
        veg = _load_json(veg_path)
        if isinstance(veg, dict):
            flat = _flatten_numeric(veg, prefix="vegetation")
            features.update(flat)

    # Morphology traits: output_root/date/plant/morphology/traits.json
    if scan.include_morphology:
        traits_path = plant_dir / "morphology" / "traits.json"
        traits = _load_json(traits_path)
        if isinstance(traits, dict):
            flat = _flatten_numeric(traits, prefix="morphology")
            features.update(flat)

    return features, meta


def _build_feature_table(output_root: Path, scan: ScanConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plant_dirs = _discover_plant_dirs(output_root)
    rows: List[Dict[str, Any]] = []
    metas: List[Dict[str, str]] = []

    for plant_dir in plant_dirs:
        feats, meta = _extract_features_from_plant_dir(plant_dir, scan)
        # Keep rows only if we have at least one feature
        if feats:
            rows.append(feats)
            metas.append(meta)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metas)
    # Create a stable index when possible
    if {"date", "plant"}.issubset(meta_df.columns):
        X.index = meta_df["date"].astype(str) + "/" + meta_df["plant"].astype(str)
        meta_df.index = X.index

    return X, meta_df


def _select_features(X: pd.DataFrame, max_features: int) -> pd.DataFrame:
    """
    If there are too many features, keep a compact subset that is:
    - present for many samples (high non-null count)
    - non-constant (variance > 0)
    """
    if X.empty:
        return X

    # Drop all-NaN columns
    X = X.dropna(axis=1, how="all")
    if X.empty:
        return X

    # Keep only non-constant columns (variance computed ignoring NaNs)
    variances = X.var(axis=0, skipna=True)
    X = X.loc[:, variances.fillna(0.0) > 0.0]
    if X.empty:
        return X

    if X.shape[1] <= max_features:
        return X

    non_null = X.notna().sum(axis=0)
    score = (non_null / max(1, X.shape[0])) * np.sqrt(variances.loc[X.columns].fillna(0.0) + 1e-12)
    keep = score.sort_values(ascending=False).head(max_features).index
    return X.loc[:, keep]


def _plot_corr_heatmap(
    corr: pd.DataFrame,
    out_png: Path,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    n = corr.shape[0]
    if n == 0:
        return

    # Dynamic sizing; cap to avoid absurd images
    cell = 0.28
    fig_w = min(40.0, max(10.0, n * cell))
    fig_h = min(34.0, max(8.0, n * cell))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns.tolist(), fontsize=6, rotation=90)
    ax.set_yticklabels(corr.index.tolist(), fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=90)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a feature correlation heatmap PNG from pipeline outputs.")
    ap.add_argument("--output-root", required=True, help="Pipeline output folder (the one containing date/plant dirs).")
    ap.add_argument("--out-dir", default=None, help="Directory to write outputs (default: <output-root>/analysis).")
    ap.add_argument("--method", default="pearson", choices=["pearson", "spearman"], help="Correlation method.")
    ap.add_argument(
        "--max-features",
        type=int,
        default=80,
        help="Limit number of features in the heatmap for readability (default: 80).",
    )
    ap.add_argument("--no-texture", action="store_true", help="Exclude texture features.")
    ap.add_argument("--no-vegetation", action="store_true", help="Exclude vegetation features.")
    ap.add_argument("--no-morphology", action="store_true", help="Exclude morphology traits.")

    args = ap.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (output_root / "analysis")

    scan = ScanConfig(
        include_texture=not args.no_texture,
        include_vegetation=not args.no_vegetation,
        include_morphology=not args.no_morphology,
    )

    X, meta_df = _build_feature_table(output_root, scan)
    if X.empty:
        print("No feature JSONs found under output root. Nothing to correlate.")
        return 2

    X_sel = _select_features(X, max_features=int(args.max_features))
    if X_sel.empty or X_sel.shape[1] < 2:
        print("Not enough non-empty, non-constant numeric features to compute correlation.")
        return 3

    # Correlation (pairwise complete observations)
    corr = X_sel.corr(method=args.method)

    # Save tables
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "feature_table_all.csv", index=True)
    X_sel.to_csv(out_dir / "feature_table_selected.csv", index=True)
    corr.to_csv(out_dir / f"feature_correlations_{args.method}.csv", index=True)
    if not meta_df.empty:
        meta_df.to_csv(out_dir / "feature_table_meta.csv", index=True)

    # Plot image
    out_png = out_dir / f"feature_correlations_{args.method}.png"
    title = f"Feature correlation ({args.method}) | n_samples={X.shape[0]} | n_features={X_sel.shape[1]}"
    _plot_corr_heatmap(corr, out_png, title=title)

    print(f"Saved correlation heatmap to: {out_png}")
    print(f"Saved tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

