"""
Collect a tidy feature matrix from ``PlantPipeline`` output directories.

The image-processing pipeline (``pipeline.py`` / ``output/manager.py``) writes
one directory per plant and imaging date::

    output_folder/
    └── YYYY_MM_DD/
        └── plantN/
            ├── vegetation_indices/vegetation_statistics.json
            ├── texture/<band>/texture_statistics.json
            ├── morphology/traits.json
            └── metadata.json

This module walks that tree and flattens it into the single tidy table (one
row per plant x imaging date, one column per feature) expected by
``analysis.feature_matrix``, ``analysis.statistics``, and
``analysis.multivariate`` — the "863-dimensional CSV feature matrix" described
in Section 3.7 of the paper.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def _flatten_vegetation(plant_dir: Path) -> Dict[str, float]:
    path = plant_dir / "vegetation_indices" / "vegetation_statistics.json"
    data = _load_json(path)
    if not data:
        return {}
    out: Dict[str, float] = {}
    for index_name, stats in data.items():
        if not isinstance(stats, dict):
            continue
        for stat_name, value in stats.items():
            out[f"veg__{index_name}__{stat_name}"] = value
    return out


def _flatten_texture(plant_dir: Path) -> Dict[str, float]:
    texture_dir = plant_dir / "texture"
    out: Dict[str, float] = {}
    if not texture_dir.is_dir():
        return out
    for band_dir in sorted(texture_dir.iterdir()):
        if not band_dir.is_dir():
            continue
        data = _load_json(band_dir / "texture_statistics.json")
        if not data:
            continue
        for descriptor, stats in data.items():
            if not isinstance(stats, dict):
                continue
            for stat_name, value in stats.items():
                out[f"texture__{band_dir.name}__{descriptor}__{stat_name}"] = value
    return out


def _flatten_morphology(plant_dir: Path) -> Dict[str, float]:
    data = _load_json(plant_dir / "morphology" / "traits.json")
    if not data:
        return {}
    out: Dict[str, float] = {}
    for trait_name, value in data.items():
        if isinstance(value, (int, float)):
            out[f"morph__{trait_name}"] = value
    return out


def collect_feature_matrix(
    output_folder: str,
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Walk a pipeline ``output_folder`` and build the tidy feature matrix.

    Returns a DataFrame with ``plant_id`` and ``date`` identifier columns
    followed by every numeric feature found under each plant's
    ``vegetation_indices/``, ``texture/``, and ``morphology/`` output
    directories. Missing feature groups for a given plant/date are simply
    omitted (resulting in NaNs after concatenation), matching how the pipeline
    skips extractors that were disabled in the config.
    """
    root = Path(output_folder)
    rows = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for plant_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            record: Dict[str, object] = {
                "plant_id": plant_dir.name,
                "date": date_dir.name.replace("_", "-"),
            }
            record.update(_flatten_vegetation(plant_dir))
            record.update(_flatten_texture(plant_dir))
            record.update(_flatten_morphology(plant_dir))
            if len(record) > 2:  # more than just the identifiers
                rows.append(record)

    if not rows:
        logger.warning("No feature records found under %s", output_folder)
        return pd.DataFrame(columns=["plant_id", "date"])

    df = pd.DataFrame(rows)
    id_cols = ["plant_id", "date"]
    feature_cols = sorted(c for c in df.columns if c not in id_cols)
    df = df[id_cols + feature_cols]

    logger.info(
        "Collected %d feature columns for %d (plant, date) records from %s",
        len(feature_cols), len(df), output_folder,
    )
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        logger.info("Saved feature matrix to %s", save_csv)
    return df
