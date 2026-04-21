#!/usr/bin/env python3
"""
Prepare UNL-CPPD (keypoints) into Ultralytics YOLO-Pose format.

UNL-CPPD annotations (Ground Truth) provide:
  - plant base point
  - for each leaf: collar (junction) + tip
  - leaf status (alive/dead/missing)

Ultralytics YOLO-Pose requires a fixed keypoint layout per instance, so we model
each image as ONE "plant" instance with K keypoints:

  kpt[0]              = base
  kpt[1], kpt[2]      = leaf_1 collar, leaf_1 tip
  kpt[3], kpt[4]      = leaf_2 collar, leaf_2 tip
  ...
  kpt[21], kpt[22]    = leaf_11 collar, leaf_11 tip

Missing leaves are marked with v=0 for their points.
We also export a sidecar JSON with leaf statuses per image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install with: pip install pillow\n"
        f"Import error: {e}"
    )


MAX_LEAF_ID = 11
NUM_KEYPOINTS = 1 + 2 * MAX_LEAF_ID  # base + (collar,tip) per leaf
CLASS_ID = 0
CLASS_NAME = "plant"


@dataclass(frozen=True)
class Sample:
    gt_json: Path
    img_path: Path
    rel_id: str  # e.g. Plant_191-28_SideView90/Day_026


def _parse_gt_rel_id(gt_json_path: Path) -> str:
    # Ground Truth layout: Ground Truth/<Plant_XXX_SideViewYY>/<Day_NNN>.json
    return f"{gt_json_path.parent.name}/{gt_json_path.stem}"


def _map_gt_to_image(unl_root: Path, gt_rel_id: str) -> Path:
    """
    Map e.g. 'Plant_191-28_SideView90/Day_026' to:
      <unl_root>/Original Image/Plant_191-28/SideView90/Day_026.png
    """
    plant_view, day = gt_rel_id.split("/", 1)
    if "_SideView" not in plant_view:
        raise ValueError(f"Unexpected GT folder name (missing _SideView): {plant_view}")
    plant, view_suffix = plant_view.rsplit("_SideView", 1)
    view = f"SideView{view_suffix}"
    return unl_root / "Original Image" / plant / view / f"{day}.png"


def _stable_split(rel_id: str, val_ratio: float, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{rel_id}".encode("utf-8")).hexdigest()
    r = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if r < val_ratio else "train"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    _ensure_dir(dst.parent)
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unknown mode: {mode}")


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bbox_from_points(pts: List[Tuple[float, float]], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max(1.0, max_x - min_x)
    bh = max(1.0, max_y - min_y)
    pad_x = 0.15 * bw + 10.0
    pad_y = 0.15 * bh + 10.0
    x1 = _clip(min_x - pad_x, 0.0, img_w - 1.0)
    y1 = _clip(min_y - pad_y, 0.0, img_h - 1.0)
    x2 = _clip(max_x + pad_x, 0.0, img_w - 1.0)
    y2 = _clip(max_y + pad_y, 0.0, img_h - 1.0)
    # enforce a minimum bbox size to avoid tiny boxes (e.g., base-only frames)
    min_box_w = 0.15 * img_w
    min_box_h = 0.15 * img_h
    cur_w = x2 - x1
    cur_h = y2 - y1
    if cur_w < min_box_w:
        cx = (x1 + x2) / 2.0
        x1 = _clip(cx - min_box_w / 2.0, 0.0, img_w - 1.0)
        x2 = _clip(cx + min_box_w / 2.0, 0.0, img_w - 1.0)
    if cur_h < min_box_h:
        cy = (y1 + y2) / 2.0
        y1 = _clip(cy - min_box_h / 2.0, 0.0, img_h - 1.0)
        y2 = _clip(cy + min_box_h / 2.0, 0.0, img_h - 1.0)
    if x2 <= x1:
        x2 = _clip(x1 + 1.0, 0.0, img_w - 1.0)
    if y2 <= y1:
        y2 = _clip(y1 + 1.0, 0.0, img_h - 1.0)
    return x1, y1, x2, y2


def _load_gt_points(gt_json: Path) -> Tuple[List[Tuple[float, float, int]], Dict[str, str]]:
    """
    Returns:
      keypoints: list of (x,y,v) length NUM_KEYPOINTS in pixels
      statuses: dict leaf_id(str)->status
    """
    data = json.loads(gt_json.read_text())

    kpts: List[Tuple[float, float, int]] = [(0.0, 0.0, 0) for _ in range(NUM_KEYPOINTS)]
    statuses: Dict[str, str] = {}

    base = data.get("base") or {}
    bx, by = float(base["x"]), float(base["y"])
    kpts[0] = (bx, by, 2)

    leaves = data.get("leaf") or []
    for leaf in leaves:
        try:
            leaf_id = int(leaf.get("id"))
        except Exception:
            continue
        if not (1 <= leaf_id <= MAX_LEAF_ID):
            continue

        statuses[str(leaf_id)] = str(leaf.get("status", "unknown"))

        collar = leaf.get("collar") or {}
        tip = leaf.get("tip") or {}

        if "x" in collar and "y" in collar:
            cx, cy = float(collar["x"]), float(collar["y"])
            kpts[1 + (leaf_id - 1) * 2] = (cx, cy, 2)
        if "x" in tip and "y" in tip:
            tx, ty = float(tip["x"]), float(tip["y"])
            kpts[1 + (leaf_id - 1) * 2 + 1] = (tx, ty, 2)

    return kpts, statuses


def _normalize_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def _normalize_kpts(kpts: List[Tuple[float, float, int]], w: int, h: int) -> List[Tuple[float, float, int]]:
    out: List[Tuple[float, float, int]] = []
    for x, y, v in kpts:
        if v == 0:
            out.append((0.0, 0.0, 0))
        else:
            out.append((x / w, y / h, int(v)))
    return out


def _write_label_pose_txt(
    label_path: Path,
    bbox_n: Tuple[float, float, float, float],
    kpts_n: List[Tuple[float, float, int]],
) -> None:
    _ensure_dir(label_path.parent)
    xc, yc, bw, bh = bbox_n
    parts: List[str] = [str(CLASS_ID), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for x, y, v in kpts_n:
        parts.extend([f"{x:.6f}", f"{y:.6f}", str(int(v))])
    label_path.write_text(" ".join(parts) + "\n")


def _write_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "unl_cppd_pose.yaml"
    flip_idx = list(range(NUM_KEYPOINTS))
    # Ultralytics pose evaluation needs sigmas; use a reasonable constant if unknown.
    kpt_sigmas = [0.05] * NUM_KEYPOINTS
    yaml_text = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
        f"kpt_shape: [{NUM_KEYPOINTS}, 3]\n"
        f"flip_idx: {flip_idx}\n"
        f"kpt_sigmas: {kpt_sigmas}\n"
    )
    yaml_path.write_text(yaml_text)
    return yaml_path


def collect_samples(unl_root: Path) -> List[Sample]:
    gt_root = unl_root / "Ground Truth"
    if not gt_root.is_dir():
        raise FileNotFoundError(f"Ground Truth not found at: {gt_root}")

    samples: List[Sample] = []
    for gt_json in gt_root.rglob("*.json"):
        rel_id = _parse_gt_rel_id(gt_json)
        img_path = _map_gt_to_image(unl_root, rel_id)
        if not img_path.is_file():
            continue
        samples.append(Sample(gt_json=gt_json, img_path=img_path, rel_id=rel_id))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unl-root",
        type=Path,
        default=Path("/home/grads/f/fahimehorvatinia/Documents/my_full_project/Dataset/UNL-CPPD"),
        help="Path to extracted UNL-CPPD root (contains 'Ground Truth' and 'Original Image')",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/grads/f/fahimehorvatinia/Documents/my_full_project/Dataset/_ultralytics_unl_cppd_pose"),
        help="Output dataset directory (will create images/ and labels/)",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to place images into output dataset",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    samples = collect_samples(args.unl_root)
    if not samples:
        raise SystemExit(f"No samples found under: {args.unl_root}")

    out_images = args.out_root / "images"
    out_labels = args.out_root / "labels"
    out_status = args.out_root / "leaf_status"

    n_written = 0
    for s in samples:
        split = _stable_split(s.rel_id, args.val_ratio, args.seed)
        safe_name = s.rel_id.replace("/", "__")
        img_dst = out_images / split / f"{safe_name}.png"
        label_dst = out_labels / split / f"{safe_name}.txt"
        status_dst = out_status / split / f"{safe_name}.json"

        _link_or_copy(s.img_path, img_dst, args.mode)

        with Image.open(s.img_path) as im:
            w, h = im.size

        kpts_px, statuses = _load_gt_points(s.gt_json)
        pts_for_bbox = [(x, y) for (x, y, v) in kpts_px if v > 0]
        x1, y1, x2, y2 = _bbox_from_points(pts_for_bbox, w, h)
        bbox_n = _normalize_bbox(x1, y1, x2, y2, w, h)
        kpts_n = _normalize_kpts(kpts_px, w, h)
        _write_label_pose_txt(label_dst, bbox_n, kpts_n)

        _ensure_dir(status_dst.parent)
        status_dst.write_text(
            json.dumps(
                {
                    "image": str(s.img_path),
                    "gt": str(s.gt_json),
                    "rel_id": s.rel_id,
                    "leaf_status": statuses,
                    "keypoint_layout": {
                        "0": "base",
                        **{str(1 + (i - 1) * 2): f"leaf_{i}_collar" for i in range(1, MAX_LEAF_ID + 1)},
                        **{str(1 + (i - 1) * 2 + 1): f"leaf_{i}_tip" for i in range(1, MAX_LEAF_ID + 1)},
                    },
                },
                indent=2,
            )
            + "\n"
        )

        n_written += 1

    yaml_path = _write_yaml(args.out_root)
    print(f"Wrote {n_written} samples to: {args.out_root}")
    print(f"Data YAML: {yaml_path}")


if __name__ == "__main__":
    main()

