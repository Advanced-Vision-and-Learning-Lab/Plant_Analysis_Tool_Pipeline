#!/usr/bin/env python3
"""
Run a trained Ultralytics YOLO-Pose model on Mullet images and export predicted keypoints.

Output format (per image): JSON with
  - best plant detection score
  - bbox (xyxy in pixels)
  - keypoints: base + leaf_i collar/tip (pixel coords + confidence if available)

This is intended for pseudo-labeling / auto-annotation, then optionally importing into CVAT
or converting into a training set for self-training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


MAX_LEAF_ID = 11
NUM_KEYPOINTS = 1 + 2 * MAX_LEAF_ID


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _layout() -> Dict[str, str]:
    layout: Dict[str, str] = {"0": "base"}
    for i in range(1, MAX_LEAF_ID + 1):
        layout[str(1 + (i - 1) * 2)] = f"leaf_{i}_collar"
        layout[str(1 + (i - 1) * 2 + 1)] = f"leaf_{i}_tip"
    return layout


def _iter_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths: List[Path] = []
    if root.is_file():
        return [root]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def _best_det(result: Any) -> Optional[int]:
    # pick the best by confidence
    if result.boxes is None or len(result.boxes) == 0:
        return None
    conf = result.boxes.conf
    if conf is None or len(conf) == 0:
        return 0
    return int(conf.argmax().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained pose model weights (.pt)",
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=Path("/home/grads/f/fahimehorvatinia/Documents/my_full_project/Main_Pipeline/Mullet_with_white_bg"),
        help="Path to Mullet images folder (or a single image file)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/grads/f/fahimehorvatinia/Documents/my_full_project/Main_Pipeline/Mullet_keypoints_predictions"),
        help="Output folder for JSON predictions",
    )
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="")  # '' => ultralytics default
    args = ap.parse_args()

    model = YOLO(str(args.model))
    images = _iter_images(args.images)
    if not images:
        raise SystemExit(f"No images found under: {args.images}")

    _ensure_dir(args.out)
    layout = _layout()

    for img in images:
        results = model.predict(
            source=str(img),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        r = results[0]
        bi = _best_det(r)
        out: Dict[str, Any] = {
            "image": str(img),
            "keypoint_layout": layout,
            "found": False,
        }
        if bi is None:
            (args.out / f"{img.stem}.json").write_text(json.dumps(out, indent=2) + "\n")
            continue

        box = r.boxes[bi]
        out["found"] = True
        out["score"] = float(box.conf.item()) if box.conf is not None else None
        out["class_id"] = int(box.cls.item()) if box.cls is not None else None
        out["bbox_xyxy"] = [float(x) for x in box.xyxy.squeeze(0).tolist()]

        kps = None
        if r.keypoints is not None and len(r.keypoints) > bi:
            kps = r.keypoints[bi]

        # Ultralytics keypoints object exposes .xy (Nx2) and sometimes .conf (N)
        keypoints: List[Dict[str, Any]] = []
        if kps is not None:
            xy = kps.xy.squeeze(0).tolist()  # (N,2)
            conf = None
            if hasattr(kps, "conf") and kps.conf is not None:
                conf = kps.conf.squeeze(0).tolist()
            for i in range(min(len(xy), NUM_KEYPOINTS)):
                kp: Dict[str, Any] = {
                    "i": i,
                    "name": layout.get(str(i), f"kpt_{i}"),
                    "x": float(xy[i][0]),
                    "y": float(xy[i][1]),
                }
                if conf is not None and i < len(conf):
                    kp["conf"] = float(conf[i])
                keypoints.append(kp)

        out["keypoints"] = keypoints
        (args.out / f"{img.stem}.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote predictions to: {args.out}")


if __name__ == "__main__":
    main()

