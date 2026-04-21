#!/usr/bin/env python3
"""
Batch-apply SAM3 (text-prompt instance segmentation) to two sources:

1) Composite-with-background images:
   Main_Pipeline/Mullet_with_white_bg/*.png

2) Composite images (no white background):
   Main_Pipeline/New_Mullet_sorghum/**/*__01_composite*.png

Outputs are written into ONE folder. Output basenames end with:
  - "_composite_withbg" for the white_bg source
  - "_composite" for the composite source

For each input image we write:
  - <base>__sam3_mask.png      (union mask, 0/255)
  - <base>__sam3_cutout.png    (RGBA cutout using the union mask)
  - <base>__sam3_overlay.png   (RGB overlay visualization)
  - <base>__sam3.json          (metadata: scores, boxes, source path)

Notes:
- This uses transformers SAM3 if available, otherwise tries the official sam3 repo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _set_hf_cache(project_root: Path) -> None:
    # Ensure a writable HF cache (important on shared filesystems).
    cache_dir = project_root / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))


def _load_sam3(device: str, resolution: int, confidence_threshold: float) -> Tuple[str, Any, Any]:
    """
    Returns (backend, model, processor)
      backend in {"transformers","official"}
    """
    # Try transformers first
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor

        model = Sam3Model.from_pretrained("facebook/sam3-vit-huge").to(device)
        model.eval()
        processor = Sam3Processor.from_pretrained("facebook/sam3-vit-huge")
        return "transformers", model, processor
    except Exception:
        pass

    # Fallback to official repo
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as Sam3ProcessorOfficial

        # NOTE: current official SAM3 model expects a fixed internal resolution.
        # Some versions assert when resolution is changed (e.g., 512). We therefore
        # clamp to the default (1008) for stability.
        if int(resolution) != 1008:
            print(f"[sam3_batch_segment] Warning: official SAM3 resolution={resolution} not supported; using 1008 instead.")
            resolution = 1008

        # Official processor supports controlling resize resolution + confidence filtering.
        processor = Sam3ProcessorOfficial(
            build_sam3_image_model(),
            resolution=int(resolution),
            device=device,
            confidence_threshold=float(confidence_threshold),
        )
        return "official", None, processor
    except Exception as e:
        raise RuntimeError(
            "Could not import SAM3 from transformers or official repo.\n"
            "Install one of:\n"
            "- transformers with Sam3Model/Sam3Processor\n"
            "- official repo: pip install git+https://github.com/facebookresearch/segment-anything-3.git\n"
            f"Last error: {e}"
        )


def _pil_rgb(image_path: Path) -> Image.Image:
    img = Image.open(str(image_path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _overlay_masks(image: Image.Image, masks: np.ndarray, alpha: float = 0.5) -> Image.Image:
    # masks: (N,H,W) uint8 {0,1} or {0,255}
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    n = masks.shape[0]
    if n == 0:
        return img.convert("RGB")

    # deterministic colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0), (255, 0, 255)]
    for i in range(n):
        m = masks[i]
        if m.max() > 1:
            m = (m > 0).astype(np.uint8)
        mask_img = Image.fromarray(m * 255)
        color = colors[i % len(colors)]
        color_overlay = Image.new("RGBA", img.size, color + (int(255 * alpha),))
        alpha_mask = mask_img.point(lambda v: int(v * alpha) if v > 0 else 0)
        color_overlay.putalpha(alpha_mask)
        overlay = Image.alpha_composite(overlay, color_overlay)

    return Image.alpha_composite(img, overlay).convert("RGB")


def _segment_transformers(
    model: Any,
    processor: Any,
    image: Image.Image,
    text_prompt: str,
    device: str,
    threshold: float,
    mask_threshold: float,
) -> Dict[str, Any]:
    import torch

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")
    # normalize to numpy
    if masks is None:
        masks_np = np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
    else:
        masks_np = masks.detach().cpu().numpy()
        # transformers may return (N,1,H,W) or (N,H,W)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        masks_np = (masks_np > 0).astype(np.uint8)
    return {
        "masks": masks_np,
        "boxes": [] if boxes is None else boxes.detach().cpu().numpy().tolist(),
        "scores": [] if scores is None else scores.detach().cpu().numpy().tolist(),
    }


def _segment_official(
    processor: Any,
    image: Image.Image,
    text_prompt: str,
    threshold: float,
    mask_threshold: float,
) -> Dict[str, Any]:
    def _to_list(x: Any) -> List:
        if x is None:
            return []
        # torch Tensor (CPU or CUDA)
        if hasattr(x, "detach") and hasattr(x, "device"):
            try:
                return x.detach().cpu().numpy().tolist()
            except Exception:
                return x.detach().cpu().tolist()
        # numpy or list-like
        try:
            return np.asarray(x).tolist()
        except Exception:
            return list(x)

    # API mirrors what your test script uses.
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=text_prompt)
    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")
    masks_np = np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
    if masks is not None:
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        masks_np = np.asarray(masks)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        # Official backend may return boolean masks already.
        # If masks are probabilistic, apply mask_threshold; otherwise binarize > 0.
        if masks_np.dtype == np.bool_:
            masks_np = masks_np.astype(np.uint8)
        else:
            try:
                masks_np = (masks_np >= float(mask_threshold)).astype(np.uint8)
            except Exception:
                masks_np = (masks_np > 0).astype(np.uint8)

    # Apply score threshold if scores are available.
    scores_list = _to_list(scores)
    boxes_list = _to_list(boxes)
    if scores_list:
        keep_idx = [i for i, s in enumerate(scores_list) if float(s) >= float(threshold)]
        if keep_idx:
            masks_np = masks_np[keep_idx] if masks_np.shape[0] else masks_np
            scores_list = [scores_list[i] for i in keep_idx]
            boxes_list = [boxes_list[i] for i in keep_idx] if boxes_list else []
        else:
            masks_np = np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
            scores_list = []
            boxes_list = []
    return {
        "masks": masks_np,
        "boxes": boxes_list,
        "scores": scores_list,
    }


def _union_mask(masks: np.ndarray) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8) if masks.ndim == 3 else np.zeros((1, 1), dtype=np.uint8)
    return (masks.max(axis=0) > 0).astype(np.uint8)


def _cutout_rgba(image: Image.Image, mask01: np.ndarray) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    alpha = (mask01 * 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _mask_rgb(mask01: np.ndarray) -> Image.Image:
    """Convert 0/1 mask to a viewable RGB image."""
    m = (mask01 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([m, m, m], axis=-1), mode="RGB")


def _save_preview_gif(
    gif_path: Path,
    original: Image.Image,
    overlay: Image.Image,
    cutout: Image.Image,
    mask01: np.ndarray,
    duration_ms: int = 450,
) -> None:
    """
    Create a small looping GIF preview: original -> overlay -> cutout -> mask.
    """
    frames: List[Image.Image] = [
        original.convert("RGB"),
        overlay.convert("RGB"),
        cutout.convert("RGBA").convert("RGB"),
        _mask_rgb(mask01),
    ]
    # Resize to a reasonable width to keep GIF size manageable (preserve aspect).
    max_w = 960
    w, h = frames[0].size
    if w > max_w:
        new_h = int(h * (max_w / w))
        frames = [f.resize((max_w, new_h), resample=Image.BILINEAR) for f in frames]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _safe_base_from_filename(p: Path, kind: str) -> str:
    """
    kind: "composite" or "composite_withbg"
    Create a stable base name without duplicating the composite marker.
    """
    stem = p.stem
    stem = stem.replace("__01_composite__white_bg", "")
    stem = stem.replace("__01_composite", "")
    stem = stem.replace("__white_bg", "")
    stem = stem.rstrip("_")
    return f"{stem}_{kind}"


def _gather_inputs(withbg_dir: Path, composite_root: Path) -> Tuple[List[Path], List[Path]]:
    withbg = sorted([p for p in withbg_dir.glob("*.png") if p.is_file()])
    composite = sorted([p for p in composite_root.rglob("*__01_composite*.png") if p.is_file() and "__white_bg" not in p.name])
    return withbg, composite


def main() -> None:
    # Resolve project root robustly (repo_root/sorghum_pipeline/tools/<this_file>)
    project_root = Path(__file__).resolve().parents[2]
    _set_hf_cache(project_root)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--withbg-dir",
        type=Path,
        default=project_root / "Main_Pipeline" / "Mullet_with_white_bg",
        help="Folder containing *_white_bg composite PNGs",
    )
    ap.add_argument(
        "--composite-root",
        type=Path,
        default=project_root / "Main_Pipeline" / "New_Mullet_sorghum",
        help="Folder to search recursively for *__01_composite*.png",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "Main_Pipeline" / "SAM3dobject_outputs_combined",
        help="Output folder (all results saved here)",
    )
    ap.add_argument(
        "--output-mode",
        type=str,
        default="full",
        choices=["full", "minimal"],
        help=(
            "full: mask + cutout + overlay + json + preview gif (default)\n"
            "minimal: ONLY mask + overlay (no cutout/json/gif)"
        ),
    )
    ap.add_argument("--prompt", type=str, default="plant")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Official SAM3 only: internal resize to NxN. (Most installs only support 1008; others may error.)",
    )
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Official SAM3 only: filter predicted instances by confidence score.",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0, help="0 = process all")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    withbg, composite = _gather_inputs(args.withbg_dir, args.composite_root)
    all_items: List[Tuple[Path, str]] = [(p, "composite_withbg") for p in withbg] + [(p, "composite") for p in composite]
    if args.max_images and args.max_images > 0:
        all_items = all_items[: args.max_images]

    backend, model, processor = _load_sam3(
        args.device,
        resolution=args.resolution,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"SAM3 backend: {backend}")
    print(f"Found {len(withbg)} withbg images and {len(composite)} composite images. Processing {len(all_items)} total.")

    for idx, (img_path, kind) in enumerate(all_items, 1):
        base = _safe_base_from_filename(img_path, kind)
        out_mask = args.out_dir / f"{base}__sam3_mask.png"
        out_cutout = args.out_dir / f"{base}__sam3_cutout.png"
        out_overlay = args.out_dir / f"{base}__sam3_overlay.png"
        out_meta = args.out_dir / f"{base}__sam3.json"
        out_gif = args.out_dir / f"{base}__sam3_preview.gif"

        if args.output_mode == "minimal":
            # Minimal mode: only require mask + overlay
            if out_mask.exists() and out_overlay.exists():
                continue
        else:
            # Full mode: If everything exists (including GIF), skip.
            if out_mask.exists() and out_cutout.exists() and out_overlay.exists() and out_meta.exists() and out_gif.exists():
                continue

            # If SAM outputs exist but GIF missing, generate GIF without rerunning SAM3.
            if out_mask.exists() and out_cutout.exists() and out_overlay.exists() and out_meta.exists() and not out_gif.exists():
                try:
                    original = _pil_rgb(img_path)
                    overlay_img = Image.open(str(out_overlay)).convert("RGB")
                    cutout_img = Image.open(str(out_cutout))
                    mask_img = Image.open(str(out_mask)).convert("L")
                    mask01 = (np.array(mask_img) > 0).astype(np.uint8)
                    _save_preview_gif(out_gif, original=original, overlay=overlay_img, cutout=cutout_img, mask01=mask01)
                except Exception:
                    # If GIF generation fails, fall back to rerun SAM3 below.
                    pass
                else:
                    if idx % 10 == 0 or idx == len(all_items):
                        print(f"[{idx}/{len(all_items)}] processed")
                    continue

        image = _pil_rgb(img_path)
        if backend == "transformers":
            res = _segment_transformers(
                model=model,
                processor=processor,
                image=image,
                text_prompt=args.prompt,
                device=args.device,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )
        else:
            res = _segment_official(
                processor=processor,
                image=image,
                text_prompt=args.prompt,
                # Official backend already uses confidence_threshold internally;
                # also apply the same value here for consistent post-filtering.
                threshold=args.confidence_threshold,
                mask_threshold=args.mask_threshold,
            )

        masks = res["masks"]
        union = _union_mask(masks)

        # Always compute + write the requested outputs.
        overlay_img = _overlay_masks(image, masks)

        # Black/white union mask (0/255) and overlay are always meaningful.
        Image.fromarray((union * 255).astype(np.uint8)).save(out_mask)
        overlay_img.save(out_overlay)

        if args.output_mode == "full":
            # Optional extras (full mode only)
            _cutout_rgba(image, union).save(out_cutout)
            out_meta.write_text(
                json.dumps(
                    {
                        "source": str(img_path),
                        "kind": kind,
                        "prompt": args.prompt,
                        "backend": backend,
                        "resolution": args.resolution,
                        "confidence_threshold": args.confidence_threshold,
                        "threshold": args.threshold,
                        "mask_threshold": args.mask_threshold,
                        "num_objects": int(masks.shape[0]),
                        "scores": res.get("scores", []),
                        "boxes": res.get("boxes", []),
                        "outputs": {
                            "mask": str(out_mask),
                            "cutout": str(out_cutout),
                            "overlay": str(out_overlay),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

            try:
                _save_preview_gif(out_gif, original=image, overlay=overlay_img, cutout=Image.open(str(out_cutout)), mask01=union)
            except Exception:
                # Non-fatal; keep PNG outputs.
                pass

        if idx % 10 == 0 or idx == len(all_items):
            print(f"[{idx}/{len(all_items)}] processed")


if __name__ == "__main__":
    main()

