#!/usr/bin/env python3
"""
Create a TRUE 3D reconstruction GIF using the local SAM-3D-Objects repo.

Pipeline (per image):
  1) Read RGB image + a binary mask (0/1) for the object.
  2) Run SAM-3D-Objects inference -> gaussian splat reconstruction.
  3) Render turntable frames and save as GIF.

This script is designed to work with masks produced by:
  Main_Pipeline/SAM3dobject_outputs_combined/*__sam3_mask.png
and the original image path recorded in the corresponding JSON:
  *.__sam3.json  (field: "source")

Requirements:
  - A working SAM-3D-Objects environment (see sam-3d-objects/doc/setup.md)
  - Downloaded checkpoints (checkpoints/hf/pipeline.yaml, etc.)
  - GPU with large VRAM (the official docs recommend 32GB+)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _import_sam3d_objects(sam3d_repo: Path) -> Any:
    """
    Import notebook inference utilities from the local repo.
    Returns the imported module (inference.py).
    """
    nb_dir = sam3d_repo / "notebook"
    if not nb_dir.is_dir():
        raise FileNotFoundError(f"Expected notebook/ in {sam3d_repo}")

    # Ensure we skip repo init hook that may not be present in this checkout.
    # The official notebook code also sets this before importing sam3d_objects.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    # Make sure all caches are user-writable (some systems default to /scratch which may be forbidden).
    cache_root = sam3d_repo.parent.parent / ".cache"  # <my_full_project>/.cache
    hf_cache = cache_root / "huggingface"
    mpl_cache = cache_root / "matplotlib"
    hf_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    # Torch hub/torchvision may try to cache under /scratch on some systems unless set.
    torch_home = cache_root / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    # Kaolin -> warp may try to compile kernels into a non-writable directory (e.g. /scratch).
    # Preconfigure Warp kernel cache and Torch extension build dir to user-writable locations
    # BEFORE importing sam-3d-objects inference (which imports kaolin/warp).
    warp_cache = cache_root / "warp_kernels"
    torch_ext = cache_root / "torch_extensions"
    warp_cache.mkdir(parents=True, exist_ok=True)
    torch_ext.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_ext))
    try:
        import warp as wp  # type: ignore

        wp.config.kernel_cache_dir = str(warp_cache)
        # do NOT call wp.init() here; let downstream import call it with our config applied
    except Exception:
        # warp may not be importable until dependencies are fully installed; ignore here
        pass

    sys.path.insert(0, str(nb_dir))
    try:
        import inference as sam3d_inference  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import SAM-3D-Objects notebook inference.\n"
            "This usually means the SAM-3D-Objects dependencies are not installed.\n"
            "Please follow:\n"
            f"  {sam3d_repo}/doc/setup.md\n"
            f"Import error: {type(e).__name__}: {e}"
        )
    return sam3d_inference


def _load_mask01(mask_path: Path) -> np.ndarray:
    m = Image.open(str(mask_path)).convert("L")
    arr = np.array(m)
    return (arr > 0).astype(np.uint8)


def _load_rgb(image_path: Path) -> np.ndarray:
    img = Image.open(str(image_path)).convert("RGB")
    return np.array(img)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_gif(frames: List[np.ndarray], out_gif: Path, fps: float = 30.0) -> None:
    duration = 1.0 / max(1e-6, fps)
    try:
        import imageio

        imageio.mimsave(str(out_gif), frames, format="GIF", duration=duration, loop=0)
        return
    except Exception:
        # Pillow fallback
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            str(out_gif),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(duration * 1000),
            loop=0,
            optimize=True,
        )


def run_one(
    sam3d_inference: Any,
    config_path: Path,
    image_path: Path,
    mask_path: Path,
    out_gif: Path,
    seed: int,
    resolution: int,
    num_frames: int,
    r: float,
    fov: float,
    pitch_deg: float,
    yaw_start_deg: float,
    fps: float,
) -> None:
    # Load model
    inference = sam3d_inference.Inference(str(config_path), compile=False)

    # Load inputs
    image = _load_rgb(image_path)
    mask01 = _load_mask01(mask_path)

    # Run inference.
    # Use the underlying pipeline directly so we can request only Gaussian output
    # (lower VRAM than decoding meshes/textures).
    rgba = inference.merge_mask_to_rgba(image, mask01)
    output = inference._pipeline.run(  # type: ignore[attr-defined]
        rgba,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        decode_formats=["gaussian"],
    )

    # Build scene + render turntable
    scene_gs = sam3d_inference.make_scene(output)
    scene_gs = sam3d_inference.ready_gaussian_for_video_rendering(scene_gs)

    video = sam3d_inference.render_video(
        scene_gs,
        r=r,
        fov=fov,
        pitch_deg=pitch_deg,
        yaw_start_deg=yaw_start_deg,
        resolution=resolution,
        num_frames=num_frames,
    )["color"]

    _ensure_dir(out_gif.parent)
    _save_gif(list(video), out_gif, fps=fps)


def main() -> None:
    # Resolve project root robustly (repo_root/sorghum_pipeline/tools/<this_file>)
    project_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sam3d-repo",
        type=Path,
        default=project_root / "Main_Pipeline" / "sam-3d-objects",
        help="Path to local sam-3d-objects repo",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to pipeline.yaml (default: <repo>/checkpoints/hf/pipeline.yaml)",
    )
    ap.add_argument(
        "--sam3-json",
        type=Path,
        default=None,
        help="Path to *.__sam3.json produced by our SAM3 segmentation batch (contains original 'source')",
    )
    ap.add_argument("--image", type=Path, default=None, help="Path to RGB input image (override)")
    ap.add_argument("--mask", type=Path, default=None, help="Path to binary mask PNG (override)")
    ap.add_argument("--out-gif", type=Path, required=True, help="Output GIF path")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=120)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--pitch-deg", type=float, default=15.0)
    ap.add_argument("--yaw-start-deg", type=float, default=-45.0)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    config_path = args.config or (args.sam3d_repo / "checkpoints" / "hf" / "pipeline.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing SAM-3D-Objects checkpoints config: {config_path}\n"
            "Download checkpoints as described in sam-3d-objects/doc/setup.md"
        )

    # Resolve image/mask from sam3 json if provided
    image_path = args.image
    mask_path = args.mask
    if args.sam3_json is not None:
        meta = json.loads(Path(args.sam3_json).read_text())
        if image_path is None:
            image_path = Path(meta["source"])
        if mask_path is None:
            # infer mask next to json: replace __sam3.json with __sam3_mask.png
            mask_guess = Path(str(args.sam3_json)).with_suffix("")  # remove .json
            if str(mask_guess).endswith("__sam3"):
                mask_guess = Path(str(mask_guess) + "_mask.png")  # unlikely
            # actual naming from our pipeline: __sam3_mask.png and __sam3.json
            mask_guess = Path(str(args.sam3_json).replace("__sam3.json", "__sam3_mask.png"))
            mask_path = mask_guess

    if image_path is None or mask_path is None:
        raise SystemExit("Provide either --sam3-json or both --image and --mask.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    sam3d_inference = _import_sam3d_objects(args.sam3d_repo)
    run_one(
        sam3d_inference=sam3d_inference,
        config_path=config_path,
        image_path=image_path,
        mask_path=mask_path,
        out_gif=args.out_gif,
        seed=args.seed,
        resolution=args.resolution,
        num_frames=args.num_frames,
        r=args.r,
        fov=args.fov,
        pitch_deg=args.pitch_deg,
        yaw_start_deg=args.yaw_start_deg,
        fps=args.fps,
    )
    print(f"Wrote 3D GIF: {args.out_gif}")


if __name__ == "__main__":
    main()

