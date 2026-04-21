#!/usr/bin/env python3
"""
Batch upscale composite images using the PyTorch Real-ESRGAN implementation.

This script includes a tiny compatibility shim for newer torchvision versions
where `torchvision.transforms.functional_tensor` is no longer present, but
some BasicSR versions still import it.

Default target:
  Main_Pipeline/New_Mullet_sorghum/**/*_composite.png

Example:
  python sorghum_pipeline/tools/realesrgan_upscale_composites_torch.py --outscale 4
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import types
from pathlib import Path
from urllib.request import urlopen


def _project_root() -> Path:
    # repo_root/sorghum_pipeline/tools/<this_file>
    return Path(__file__).resolve().parents[2]


def _ensure_torchvision_shim() -> None:
    """
    BasicSR (used by Real-ESRGAN) may import:
        from torchvision.transforms.functional_tensor import rgb_to_grayscale

    Newer torchvision versions removed `functional_tensor`. We provide a module
    alias that forwards the needed symbol.
    """
    try:
        import torchvision.transforms.functional_tensor as _ft  # noqa: F401
        return
    except Exception:
        pass

    from torchvision.transforms import functional as F  # import after torchvision is available

    m = types.ModuleType("torchvision.transforms.functional_tensor")
    m.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = m


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())


def _ensure_weight(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    if dest.exists():
        if expected_sha256 is None:
            return dest
        try:
            if _sha256(dest) == expected_sha256:
                return dest
        except Exception:
            # if hashing fails for any reason, re-download
            pass
        dest.unlink(missing_ok=True)

    print(f"[realesrgan_torch] Downloading weights to {dest} ...")
    _download(url, dest)
    if expected_sha256 is not None:
        got = _sha256(dest)
        if got != expected_sha256:
            raise RuntimeError(f"Downloaded weight hash mismatch for {dest.name}: expected {expected_sha256}, got {got}")
    return dest


def _gather_images(input_root: Path, pattern: str) -> list[Path]:
    return sorted([p for p in input_root.rglob(pattern) if p.is_file()])


def _output_path_for(in_path: Path, input_root: Path, output_root: Path, suffix: str, ext: str) -> Path:
    rel = in_path.relative_to(input_root)
    out_rel = rel.with_suffix(ext)
    out_dir = output_root / out_rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{out_rel.stem}{suffix}{ext}"


def main() -> None:
    project_root = _project_root()
    default_input_root = project_root / "Main_Pipeline" / "New_Mullet_sorghum"
    default_output_root = project_root / "Main_Pipeline" / "New_Mullet_sorghum_upscaled"
    default_cache_dir = project_root / ".cache" / "realesrgan" / "weights"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=default_input_root)
    ap.add_argument("--pattern", type=str, default="*_composite.png")
    ap.add_argument("--output-root", type=Path, default=default_output_root)
    ap.add_argument("--suffix", type=str, default="", help="Output suffix (before extension).")
    ap.add_argument(
        "--model-name",
        type=str,
        default="RealESRGAN_x4plus",
        choices=["RealESRGAN_x4plus"],
        help="Currently supported model(s).",
    )
    ap.add_argument("--outscale", type=float, default=4.0, help="Final scale (can be non-integer).")
    ap.add_argument("--tile", type=int, default=0, help="Tile size, 0 disables tiling.")
    ap.add_argument("--fp32", action="store_true", help="Use fp32 (default fp16 on CUDA).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gpu-id", type=int, default=0, help="GPU id when device=cuda.")
    ap.add_argument("--cache-dir", type=Path, default=default_cache_dir, help="Where to store downloaded weights.")
    ap.add_argument("--max-images", type=int, default=0, help="0 = process all")
    ap.add_argument("--ext", type=str, default="png", choices=["png", "jpg"], help="Output image format.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    # Ensure compatibility before importing realesrgan/basicsr.
    _ensure_torchvision_shim()

    import cv2  # noqa: E402
    import torch  # noqa: E402
    from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
    from realesrgan.utils import RealESRGANer  # noqa: E402

    # Model definition for RealESRGAN_x4plus (official).
    netscale = 4
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)

    # Official weight URL used in upstream README.
    weight_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    weight_path = args.cache_dir / "RealESRGAN_x4plus.pth"
    _ensure_weight(weight_url, weight_path)

    if args.device == "cpu":
        device = torch.device("cpu")
        half = False
        gpu_id = None
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        half = (not args.fp32) and (device.type == "cuda")
        gpu_id = args.gpu_id

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=str(weight_path),
        model=model,
        tile=int(args.tile),
        tile_pad=10,
        pre_pad=10,
        half=half,
        device=device,
        gpu_id=gpu_id,
    )

    images = _gather_images(args.input_root, args.pattern)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    suffix = args.suffix or f"__realesrgan_torch_x{args.outscale:g}"
    ext = "." + args.ext.lstrip(".")

    print(f"[realesrgan_torch] device={device} half={half} tile={args.tile}")
    print(f"[realesrgan_torch] Found {len(images)} images (pattern={args.pattern!r}) under {args.input_root}")
    print(f"[realesrgan_torch] Output root: {args.output_root}")

    n_ok = 0
    for i, in_path in enumerate(images, 1):
        out_path = _output_path_for(in_path, input_root=args.input_root, output_root=args.output_root, suffix=suffix, ext=ext)
        if out_path.exists():
            continue

        if args.dry_run:
            print(f"{in_path} -> {out_path}")
            continue

        img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[realesrgan_torch] WARNING: could not read image: {in_path}")
            continue

        try:
            output, _ = upsampler.enhance(img, outscale=float(args.outscale))
        except Exception as e:
            print(f"[realesrgan_torch] FAILED ({i}/{len(images)}): {in_path} ({e})")
            continue

        ok = cv2.imwrite(str(out_path), output)
        if not ok:
            print(f"[realesrgan_torch] WARNING: could not write output: {out_path}")
            continue

        n_ok += 1
        if i % 10 == 0 or i == len(images):
            print(f"[realesrgan_torch] processed {i}/{len(images)} (new outputs: {n_ok})")

    if args.dry_run:
        print("[realesrgan_torch] dry-run complete (no files written).")
    else:
        print(f"[realesrgan_torch] complete. Wrote {n_ok} new images to: {args.output_root}")


if __name__ == "__main__":
    # Avoid inheriting accidental PYTHONPATH manipulations from other tools.
    os.environ.pop("PYTHONPATH", None)
    main()

