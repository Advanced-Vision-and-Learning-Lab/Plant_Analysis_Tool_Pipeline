#!/usr/bin/env python3
"""
Batch upscale composite images using Real-ESRGAN (portable ncnn-vulkan binary).

Why this script exists:
- The PyTorch Real-ESRGAN python package depends on BasicSR, which can be
  version-sensitive with your local torch/torchvision stack.
- The official "ncnn-vulkan" build is portable and avoids Python deps entirely.

Default target:
  Main_Pipeline/New_Mullet_sorghum/**/*_composite.png

Outputs:
  Mirrors the input folder structure into an output root, with a suffix.

Example:
  python sorghum_pipeline/tools/realesrgan_upscale_composites.py --scale 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


def _project_root() -> Path:
    # repo_root/sorghum_pipeline/tools/<this_file>
    return Path(__file__).resolve().parents[2]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _chmod_plus_x(p: Path) -> None:
    try:
        mode = p.stat().st_mode
        p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        # best-effort
        pass


def _ensure_realesrgan_ncnn(install_dir: Path, version: str = "v0.2.0") -> Path:
    """
    Downloads and unzips Real-ESRGAN-ncnn-vulkan into install_dir if needed.
    """
    # Official release assets are stable and small; keep pinned for reproducibility.
    zip_name = f"realesrgan-ncnn-vulkan-{version}-ubuntu.zip"
    url = f"https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/{version}/realesrgan-ncnn-vulkan-{version}-ubuntu.zip"

    # We treat install_dir as the extraction root.
    marker = install_dir / ".installed"
    if marker.exists():
        bin_path = next(install_dir.rglob("realesrgan-ncnn-vulkan"), None)
        if bin_path:
            _chmod_plus_x(bin_path)
            return bin_path

    install_dir.mkdir(parents=True, exist_ok=True)
    zip_path = install_dir / zip_name

    if not zip_path.exists():
        print(f"[realesrgan_upscale] Downloading {version} ncnn-vulkan binary...")
        _download(url, zip_path)

    print(f"[realesrgan_upscale] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(install_dir)

    bin_path = next(install_dir.rglob("realesrgan-ncnn-vulkan"), None)
    if not bin_path:
        raise RuntimeError(
            "Could not locate realesrgan-ncnn-vulkan binary after extraction. "
            f"Looked under: {install_dir}"
        )

    _chmod_plus_x(bin_path)
    marker.write_text("ok\n")
    return bin_path


def _gather_images(input_root: Path, pattern: str) -> list[Path]:
    # Allow both explicit glob and simple filenames.
    # Use rglob for recursion to match the New_Mullet_sorghum layout.
    return sorted([p for p in input_root.rglob(pattern) if p.is_file()])


def _output_path_for(
    in_path: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
    ext: str = ".png",
) -> Path:
    rel = in_path.relative_to(input_root)
    out_rel = rel.with_suffix(ext)
    out_dir = output_root / out_rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{out_rel.stem}{suffix}{ext}"


def main() -> None:
    project_root = _project_root()
    default_input_root = project_root / "Main_Pipeline" / "New_Mullet_sorghum"
    default_output_root = project_root / "Main_Pipeline" / "New_Mullet_sorghum_upscaled"
    default_cache_dir = project_root / ".cache" / "realesrgan-ncnn-vulkan" / "v0.2.0"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=default_input_root)
    ap.add_argument("--pattern", type=str, default="*_composite.png", help="Glob pattern (recursive under input-root).")
    ap.add_argument("--output-root", type=Path, default=default_output_root)
    ap.add_argument("--suffix", type=str, default="", help="Extra suffix appended to the output filename (before .png).")
    ap.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Upscale ratio supported by ncnn build (2/3/4).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="realesrgan-x4plus",
        help="Model name for ncnn build (e.g. realesrgan-x4plus, realesrnet-x4plus, realesr-animevideov3, realesrgan-x4plus-anime).",
    )
    ap.add_argument("--tile", type=int, default=0, help="Tile size (0=auto).")
    ap.add_argument("--gpu-id", type=str, default="auto", help="GPU id for ncnn build (e.g. 0,1,2 or 'auto').")
    ap.add_argument("--jobs", type=str, default="1:2:2", help="Thread count load:proc:save, e.g. 1:2:2")
    ap.add_argument("--cache-dir", type=Path, default=default_cache_dir, help="Where to download/extract the binary.")
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing ncnn model files (e.g. realesrgan-x4plus.param/bin). "
            "Note: upstream Linux zip does not include models."
        ),
    )
    ap.add_argument("--max-images", type=int, default=0, help="0 = process all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    # auto-suffix if user didn't provide one
    suffix = args.suffix
    if suffix == "":
        suffix = f"__realesrgan_x{args.scale}"

    bin_path = _ensure_realesrgan_ncnn(args.cache_dir)
    models_dir = args.models_dir
    if models_dir is None:
        # Try a best-effort lookup (some third-party zips bundle models).
        models_dir = next((p for p in args.cache_dir.rglob("models") if p.is_dir()), None)
    if models_dir is None or not Path(models_dir).exists():
        raise SystemExit(
            "NCNN/Vulkan binary is installed, but no model folder was found.\n"
            "Use the PyTorch script instead:\n"
            "  python sorghum_pipeline/tools/realesrgan_upscale_composites_torch.py --outscale 4\n"
            "Or provide a model folder via:\n"
            "  --models-dir /path/to/models\n"
        )

    images = _gather_images(input_root, args.pattern)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    print(f"[realesrgan_upscale] Binary: {bin_path}")
    print(f"[realesrgan_upscale] Models: {models_dir}")
    print(f"[realesrgan_upscale] Found {len(images)} images under {input_root} (pattern={args.pattern!r})")
    print(f"[realesrgan_upscale] Output root: {output_root}")

    n_ok = 0
    for i, in_path in enumerate(images, 1):
        out_path = _output_path_for(in_path, input_root=input_root, output_root=output_root, suffix=suffix, ext=".png")

        if out_path.exists():
            continue

        cmd = [
            str(bin_path),
            "-i",
            str(in_path),
            "-o",
            str(out_path),
            "-s",
            str(args.scale),
            "-t",
            str(args.tile),
            "-m",
            str(models_dir),
            "-n",
            str(args.model),
            "-g",
            str(args.gpu_id),
            "-j",
            str(args.jobs),
            "-f",
            "png",
        ]

        if args.dry_run:
            print(" ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[realesrgan_upscale] FAILED ({i}/{len(images)}): {in_path}")
            print(e.stdout)
            continue

        n_ok += 1
        if i % 10 == 0 or i == len(images):
            print(f"[realesrgan_upscale] processed {i}/{len(images)} (new outputs: {n_ok})")

    if args.dry_run:
        print("[realesrgan_upscale] dry-run complete (no files written).")
    else:
        print(f"[realesrgan_upscale] complete. Wrote {n_ok} new upscaled images to: {output_root}")


if __name__ == "__main__":
    # Avoid inheriting accidental PYTHONPATH manipulations from other tools.
    os.environ.pop("PYTHONPATH", None)
    main()

