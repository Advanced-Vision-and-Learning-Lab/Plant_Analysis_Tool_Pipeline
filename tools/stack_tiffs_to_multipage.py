#!/usr/bin/env python3
"""
Stack many single-page TIFFs into one multi-page TIFF (so you can scroll pages).

Example:
  python sorghum_pipeline/tools/stack_tiffs_to_multipage.py \
    --input-dir "/path/to/TIFF/Camera1" \
    --output "/path/to/TIFF/Camera1_stack.tif"

Notes:
- Files are sorted by numeric index in filename (e.g., Image_000123.tif -> 123).
- Uses `tifffile` for streaming write (does not load all images at once).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
from PIL import Image


NUM_RE = re.compile(r"(\d+)")


def _num_key(p: Path) -> Tuple[int, str]:
    m = NUM_RE.findall(p.stem)
    if not m:
        return (10**18, p.name)
    return (int(m[-1]), p.name)


def list_tiffs_sorted(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    return sorted(files, key=_num_key)


def read_as_array(path: Path) -> np.ndarray:
    # PIL handles many TIFF variants reliably
    with Image.open(str(path)) as im:
        arr = np.array(im)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Stack single-page TIFFs into one multi-page TIFF.")
    ap.add_argument("--input-dir", required=True, help="Folder containing .tif/.tiff files")
    ap.add_argument("--output", required=True, help="Output multi-page TIFF path")
    ap.add_argument("--compression", default="deflate", help="TIFF compression (default: deflate)")
    ap.add_argument("--dry-run", action="store_true", help="Print planned action only")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    files = list_tiffs_sorted(input_dir)
    if not files:
        raise SystemExit(f"No TIFF files found in: {input_dir}")

    print(f"Input:  {input_dir}")
    print(f"Files:  {len(files)}")
    print(f"Output: {out_path}")
    print(f"Compression: {args.compression}")
    print(f"First: {files[0].name}")
    print(f"Last:  {files[-1].name}")

    if args.dry_run:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Stream write: one page at a time
    with tifffile.TiffWriter(str(out_path), bigtiff=True) as tw:
        for i, p in enumerate(files, start=1):
            arr = read_as_array(p)
            # Ensure at least 2D
            if arr.ndim == 0:
                arr = arr.reshape((1, 1))
            tw.write(arr, compression=args.compression)
            if i % 50 == 0 or i == len(files):
                print(f"  wrote {i}/{len(files)}")


if __name__ == "__main__":
    main()

