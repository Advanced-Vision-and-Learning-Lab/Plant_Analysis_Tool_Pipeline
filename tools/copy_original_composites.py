#!/usr/bin/env python3
"""
Copy original *_composite.png images into a single "composite" folder.

Default behavior:
- Source:   Main_Pipeline/New_Mullet_sorghum (recursive)
- Pattern:  *_composite.png
- Dest:     Main_Pipeline/composite
- Preserves relative folder structure under the source root to avoid name collisions.

Example:
  python sorghum_pipeline/tools/copy_original_composites.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    project_root = _project_root()
    default_src = project_root / "Main_Pipeline" / "New_Mullet_sorghum"
    default_dst = project_root / "Main_Pipeline" / "composite"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=default_src)
    ap.add_argument("--pattern", type=str, default="*_composite.png")
    ap.add_argument("--dst-root", type=Path, default=default_dst)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if destination file exists.")
    args = ap.parse_args()

    src_root: Path = args.src_root
    dst_root: Path = args.dst_root
    if not src_root.exists():
        raise SystemExit(f"Source root does not exist: {src_root}")

    images = sorted([p for p in src_root.rglob(args.pattern) if p.is_file()])
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for p in images:
        rel = p.relative_to(src_root)
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(p, out)
        copied += 1

    print(f"Found {len(images)} images in {src_root}")
    print(f"Copied {copied} to {dst_root}")
    print(f"Skipped {skipped} (already existed)")


if __name__ == "__main__":
    main()

