#!/usr/bin/env python3
"""
Organize a recording's TIFF images into:

  <output_root>/<YYYY-MM-DD>/Plant_###/CameraX/{front,back}/Image_XXXXXX.tif

Assumptions (configurable):
- Images are stored under <tiff_root>/Camera1, <tiff_root>/Camera2, ...
- Within each camera folder, images belong to plants in sorted order by numeric index
  extracted from filename (e.g., Image_000041.tif -> 41).
- For each plant: first N images are "front", next M images are "back"

Any remainder images that don't make a full plant chunk are placed under:
  <output_root>/<YYYY-MM-DD>/_extras/CameraX/
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional


OnExists = Literal["error", "skip", "overwrite"]
Mode = Literal["copy", "move"]


DATE_RE = re.compile(r"(?:^|[^0-9])(\d{4}-\d{2}-\d{2})(?:[^0-9]|$)")
NUM_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class PlanItem:
    src: Path
    dst: Path


@dataclass(frozen=True)
class PlantBlock:
    num_plants: int
    front_count: int
    back_count: int

    @property
    def chunk(self) -> int:
        return int(self.front_count) + int(self.back_count)

    @property
    def total_images(self) -> int:
        return int(self.num_plants) * self.chunk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Organize recording TIFFs into date/plant/camera/front-back folders."
    )
    p.add_argument(
        "--tiff-root",
        required=True,
        help="Path to the TIFF folder containing Camera1/, Camera2/, ...",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help=(
            "Where to create the date folder. Defaults to parent of --tiff-root "
            "(i.e., the recording folder)."
        ),
    )
    p.add_argument(
        "--date",
        default=None,
        help=(
            "Date folder name to create (YYYY-MM-DD). If omitted, attempts to extract "
            "from parent folder name (e.g., Recording_2026-02-09_...)."
        ),
    )
    p.add_argument("--front-count", type=int, default=10, help="Images per plant (front).")
    p.add_argument("--back-count", type=int, default=10, help="Images per plant (back).")
    p.add_argument(
        "--plant-blocks",
        default=None,
        help=(
            "Optional mixed per-plant layout. Format: 'N:F:B,N:F:B,...' where "
            "N=num plants in block, F=front images/plant, B=back images/plant. "
            "Example for your 2026-02-09 rule: '7:10:10,8:12:12'. "
            "If provided, --front-count/--back-count are ignored for plant assignment."
        ),
    )
    p.add_argument(
        "--plant-prefix",
        default="Plant_",
        help="Prefix for plant folders (default: Plant_).",
    )
    p.add_argument(
        "--plant-padding",
        type=int,
        default=3,
        help="Zero-padding width for plant numbers (default: 3 -> Plant_001).",
    )
    p.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Copy or move files into the new structure (default: copy).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying/moving files.",
    )
    p.add_argument(
        "--on-exists",
        choices=("error", "skip", "overwrite"),
        default="error",
        help="What to do if destination file exists (default: error).",
    )
    return p.parse_args()


def parse_plant_blocks(spec: Optional[str]) -> Optional[list[PlantBlock]]:
    if not spec:
        return None
    s = str(spec).strip()
    if not s:
        return None

    blocks: list[PlantBlock] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        cols = [c.strip() for c in part.split(":")]
        if len(cols) != 3:
            raise SystemExit(
                f"Invalid --plant-blocks part {part!r}. Expected 'N:F:B' like '7:10:10'."
            )
        try:
            n = int(cols[0])
            f = int(cols[1])
            b = int(cols[2])
        except Exception:
            raise SystemExit(
                f"Invalid --plant-blocks numbers in {part!r}. Expected integers like '7:10:10'."
            )
        if n <= 0 or f < 0 or b < 0 or (f + b) <= 0:
            raise SystemExit(
                f"Invalid block {part!r}. Need N>0 and F>=0 and B>=0 and (F+B)>0."
            )
        blocks.append(PlantBlock(num_plants=n, front_count=f, back_count=b))

    if not blocks:
        return None
    return blocks


def infer_date_folder(tiff_root: Path, explicit_date: str | None) -> str:
    if explicit_date:
        return explicit_date

    # Try the recording folder name first (parent of TIFF)
    candidates = [tiff_root.parent.name, tiff_root.name]
    for c in candidates:
        m = DATE_RE.search(c)
        if m:
            return m.group(1)

    raise SystemExit(
        "Could not infer date (YYYY-MM-DD). Provide --date explicitly."
    )


def iter_camera_dirs(tiff_root: Path) -> list[Path]:
    if not tiff_root.exists():
        raise SystemExit(f"TIFF root does not exist: {tiff_root}")
    if not tiff_root.is_dir():
        raise SystemExit(f"TIFF root is not a directory: {tiff_root}")

    cams: list[Path] = []
    for p in sorted(tiff_root.iterdir()):
        if p.is_dir() and p.name.lower().startswith("camera"):
            cams.append(p)
    if not cams:
        raise SystemExit(f"No Camera*/ directories found under: {tiff_root}")
    return cams


def tif_files_sorted(camera_dir: Path) -> list[Path]:
    files = [p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]

    def key(p: Path) -> tuple[int, str]:
        m = NUM_RE.findall(p.stem)
        if not m:
            # Put non-numeric names last but stable
            return (10**18, p.name)
        return (int(m[-1]), p.name)

    return sorted(files, key=key)


def plant_dir_name(prefix: str, idx: int, padding: int) -> str:
    return f"{prefix}{idx:0{padding}d}"


def build_plan_for_camera(
    *,
    camera_dir: Path,
    out_date_dir: Path,
    front_count: int,
    back_count: int,
    plant_blocks: Optional[list[PlantBlock]],
    plant_prefix: str,
    plant_padding: int,
) -> tuple[list[PlanItem], list[PlanItem]]:
    """Returns (main_items, extras_items)."""
    files = tif_files_sorted(camera_dir)
    if plant_blocks:
        total_needed = sum(b.total_images for b in plant_blocks)
        if total_needed <= 0:
            raise SystemExit("--plant-blocks resulted in 0 total images.")
    else:
        chunk = front_count + back_count
        if chunk <= 0:
            raise SystemExit("front_count + back_count must be > 0")
        total_needed = (len(files) // chunk) * chunk

    main: list[PlanItem] = []
    extras: list[PlanItem] = []

    cam_name = camera_dir.name
    full_len = min(int(total_needed), len(files))

    for i, src in enumerate(files):
        if i < full_len:
            if plant_blocks:
                # Determine which block this index falls into
                offset = i
                plant_idx = None
                within = None
                fcnt = None
                bcnt = None
                plant_base = 0
                for blk in plant_blocks:
                    blk_total = blk.total_images
                    if offset < blk_total:
                        blk_chunk = blk.chunk
                        plant_in_block = (offset // blk_chunk) + 1
                        plant_idx = plant_base + plant_in_block
                        within = offset % blk_chunk
                        fcnt = blk.front_count
                        bcnt = blk.back_count
                        break
                    offset -= blk_total
                    plant_base += blk.num_plants
                if plant_idx is None or within is None or fcnt is None or bcnt is None:
                    # Shouldn't happen because i < full_len <= total_needed
                    dst = out_date_dir / "_extras" / cam_name / src.name
                    extras.append(PlanItem(src=src, dst=dst))
                    continue
                side = "front" if within < fcnt else "back"
            else:
                chunk = front_count + back_count
                plant_idx = (i // chunk) + 1
                within = i % chunk
                side = "front" if within < front_count else "back"

            plant_dir = out_date_dir / plant_dir_name(plant_prefix, plant_idx, plant_padding)
            dst = plant_dir / cam_name / side / src.name
            main.append(PlanItem(src=src, dst=dst))
            continue

        dst = out_date_dir / "_extras" / cam_name / src.name
        extras.append(PlanItem(src=src, dst=dst))

    return main, extras


def ensure_parent_dir(path: Path, dry_run: bool) -> None:
    parent = path.parent
    if parent.exists():
        return
    if dry_run:
        return
    parent.mkdir(parents=True, exist_ok=True)


def apply_items(
    items: Iterable[PlanItem],
    *,
    mode: Mode,
    on_exists: OnExists,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Returns (copied_or_moved, skipped, overwritten)."""
    done = 0
    skipped = 0
    overwritten = 0

    for it in items:
        if it.dst.exists():
            if on_exists == "skip":
                skipped += 1
                continue
            if on_exists == "error":
                raise SystemExit(f"Destination exists: {it.dst}")
            if on_exists == "overwrite":
                overwritten += 1
                if not dry_run:
                    it.dst.unlink()

        ensure_parent_dir(it.dst, dry_run=dry_run)

        if dry_run:
            done += 1
            continue

        if mode == "copy":
            shutil.copy2(it.src, it.dst)
        else:
            shutil.move(str(it.src), str(it.dst))
        done += 1

    return done, skipped, overwritten


def main() -> None:
    args = parse_args()
    tiff_root = Path(args.tiff_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else tiff_root.parent.resolve()
    date_folder = infer_date_folder(tiff_root, args.date)
    out_date_dir = output_root / date_folder

    camera_dirs = iter_camera_dirs(tiff_root)
    plant_blocks = parse_plant_blocks(args.plant_blocks)

    print(f"TIFF root:      {tiff_root}")
    print(f"Output root:    {output_root}")
    print(f"Date folder:    {date_folder}")
    print(f"Output date dir:{out_date_dir}")
    print(f"Mode:           {args.mode}{' (dry-run)' if args.dry_run else ''}")
    if plant_blocks:
        rule_txt = ", ".join(
            [f"{b.num_plants} plants: {b.front_count}+{b.back_count} (chunk={b.chunk})" for b in plant_blocks]
        )
        total_plants = sum(b.num_plants for b in plant_blocks)
        total_imgs = sum(b.total_images for b in plant_blocks)
        print(f"Rule:           mixed blocks -> {rule_txt}")
        print(f"               total plants={total_plants}, total images/camera={total_imgs}")
    else:
        print(f"Rule:           {args.front_count} front + {args.back_count} back per plant (chunk={args.front_count + args.back_count})")
    print(f"Cameras:        {', '.join([c.name for c in camera_dirs])}")

    total_main = 0
    total_extras = 0
    total_skipped = 0
    total_overwritten = 0

    # Create date directory early (for move mode, we want a stable target)
    if not args.dry_run:
        out_date_dir.mkdir(parents=True, exist_ok=True)

    for cam_dir in camera_dirs:
        main_items, extras_items = build_plan_for_camera(
            camera_dir=cam_dir,
            out_date_dir=out_date_dir,
            front_count=int(args.front_count),
            back_count=int(args.back_count),
            plant_blocks=plant_blocks,
            plant_prefix=str(args.plant_prefix),
            plant_padding=int(args.plant_padding),
        )

        # Full plant chunks count
        if plant_blocks:
            full_plants = sum(b.num_plants for b in plant_blocks)
            chunk = None
        else:
            chunk = int(args.front_count) + int(args.back_count)
            full_plants = len(main_items) // chunk if chunk > 0 else 0
        remainder = len(extras_items)

        print(f"\n{cam_dir.name}: {len(main_items) + len(extras_items)} files")
        if chunk is None:
            print(f"- full plants:  {full_plants} (mixed chunks)")
        else:
            print(f"- full plants:  {full_plants} (each {chunk} images)")
        print(f"- extras:       {remainder}")

        done, skipped, overwritten = apply_items(
            main_items,
            mode=args.mode,
            on_exists=args.on_exists,
            dry_run=bool(args.dry_run),
        )
        e_done, e_skipped, e_overwritten = apply_items(
            extras_items,
            mode=args.mode,
            on_exists=args.on_exists,
            dry_run=bool(args.dry_run),
        )

        total_main += done
        total_extras += e_done
        total_skipped += skipped + e_skipped
        total_overwritten += overwritten + e_overwritten

    print("\nSummary:")
    print(f"- main files processed:   {total_main}")
    print(f"- extras files processed: {total_extras}")
    if args.on_exists != "error":
        print(f"- skipped:                {total_skipped}")
        print(f"- overwritten:            {total_overwritten}")

    if args.dry_run:
        print("\nDry-run only: no files were modified.")


if __name__ == "__main__":
    main()

