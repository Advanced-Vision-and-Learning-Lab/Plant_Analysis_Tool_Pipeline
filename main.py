"""
End-to-end entry point for the Plant Analysis Tool Pipeline.

Run with:
    python main.py --config config.yaml
    python main.py --input /data/plants --output /results
    python main.py --config config.yaml --all-frames --segmentation-only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Plant Analysis Tool Pipeline — end-to-end plant phenotyping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal run — point at config file
  python main.py --config config.yaml

  # Override input/output paths on the command line
  python main.py --config config.yaml --input /data/plants --output /results

  # Process every frame of every plant
  python main.py --config config.yaml --all-frames

  # Segmentation only (skip feature extraction)
  python main.py --config config.yaml --segmentation-only

  # Process specific plants or frames
  python main.py --config config.yaml --plants plant1 plant2
  python main.py --config config.yaml --frames 8 9

  # Force reprocess even if outputs already exist
  python main.py --config config.yaml --force-reprocess
""",
    )

    parser.add_argument(
        "--config", "-c",
        metavar="PATH",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        metavar="DIR",
        default=None,
        help="Override paths.input_folder from config",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=None,
        help="Override paths.output_folder from config",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        default=False,
        help="Load and process all frames per plant (default: selected frames only)",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        default=False,
        help="Run segmentation only — skip texture, vegetation, and morphology extraction",
    )
    parser.add_argument(
        "--plants",
        nargs="+",
        metavar="PLANT",
        default=None,
        help="Process only these plant names, e.g. --plants plant1 plant2",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        metavar="FRAME",
        default=None,
        help="Process only these frame numbers, e.g. --frames 8 9",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        default=False,
        help="Reprocess even if outputs already exist",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help='Compute device: "cuda", "cpu", or leave blank for auto-detect',
    )
    parser.add_argument(
        "--summary",
        metavar="PATH",
        default=None,
        help="Save a JSON run summary to this path (default: <output>/run_summary.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("main")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        log.error("Create one by copying config.yaml.example and filling in your paths.")
        sys.exit(1)

    log.info(f"Loading config from {config_path}")
    from sorghum_pipeline.config import Config  # noqa: E402

    config = Config(str(config_path))

    # Command-line overrides
    if args.input:
        config.paths.input_folder = str(Path(args.input).resolve())
        log.info(f"  input_folder overridden → {config.paths.input_folder}")
    if args.output:
        config.paths.output_folder = str(Path(args.output).resolve())
        log.info(f"  output_folder overridden → {config.paths.output_folder}")
    if args.device:
        config.processing.device = args.device
        log.info(f"  device overridden → {args.device}")

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    from sorghum_pipeline.pipeline import SorghumPipeline  # noqa: E402

    log.info("Initialising pipeline …")
    pipeline = SorghumPipeline(config=config)

    log.info("Running pipeline …")
    t0 = time.perf_counter()
    results = pipeline.run(
        load_all_frames=args.all_frames,
        segmentation_only=args.segmentation_only,
        filter_plants=args.plants,
        filter_frames=args.frames,
        force_reprocess=args.force_reprocess,
    )
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    summary = results.get("summary", {})
    total   = summary.get("total_plants", len(results.get("plants", {})))
    success = summary.get("successful", "—")
    failed  = summary.get("failed", "—")

    log.info("=" * 60)
    log.info(f"Pipeline complete in {elapsed:.1f} s")
    log.info(f"  Plants processed : {total}")
    log.info(f"  Successful       : {success}")
    log.info(f"  Failed           : {failed}")
    log.info(f"  Results saved to : {config.paths.output_folder}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary_path = args.summary or str(
        Path(config.paths.output_folder) / "run_summary.json"
    )
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "config":         str(config_path.resolve()),
        "input_folder":   config.paths.input_folder,
        "output_folder":  config.paths.output_folder,
        "total_plants":   total,
        "successful":     success,
        "failed":         failed,
        "elapsed_seconds": round(elapsed, 2),
        "pipeline_summary": summary,
    }
    with open(summary_path, "w") as fh:
        json.dump(run_summary, fh, indent=2, default=str)
    log.info(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
