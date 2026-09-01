"""
Command-line entry point for the downstream analysis stage.

Runs after ``python main.py --config config.yaml`` has populated an
``output_folder`` with per-plant feature JSON files. Collects those into a
single tidy feature matrix and, optionally, runs one of the two paper case
studies against it.

Examples
--------
Just build the feature matrix CSV::

    python -m analysis.run_analysis --output-folder output/sorghum_run \\
        --save-features analysis_out/features.csv

Run the sorghum treatment-level case study (Section 2.7 / 3.8.1)::

    python -m analysis.run_analysis --output-folder output/sorghum_run \\
        --case-study sorghum --design sorghum_design.csv \\
        --control-label NT --results-dir analysis_out/sorghum

``design.csv`` must have a ``plant_id`` column matching the plant folder names
under ``output_folder``, plus a ``group`` column (e.g. G1..G7, NT).

Run the maize cold-stress PCA (Section 2.7 / 3.8.2)::

    python -m analysis.run_analysis --features maize_features.csv \\
        --case-study maize --design maize_design.csv \\
        --results-dir analysis_out/maize

``maize_features.csv`` has one row per image (``image_id``) with RGB vegetation
-index feature columns (see ``analysis.collect`` for the flattening convention
if you are instead pointing ``--output-folder`` at a maize pipeline run).
"""

import argparse
import logging
import sys

import pandas as pd

from .case_studies import run_maize_coldstress_analysis, run_sorghum_treatment_analysis
from .collect import collect_feature_matrix

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Downstream temporal/statistical analysis (Sec. 2.6-2.7)")
    p.add_argument("--output-folder", type=str, default=None,
                    help="PlantPipeline output_folder to collect features from")
    p.add_argument("--features", type=str, default=None,
                    help="Pre-built tidy feature-matrix CSV (skips --output-folder collection)")
    p.add_argument("--save-features", type=str, default=None,
                    help="Where to save the collected/loaded feature matrix CSV")
    p.add_argument("--case-study", choices=["sorghum", "maize"], default=None,
                    help="Which paper case study to run (Section 3.8)")
    p.add_argument("--design", type=str, default=None,
                    help="CSV mapping plant_id/image_id -> group/genotype/stage columns")
    p.add_argument("--control-label", type=str, default="NT",
                    help="Non-treated control group label (sorghum case study)")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold")
    p.add_argument("--results-dir", type=str, default=None, help="Where to write result tables/plots")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.features:
        features = pd.read_csv(args.features)
    elif args.output_folder:
        features = collect_feature_matrix(args.output_folder, save_csv=args.save_features)
    else:
        logger.error("Provide either --features or --output-folder")
        return 1

    if args.save_features and args.features:
        features.to_csv(args.save_features, index=False)

    if not args.case_study:
        logger.info("Feature matrix ready: %d rows x %d columns", *features.shape)
        return 0

    if not args.design:
        logger.error("--case-study requires --design")
        return 1
    design = pd.read_csv(args.design)

    if args.case_study == "sorghum":
        run_sorghum_treatment_analysis(
            features, design, output_dir=args.results_dir,
            control_label=args.control_label, alpha=args.alpha,
        )
    else:
        run_maize_coldstress_analysis(features, design, output_dir=args.results_dir)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
