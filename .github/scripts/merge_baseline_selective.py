#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Selectively merge baseline results with new benchmark results.

This script creates a new baseline by combining:
- Old baseline results for benchmarks that regressed
- New results for benchmarks that improved or stayed neutral

This allows preserving improvements in individual benchmarks while still
maintaining the old baseline for regressed benchmarks (forcing fixes).

Usage:
    python merge_baseline_selective.py \\
        --old-baseline baseline-results/ \\
        --new-results test-results/ \\
        --regression-report test-results/regression_report.json \\
        --output merged-baseline/
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def merge_baselines(
    old_baseline_dir: Path,
    new_results_dir: Path,
    regression_report: Path,
    output_dir: Path,
):
    """
    Merge old baseline with new results based on regression report.

    Args:
        old_baseline_dir: Directory containing previous baseline JSON files
        new_results_dir: Directory containing new benchmark result JSON files
        regression_report: JSON file from check_benchmark_regression.py
        output_dir: Directory to write merged baseline files
    """
    # Load regression report
    with open(regression_report) as f:
        report = json.load(f)

    files_with_regressions = set(report["benchmark_files"]["files_with_regressions"])
    files_safe_to_update = set(report["benchmark_files"]["files_safe_to_update"])

    logger.info("Merging baseline:")
    logger.info(f"   Total benchmark files: {report['summary']['total_benchmark_files']}")
    logger.info(f"   Files with regressions: {len(files_with_regressions)}")
    logger.info(f"   Files safe to update: {len(files_safe_to_update)}")
    logger.info("")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track what we do
    kept_old_regressions = []  # Files kept due to regressions
    kept_old_fallback = []  # Files kept due to missing new results
    used_new = []

    # Process each benchmark file
    all_files = report["benchmark_files"]["all_files"]

    for bench_file in all_files:
        # Determine source (old baseline or new results)
        if bench_file in files_with_regressions:
            # Keep old baseline for regressed benchmarks
            source_file = old_baseline_dir / bench_file
            if source_file.exists():
                shutil.copy2(source_file, output_dir / bench_file)
                kept_old_regressions.append(bench_file)
                logger.warning(f"[KEPT OLD] {bench_file}: Kept OLD baseline (regressions detected)")
            else:
                # No old baseline exists, use new (first run scenario)
                source_file = new_results_dir / bench_file
                if source_file.exists():
                    shutil.copy2(source_file, output_dir / bench_file)
                    used_new.append(bench_file)
                    logger.info(f"[NEW] {bench_file}: Used NEW results (no old baseline)")
        else:
            # Use new results for improved/neutral benchmarks
            source_file = new_results_dir / bench_file
            if source_file.exists():
                shutil.copy2(source_file, output_dir / bench_file)
                used_new.append(bench_file)
                logger.info(f"[UPDATED] {bench_file}: Updated to NEW results")
            else:
                # Fall back to old baseline if new doesn't exist (shouldn't happen)
                source_file = old_baseline_dir / bench_file
                if source_file.exists():
                    shutil.copy2(source_file, output_dir / bench_file)
                    kept_old_fallback.append(bench_file)
                    logger.warning(f"[KEPT OLD] {bench_file}: Kept OLD baseline (new not found)")

    # Copy combined results file if it exists in new results
    combined_new = new_results_dir / "all_benchmarks.json"
    combined_old = old_baseline_dir / "all_benchmarks.json"

    # Rebuild combined file from merged individual files
    if (output_dir / f"{all_files[0]}").exists():  # Check if we have any files
        merged_combined = {
            "timestamp": json.load(open(combined_new))["timestamp"] if combined_new.exists() else "unknown",
            "benchmarks": [],
        }

        for bench_file in sorted(all_files):
            result_file = output_dir / bench_file
            if result_file.exists():
                with open(result_file) as f:
                    merged_combined["benchmarks"].append(json.load(f))

        with open(output_dir / "all_benchmarks.json", "w") as f:
            json.dump(merged_combined, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Merge complete!")
    logger.info(f"   Updated baselines: {len(used_new)} files")
    if kept_old_regressions:
        logger.info(f"   Kept old baselines (regressions): {len(kept_old_regressions)} files")
    if kept_old_fallback:
        logger.info(f"   Kept old baselines (fallback): {len(kept_old_fallback)} files")
    logger.info(f"   Output: {output_dir}")
    logger.info("=" * 70)

    if kept_old_regressions:
        logger.info("")
        logger.warning("Files with regressions that need performance fixes:")
        for f in kept_old_regressions:
            logger.warning(f"   - {f}")

    if kept_old_fallback:
        logger.info("")
        logger.info("Files kept as fallback (new results missing):")
        for f in kept_old_fallback:
            logger.info(f"   - {f}")

    return len(kept_old_regressions) == 0  # True if all were updated (no regressions)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Selectively merge baseline with new benchmark results")
    parser.add_argument(
        "--old-baseline", type=Path, required=True, help="Directory containing previous baseline results"
    )
    parser.add_argument("--new-results", type=Path, required=True, help="Directory containing new benchmark results")
    parser.add_argument(
        "--regression-report",
        type=Path,
        required=True,
        help="JSON regression report from check_benchmark_regression.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for merged baseline")
    return parser.parse_args()


def validate_inputs(old_baseline: Path, new_results: Path, regression_report: Path):
    """Validate that all input paths exist."""
    if not old_baseline.exists():
        logger.error(f"Error: Old baseline directory not found: {old_baseline}")
        sys.exit(1)

    if not new_results.exists():
        logger.error(f"Error: New results directory not found: {new_results}")
        sys.exit(1)

    if not regression_report.exists():
        logger.error(f"Error: Regression report not found: {regression_report}")
        sys.exit(1)


def main():
    args = parse_arguments()
    validate_inputs(args.old_baseline, args.new_results, args.regression_report)
    merge_baselines(args.old_baseline, args.new_results, args.regression_report, args.output)


if __name__ == "__main__":
    main()
