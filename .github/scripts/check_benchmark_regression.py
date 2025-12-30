#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Check benchmark results for performance regressions.

Compares current benchmark results against a baseline and fails if performance
drops below a configurable threshold.

Usage:
    python check_benchmark_regression.py --current <dir> --baseline <dir> [--threshold <pct>]
    python check_benchmark_regression.py --current <dir> [--threshold <pct>]  # No baseline, just report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class RegressionChecker:
    """Check for performance regressions in benchmark results."""

    def __init__(self, threshold_pct: float = 5.0, improvement_threshold_pct: float = None):
        """
        Initialize regression checker.

        Args:
            threshold_pct: Percentage drop that triggers a regression (default: 5%)
            improvement_threshold_pct: Percentage gain required to be considered an improvement
                                      (default: same as threshold_pct for symmetric behavior)
        """
        self.threshold_pct = threshold_pct
        self.improvement_threshold_pct = (
            improvement_threshold_pct if improvement_threshold_pct is not None else threshold_pct
        )
        self.regressions = []
        self.improvements = []
        self.neutral = []
        self.no_baseline = []
        # Track which benchmark files have regressions for selective updates
        self.benchmark_files_with_regressions = set()
        self.benchmark_files_checked = set()

    def load_results(self, results_dir: Path) -> Dict:
        """Load all benchmark results from a directory."""
        results_dir = results_dir.resolve()

        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Load combined results if available
        combined_file = results_dir / "all_benchmarks.json"
        if combined_file.exists():
            with open(combined_file) as f:
                return json.load(f)

        # Otherwise, load individual files
        results = {"benchmarks": []}
        for json_file in sorted(results_dir.glob("*_results.json")):
            with open(json_file) as f:
                results["benchmarks"].append(json.load(f))

        return results

    def find_matching_config(
        self, baseline_configs: List[Dict], current_config: Dict, param_name: str
    ) -> Optional[Dict]:
        """Find a matching configuration in baseline results."""
        current_param = current_config.get(param_name)
        for baseline_config in baseline_configs:
            if baseline_config.get(param_name) == current_param:
                return baseline_config
        return None

    def compare_benchmark(self, current: Dict, baseline: Optional[Dict]) -> List[Dict]:
        """
        Compare a single benchmark against baseline.

        Returns list of regression/improvement detections.
        """
        comparisons = []

        if baseline is None:
            self.no_baseline.append(current["name"])
            return comparisons

        # Get configs
        current_configs = current.get("configs", [])
        baseline_configs = baseline.get("configs", [])

        if not current_configs:
            return comparisons

        # Determine parameter name (e.g., 'N', 'SEQ_LEN', etc.)
        param_name = None
        for key in current_configs[0].keys():
            if key not in ["CuTile", "PyTorch", "Triton", "TorchCompile"]:
                param_name = key
                break

        if param_name is None:
            return comparisons

        # Compare each configuration
        for current_config in current_configs:
            baseline_config = self.find_matching_config(baseline_configs, current_config, param_name)

            if baseline_config is None:
                continue

            # Compare each backend
            for backend in current_config.keys():
                if backend == param_name:
                    continue

                if backend not in baseline_config:
                    continue

                try:
                    current_val = float(current_config[backend])
                    baseline_val = float(baseline_config[backend])
                except (ValueError, TypeError):
                    continue

                # Calculate percentage change
                if baseline_val > 0:
                    pct_change = ((current_val - baseline_val) / baseline_val) * 100
                else:
                    continue

                comparison = {
                    "benchmark": current["name"],
                    "unit": current.get("unit", "unknown"),
                    "param": param_name,
                    "param_value": current_config[param_name],
                    "backend": backend,
                    "baseline": baseline_val,
                    "current": current_val,
                    "change_pct": pct_change,
                }

                # Classify into three zones: regression, neutral, or improvement
                if pct_change < -self.threshold_pct:
                    comparison["type"] = "regression"
                    self.regressions.append(comparison)
                elif pct_change > self.improvement_threshold_pct:
                    comparison["type"] = "improvement"
                    self.improvements.append(comparison)
                else:
                    comparison["type"] = "neutral"
                    self.neutral.append(comparison)

                comparisons.append(comparison)

        return comparisons

    def compare_all(self, current_results: Dict, baseline_results: Optional[Dict]) -> Dict[str, bool]:
        """
        Compare all benchmarks.

        Returns dict with:
            - 'no_regressions': True if no regressions found
            - 'has_improvements': True if significant improvements found
        """
        # Track which benchmark files contain which benchmarks (for selective updates)
        current_benchmark_to_file = {}
        for bench_file in current_results.get("benchmarks", []):
            file_name = bench_file.get("benchmark_file", "unknown")
            self.benchmark_files_checked.add(file_name)
            for bench in bench_file.get("benchmarks", []):
                current_benchmark_to_file[bench["name"]] = file_name

        current_benchmarks = {}
        for bench_file in current_results.get("benchmarks", []):
            for bench in bench_file.get("benchmarks", []):
                current_benchmarks[bench["name"]] = bench

        baseline_benchmarks = {}
        if baseline_results:
            for bench_file in baseline_results.get("benchmarks", []):
                for bench in bench_file.get("benchmarks", []):
                    baseline_benchmarks[bench["name"]] = bench

        # Compare each benchmark and track files with regressions
        for name, current_bench in current_benchmarks.items():
            baseline_bench = baseline_benchmarks.get(name)
            regressions_before = len(self.regressions)
            self.compare_benchmark(current_bench, baseline_bench)

            # If new regressions were found, mark this benchmark file
            if len(self.regressions) > regressions_before:
                file_name = current_benchmark_to_file.get(name)
                if file_name:
                    self.benchmark_files_with_regressions.add(file_name)

        return {
            "no_regressions": len(self.regressions) == 0,
            "has_improvements": len(self.improvements) > 0,
        }

    def print_report(self):
        """Print a human-readable report."""
        print("\n" + "=" * 80)
        print("BENCHMARK REGRESSION CHECK")
        print("=" * 80)
        print(f"Regression threshold: -{self.threshold_pct}% (fail if worse)")
        print(f"Improvement threshold: +{self.improvement_threshold_pct}% (update baseline if better)")
        print(f"Neutral zone: ±{self.threshold_pct}% (pass but don't update baseline)\n")

        if self.regressions:
            print(f"❌ FOUND {len(self.regressions)} REGRESSION(S):\n")
            for reg in self.regressions:
                print(f"  • {reg['benchmark']} ({reg['unit']})")
                print(f"    Backend: {reg['backend']}, {reg['param']}={reg['param_value']}")
                print(f"    Baseline: {reg['baseline']:.2f} → Current: {reg['current']:.2f}")
                print(f"    Change: {reg['change_pct']:.1f}%")
                print()
        else:
            print("✅ No regressions detected\n")

        if self.improvements:
            print(f"🎉 FOUND {len(self.improvements)} SIGNIFICANT IMPROVEMENT(S):\n")
            for imp in self.improvements:
                print(f"  • {imp['benchmark']} ({imp['unit']})")
                print(f"    Backend: {imp['backend']}, {imp['param']}={imp['param_value']}")
                print(f"    Baseline: {imp['baseline']:.2f} → Current: {imp['current']:.2f}")
                print(f"    Change: +{imp['change_pct']:.1f}%")
                print()

        if self.neutral:
            print(f"🟡 FOUND {len(self.neutral)} NEUTRAL CHANGE(S) (within threshold):\n")
            # Show a few examples
            for neut in self.neutral[:5]:
                print(f"  • {neut['benchmark']} ({neut['unit']})")
                print(f"    Backend: {neut['backend']}, {neut['param']}={neut['param_value']}")
                print(f"    Baseline: {neut['baseline']:.2f} → Current: {neut['current']:.2f}")
                print(f"    Change: {neut['change_pct']:+.1f}%")
                print()
            if len(self.neutral) > 5:
                print(f"  ... and {len(self.neutral) - 5} more\n")

        if self.no_baseline:
            print(f"ℹ️  {len(self.no_baseline)} benchmark(s) with no baseline for comparison:")
            for name in self.no_baseline:
                print(f"  • {name}")
            print()

        # Summary recommendation
        print("=" * 80)
        if self.regressions:
            print("❌ RECOMMENDATION: Fix regressions before updating baseline")
        elif self.improvements:
            print("✅ RECOMMENDATION: Update baseline with improved results")
        else:
            print("🟡 RECOMMENDATION: Keep current baseline (no significant changes)")
        print("=" * 80)

    def save_report(self, output_file: Path):
        """Save detailed report as JSON."""
        # Determine which benchmark files can be updated (those without regressions)
        files_safe_to_update = sorted(self.benchmark_files_checked - self.benchmark_files_with_regressions)

        report = {
            "threshold_pct": self.threshold_pct,
            "improvement_threshold_pct": self.improvement_threshold_pct,
            "summary": {
                "regressions": len(self.regressions),
                "improvements": len(self.improvements),
                "neutral": len(self.neutral),
                "no_baseline": len(self.no_baseline),
                # Update baseline only if there are no regressions and improvements exist
                # OR if this is the first run (no baseline yet)
                "should_update_baseline": len(self.regressions) == 0 and len(self.improvements) > 0,
                # NEW: Per-benchmark update eligibility
                "total_benchmark_files": len(self.benchmark_files_checked),
                "files_with_regressions": len(self.benchmark_files_with_regressions),
                "files_safe_to_update": len(files_safe_to_update),
                "can_do_partial_update": len(files_safe_to_update) > 0
                and len(self.benchmark_files_with_regressions) > 0,
            },
            "regressions": self.regressions,
            "improvements": self.improvements,
            "neutral": self.neutral,
            "no_baseline": self.no_baseline,
            # NEW: Lists of which files to update
            "benchmark_files": {
                "all_files": sorted(self.benchmark_files_checked),
                "files_with_regressions": sorted(self.benchmark_files_with_regressions),
                "files_safe_to_update": files_safe_to_update,
            },
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check benchmark results for performance regressions")
    parser.add_argument(
        "--current", type=Path, required=True, help="Directory containing current benchmark results (JSON)"
    )
    parser.add_argument("--baseline", type=Path, help="Directory containing baseline benchmark results (JSON)")
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="Regression threshold as percentage drop (default: 5.0)"
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=None,
        help="Improvement threshold as percentage gain (default: same as --threshold for symmetric behavior)",
    )
    parser.add_argument("--output", type=Path, help="Output file for detailed JSON report")
    parser.add_argument(
        "--fail-on-regression", action="store_true", help="Exit with error code if regressions are found"
    )
    return parser.parse_args()


def load_results(checker: RegressionChecker, current_path: Path, baseline_path: Path = None):
    """Load current and baseline results."""
    print(f"Loading current results from: {current_path}")
    current = checker.load_results(current_path)

    baseline = None
    if baseline_path:
        print(f"Loading baseline results from: {baseline_path}")
        baseline = checker.load_results(baseline_path)
    else:
        print("No baseline specified - will report current results only")

    return current, baseline


def main():
    args = parse_arguments()
    checker = RegressionChecker(threshold_pct=args.threshold, improvement_threshold_pct=args.improvement_threshold)
    current, baseline = load_results(checker, args.current, args.baseline)
    result = checker.compare_all(current, baseline)

    checker.print_report()
    if args.output:
        checker.save_report(args.output)

    sys.exit(1 if args.fail_on_regression and not result["no_regressions"] else 0)


if __name__ == "__main__":
    main()
