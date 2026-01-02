#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Test selective baseline merging functionality."""

import json
import sys
import tempfile
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_benchmark_regression import RegressionChecker


def create_benchmark_result(benchmark_name, benchmark_file, values):
    """Create a benchmark result JSON structure."""
    return {
        "benchmark_file": benchmark_file,
        "status": "PASSED",
        "benchmarks": [
            {
                "name": benchmark_name,
                "unit": "GBps",
                "configs": [
                    {"N": "1024", "CuTile": values[0]},
                    {"N": "2048", "CuTile": values[1]},
                    {"N": "4096", "CuTile": values[2]},
                ],
            }
        ],
    }


def create_combined_results(output_dir, benchmark_data):
    """Create combined benchmark results with multiple benchmark files."""
    results = {"timestamp": "2025-01-01T00:00:00Z", "benchmarks": []}

    for bench_file, (bench_name, values) in benchmark_data.items():
        bench_result = create_benchmark_result(bench_name, bench_file, values)
        results["benchmarks"].append(bench_result)

        # Also save individual file
        individual_file = Path(output_dir) / bench_file
        with open(individual_file, "w") as f:
            json.dump(bench_result, f)

    # Save combined file
    combined_file = Path(output_dir) / "all_benchmarks.json"
    with open(combined_file, "w") as f:
        json.dump(results, f)

    return combined_file


def test_tracks_per_benchmark_regressions():
    """Test that regression checker tracks which benchmark files have regressions."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
    ):
        # Create baseline with 3 benchmarks
        baseline_data = {
            "bench_matmul_results.json": ("matmul-benchmark", [100.0, 200.0, 300.0]),
            "bench_rmsnorm_results.json": ("rmsnorm-benchmark", [50.0, 100.0, 150.0]),
            "bench_softmax_results.json": ("softmax-benchmark", [80.0, 160.0, 240.0]),
        }
        create_combined_results(tmpdir_baseline, baseline_data)

        # Current: matmul improved, rmsnorm neutral, softmax regressed
        current_data = {
            "bench_matmul_results.json": ("matmul-benchmark", [120.0, 240.0, 360.0]),  # +20%
            "bench_rmsnorm_results.json": ("rmsnorm-benchmark", [52.0, 102.0, 153.0]),  # +3%
            "bench_softmax_results.json": ("softmax-benchmark", [70.0, 140.0, 210.0]),  # -12%
        }
        create_combined_results(tmpdir_current, current_data)

        # Run regression check
        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        # Should detect regression
        assert not result["no_regressions"]
        assert len(checker.regressions) > 0

        # Should track which files have regressions
        assert "bench_softmax_results.json" in checker.benchmark_files_with_regressions
        assert "bench_matmul_results.json" not in checker.benchmark_files_with_regressions
        assert "bench_rmsnorm_results.json" not in checker.benchmark_files_with_regressions

        print("✅ Per-benchmark regression tracking works")


def test_report_includes_file_lists():
    """Test that regression report includes lists of files for selective updates."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        # Create scenario: 2 benchmarks, 1 regressed, 1 improved
        baseline_data = {
            "bench_good_results.json": ("good-benchmark", [100.0, 200.0, 300.0]),
            "bench_bad_results.json": ("bad-benchmark", [100.0, 200.0, 300.0]),
        }
        create_combined_results(tmpdir_baseline, baseline_data)

        current_data = {
            "bench_good_results.json": ("good-benchmark", [120.0, 240.0, 360.0]),  # +20%
            "bench_bad_results.json": ("bad-benchmark", [85.0, 170.0, 255.0]),  # -15%
        }
        create_combined_results(tmpdir_current, current_data)

        # Run regression check and save report
        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        checker.compare_all(current, baseline)

        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        # Load and verify report
        with open(output_file) as f:
            report = json.load(f)

        assert "benchmark_files" in report
        assert report["summary"]["total_benchmark_files"] == 2
        assert report["summary"]["files_with_regressions"] == 1
        assert report["summary"]["files_safe_to_update"] == 1
        assert report["summary"]["can_do_partial_update"] == True

        assert "bench_bad_results.json" in report["benchmark_files"]["files_with_regressions"]
        assert "bench_good_results.json" in report["benchmark_files"]["files_safe_to_update"]

        print("✅ Report includes file lists for selective updates")


def test_all_benchmarks_improve():
    """Test that when all benchmarks improve, all files are safe to update."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        baseline_data = {
            "bench_a_results.json": ("bench-a", [100.0, 200.0, 300.0]),
            "bench_b_results.json": ("bench-b", [50.0, 100.0, 150.0]),
        }
        create_combined_results(tmpdir_baseline, baseline_data)

        # All improve by 10%
        current_data = {
            "bench_a_results.json": ("bench-a", [110.0, 220.0, 330.0]),
            "bench_b_results.json": ("bench-b", [55.0, 110.0, 165.0]),
        }
        create_combined_results(tmpdir_current, current_data)

        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        with open(output_file) as f:
            report = json.load(f)

        assert result["no_regressions"]
        assert result["has_improvements"]
        assert report["summary"]["files_with_regressions"] == 0
        assert report["summary"]["files_safe_to_update"] == 2
        assert report["summary"]["can_do_partial_update"] == False  # No regressions, so not a "partial" update
        assert report["summary"]["should_update_baseline"] == True

        print("✅ All benchmarks improve → all files safe to update")


def test_all_benchmarks_regress():
    """Test that when all benchmarks regress, no files are safe to update."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        baseline_data = {
            "bench_a_results.json": ("bench-a", [100.0, 200.0, 300.0]),
            "bench_b_results.json": ("bench-b", [50.0, 100.0, 150.0]),
        }
        create_combined_results(tmpdir_baseline, baseline_data)

        # All regress by 10%
        current_data = {
            "bench_a_results.json": ("bench-a", [90.0, 180.0, 270.0]),
            "bench_b_results.json": ("bench-b", [45.0, 90.0, 135.0]),
        }
        create_combined_results(tmpdir_current, current_data)

        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        with open(output_file) as f:
            report = json.load(f)

        assert not result["no_regressions"]
        assert report["summary"]["files_with_regressions"] == 2
        assert report["summary"]["files_safe_to_update"] == 0
        assert report["summary"]["should_update_baseline"] == False

        print("✅ All benchmarks regress → no files safe to update")


if __name__ == "__main__":
    test_tracks_per_benchmark_regressions()
    test_report_includes_file_lists()
    test_all_benchmarks_improve()
    test_all_benchmarks_regress()
    print()
    print("=" * 70)
    print("✅ All selective baseline merge tests passed!")
    print("=" * 70)
