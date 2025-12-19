# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Test the neutral zone behavior for benchmark regression checking."""

import json
import sys
import tempfile
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_benchmark_regression import RegressionChecker


def create_sample_results(output_dir, benchmark_values):
    """Create sample benchmark results for testing."""
    results = {
        "timestamp": "2025-01-01T00:00:00Z",
        "benchmarks": [
            {
                "benchmark_file": "bench_test.py",
                "status": "PASSED",
                "benchmarks": [
                    {
                        "name": "test-benchmark",
                        "unit": "GBps",
                        "configs": [
                            {"N": str(val), "CuTile": benchmark_values["cutile"][i]}
                            for i, val in enumerate([1024, 2048, 4096])
                        ],
                    }
                ],
            }
        ],
    }

    combined_file = Path(output_dir) / "all_benchmarks.json"
    with open(combined_file, "w") as f:
        json.dump(results, f)

    return combined_file


def test_neutral_zone_no_update():
    """Test that neutral zone doesn't trigger baseline update."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        # Baseline: 100 GB/s
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0]})

        # Current: +3% improvement (within neutral zone)
        create_sample_results(tmpdir_current, {"cutile": [103.0, 206.0, 309.0]})

        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        # Should pass (no regressions) but not update baseline (no significant improvements)
        assert result["no_regressions"]
        assert not result["has_improvements"]
        assert len(checker.regressions) == 0
        assert len(checker.improvements) == 0
        assert len(checker.neutral) > 0

        # Check report JSON
        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        with open(output_file) as f:
            report = json.load(f)

        assert report["summary"]["should_update_baseline"] == False


def test_improvement_zone_updates():
    """Test that improvement zone triggers baseline update."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        # Baseline: 100 GB/s
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0]})

        # Current: +10% improvement (beyond neutral zone)
        create_sample_results(tmpdir_current, {"cutile": [110.0, 220.0, 330.0]})

        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        # Should pass and update baseline
        assert result["no_regressions"]
        assert result["has_improvements"]
        assert len(checker.regressions) == 0
        assert len(checker.improvements) > 0

        # Check report JSON
        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        with open(output_file) as f:
            report = json.load(f)

        assert report["summary"]["should_update_baseline"] == True


def test_regression_zone_fails():
    """Test that regression zone fails build and doesn't update."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        # Baseline: 100 GB/s
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0]})

        # Current: -10% regression (beyond neutral zone)
        create_sample_results(tmpdir_current, {"cutile": [90.0, 180.0, 270.0]})

        checker = RegressionChecker(threshold_pct=5.0, improvement_threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        # Should fail
        assert not result["no_regressions"]
        assert not result["has_improvements"]
        assert len(checker.regressions) > 0

        # Check report JSON
        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        with open(output_file) as f:
            report = json.load(f)

        assert report["summary"]["should_update_baseline"] == False


def test_asymmetric_thresholds():
    """Test different regression and improvement thresholds."""
    with tempfile.TemporaryDirectory() as tmpdir_current, tempfile.TemporaryDirectory() as tmpdir_baseline:
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0]})

        # +3% improvement
        create_sample_results(tmpdir_current, {"cutile": [103.0, 206.0, 309.0]})

        # Strict regression (2%), lenient improvement (10%)
        checker = RegressionChecker(threshold_pct=2.0, improvement_threshold_pct=10.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        result = checker.compare_all(current, baseline)

        # Should pass but not update (improvement not significant enough)
        assert result["no_regressions"]
        assert not result["has_improvements"]
        assert len(checker.neutral) > 0


if __name__ == "__main__":
    test_neutral_zone_no_update()
    test_improvement_zone_updates()
    test_regression_zone_fails()
    test_asymmetric_thresholds()
    print("✅ All neutral zone tests passed!")
