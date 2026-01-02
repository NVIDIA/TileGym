# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unit tests for check_benchmark_regression.py"""

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
                            {
                                "N": str(val),
                                "CuTile": benchmark_values["cutile"][i],
                                "PyTorch": benchmark_values["pytorch"][i],
                            }
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


def test_load_results():
    """Test loading benchmark results from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample results
        create_sample_results(tmpdir, {"cutile": [100.0, 200.0, 300.0], "pytorch": [80.0, 160.0, 240.0]})

        checker = RegressionChecker()
        results = checker.load_results(Path(tmpdir))

        assert "benchmarks" in results
        assert len(results["benchmarks"]) == 1
        assert results["benchmarks"][0]["status"] == "PASSED"


def test_no_regression():
    """Test that no regression is detected when performance is stable."""
    with tempfile.TemporaryDirectory() as tmpdir_current, tempfile.TemporaryDirectory() as tmpdir_baseline:
        # Create baseline and current with same values (within neutral zone)
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0], "pytorch": [80.0, 160.0, 240.0]})
        create_sample_results(
            tmpdir_current,
            {
                "cutile": [102.0, 198.0, 305.0],  # Small variations within ±5%
                "pytorch": [81.0, 159.0, 242.0],
            },
        )

        checker = RegressionChecker(threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))

        result = checker.compare_all(current, baseline)

        assert result["no_regressions"]
        assert not result["has_improvements"]  # No significant improvements
        assert len(checker.regressions) == 0
        assert len(checker.improvements) == 0
        assert len(checker.neutral) > 0  # Should be in neutral zone


def test_detect_regression():
    """Test that regression is detected when performance drops."""
    with tempfile.TemporaryDirectory() as tmpdir_current, tempfile.TemporaryDirectory() as tmpdir_baseline:
        # Create baseline with good performance
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0], "pytorch": [80.0, 160.0, 240.0]})

        # Create current with 10% drop in CuTile
        create_sample_results(
            tmpdir_current,
            {
                "cutile": [90.0, 180.0, 270.0],  # 10% drop
                "pytorch": [80.0, 160.0, 240.0],
            },
        )

        checker = RegressionChecker(threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))

        result = checker.compare_all(current, baseline)

        assert not result["no_regressions"]
        assert len(checker.regressions) > 0

        # Check that CuTile regressions were detected
        cutile_regressions = [r for r in checker.regressions if r["backend"] == "CuTile"]
        assert len(cutile_regressions) == 3  # All three configs regressed


def test_detect_improvement():
    """Test that improvements are detected when performance increases."""
    with tempfile.TemporaryDirectory() as tmpdir_current, tempfile.TemporaryDirectory() as tmpdir_baseline:
        # Create baseline
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0], "pytorch": [80.0, 160.0, 240.0]})

        # Create current with 10% improvement in PyTorch
        create_sample_results(
            tmpdir_current,
            {
                "cutile": [100.0, 200.0, 300.0],
                "pytorch": [88.0, 176.0, 264.0],  # 10% improvement
            },
        )

        checker = RegressionChecker(threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))

        checker.compare_all(current, baseline)

        assert len(checker.improvements) > 0

        # Check that PyTorch improvements were detected
        pytorch_improvements = [i for i in checker.improvements if i["backend"] == "PyTorch"]
        assert len(pytorch_improvements) == 3  # All three configs improved


def test_threshold_configuration():
    """Test that threshold can be configured."""
    with tempfile.TemporaryDirectory() as tmpdir_current, tempfile.TemporaryDirectory() as tmpdir_baseline:
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0], "pytorch": [80.0, 160.0, 240.0]})

        # 3% drop - should not trigger with 5% threshold
        create_sample_results(tmpdir_current, {"cutile": [97.0, 194.0, 291.0], "pytorch": [80.0, 160.0, 240.0]})

        # Test with 5% threshold - should pass (in neutral zone)
        checker_5pct = RegressionChecker(threshold_pct=5.0)
        baseline = checker_5pct.load_results(Path(tmpdir_baseline))
        current = checker_5pct.load_results(Path(tmpdir_current))
        result_5 = checker_5pct.compare_all(current, baseline)

        assert result_5["no_regressions"]
        assert len(checker_5pct.regressions) == 0

        # Test with 2% threshold - should fail
        checker_2pct = RegressionChecker(threshold_pct=2.0)
        result_2 = checker_2pct.compare_all(current, baseline)

        assert not result_2["no_regressions"]
        assert len(checker_2pct.regressions) > 0


def test_save_report():
    """Test saving regression report to JSON."""
    with (
        tempfile.TemporaryDirectory() as tmpdir_current,
        tempfile.TemporaryDirectory() as tmpdir_baseline,
        tempfile.TemporaryDirectory() as tmpdir_output,
    ):
        create_sample_results(tmpdir_baseline, {"cutile": [100.0, 200.0, 300.0], "pytorch": [80.0, 160.0, 240.0]})
        create_sample_results(
            tmpdir_current,
            {
                "cutile": [90.0, 200.0, 320.0],  # One regression (90), one stable (200), one improvement (320)
                "pytorch": [80.0, 160.0, 240.0],
            },
        )

        checker = RegressionChecker(threshold_pct=5.0)
        baseline = checker.load_results(Path(tmpdir_baseline))
        current = checker.load_results(Path(tmpdir_current))
        checker.compare_all(current, baseline)

        output_file = Path(tmpdir_output) / "report.json"
        checker.save_report(output_file)

        assert output_file.exists()

        with open(output_file) as f:
            report = json.load(f)

        assert "threshold_pct" in report
        assert "summary" in report
        assert "regressions" in report
        assert "improvements" in report
        assert report["summary"]["regressions"] > 0
        assert report["summary"]["improvements"] > 0


if __name__ == "__main__":
    test_load_results()
    test_no_regression()
    test_detect_regression()
    test_detect_improvement()
    test_threshold_configuration()
    test_save_report()
    print("✅ All tests passed!")
