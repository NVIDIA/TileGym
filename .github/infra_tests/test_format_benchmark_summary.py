# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unit tests for format_benchmark_summary.py"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import format_benchmark_summary


class TestFormatBenchmarkSummary:
    """Tests for format_benchmark_summary.py"""

    def test_parse_failed_benchmark(self):
        """Test parsing a failed benchmark file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="_results.txt", delete=False) as f:
            f.write("FAILED")
            f.flush()

            sections, status = format_benchmark_summary.parse_benchmark_file(f.name)

            assert status == "FAILED"
            assert sections is None

        os.unlink(f.name)

    def test_parse_benchmark_with_tables(self):
        """Test parsing benchmark file with performance tables."""
        content = """Running bench_fused_attention.py...

fused-attention-batch4-head32-d128-fwd-causal=True-float16-TFLOPS:
     N_CTX      CuTile     PyTorch
0   1024.0  261.332268  230.121244
1   2048.0  298.551774  273.570558

fused-attention-batch4-head32-d128-fwd-causal=False-float8_e5m2-TFLOPS:
     N_CTX      CuTile
0   1024.0  464.326850
1   2048.0  540.122687
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix="_results.txt", delete=False) as f:
            f.write(content)
            f.flush()

            sections, status = format_benchmark_summary.parse_benchmark_file(f.name)

            assert status == "PASSED"
            assert sections is not None
            assert len(sections) == 2
            assert sections[0][0] == "fused-attention-batch4-head32-d128-fwd-causal=True-float16-TFLOPS"
            assert sections[1][0] == "fused-attention-batch4-head32-d128-fwd-causal=False-float8_e5m2-TFLOPS"

        os.unlink(f.name)

    def test_table_to_markdown(self):
        """Test converting pandas-style table to markdown."""
        table_text = """     N_CTX      CuTile     PyTorch
0   1024.0  261.332268  230.121244
1   2048.0  298.551774  273.570558"""

        md = format_benchmark_summary.table_to_markdown(table_text)

        assert "| N_CTX | CuTile | PyTorch |" in md
        assert "| --- | --- | --- |" in md
        assert "| 1024.0 | 261.332268 | 230.121244 |" in md
        assert "| 2048.0 | 298.551774 | 273.570558 |" in md

    def test_table_to_markdown_empty(self):
        """Test converting empty table."""
        md = format_benchmark_summary.table_to_markdown("")
        assert md == ""

    def test_format_benchmark_summary_no_results(self):
        """Test formatting when no results directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = Path(tmpdir) / "nonexistent"
            summary = format_benchmark_summary.format_benchmark_summary(non_existent)

            assert "No benchmark results found" in summary

    def test_format_benchmark_summary_with_results(self):
        """Test formatting with actual result files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a result file
            result_file = Path(tmpdir) / "bench_example_results.txt"
            result_file.write_text(
                """example-benchmark-TFLOPS:
     N_CTX      CuTile
0   1024.0  261.332268
1   2048.0  298.551774
"""
            )

            summary = format_benchmark_summary.format_benchmark_summary(tmpdir)

            assert "# 📊 Benchmark Results" in summary
            assert "## Bench Example" in summary
            assert "| N_CTX | CuTile |" in summary
            assert "| 1024.0 | 261.332268 |" in summary

    def test_format_benchmark_summary_with_failed_benchmark(self):
        """Test formatting with a failed benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = Path(tmpdir) / "bench_failed_results.txt"
            result_file.write_text("FAILED")

            summary = format_benchmark_summary.format_benchmark_summary(tmpdir)

            assert "# 📊 Benchmark Results" in summary
            assert "## Bench Failed" in summary
            assert "❌ **FAILED**" in summary

    def test_main_writes_to_github_summary(self):
        """Test that main() writes to GITHUB_STEP_SUMMARY."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a result file
            result_file = Path(tmpdir) / "bench_test_results.txt"
            result_file.write_text(
                """test-benchmark-TFLOPS:
     N_CTX      CuTile
0   1024.0  100.0
"""
            )

            # Create temp file for GitHub summary
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as summary_file:
                summary_path = summary_file.name

            try:
                with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": summary_path}):
                    with patch.object(sys, "argv", ["format_benchmark_summary.py", tmpdir]):
                        format_benchmark_summary.main()

                # Read the summary file
                with open(summary_path) as f:
                    content = f.read()

                assert "# 📊 Benchmark Results" in content
                assert "## Bench Test" in content
                assert "| 1024.0 | 100.0 |" in content
            finally:
                os.unlink(summary_path)

    def test_main_prints_without_github_env(self):
        """Test that main() prints to stdout when not in GitHub Actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = Path(tmpdir) / "bench_test_results.txt"
            result_file.write_text(
                """test-TFLOPS:
     N_CTX      CuTile
0   1024.0  100.0
"""
            )

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(sys, "argv", ["format_benchmark_summary.py", tmpdir]):
                    # Should not raise, just print
                    format_benchmark_summary.main()

    def test_get_results_directory_default(self):
        """Test get_results_directory with no args."""
        with patch.object(sys, "argv", ["format_benchmark_summary.py"]):
            result_dir = format_benchmark_summary.get_results_directory()
            assert result_dir == "."

    def test_get_results_directory_with_arg(self):
        """Test get_results_directory with command line arg."""
        with patch.object(sys, "argv", ["format_benchmark_summary.py", "/custom/path"]):
            result_dir = format_benchmark_summary.get_results_directory()
            assert result_dir == "/custom/path"

    def test_write_summary_to_github(self):
        """Test write_summary writes to GitHub Actions summary file."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            summary_path = f.name

        try:
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": summary_path}):
                format_benchmark_summary.write_summary("Test summary content")

            with open(summary_path) as f:
                content = f.read()

            assert content == "Test summary content"
        finally:
            os.unlink(summary_path)

    def test_write_summary_to_stdout(self):
        """Test write_summary prints to stdout when not in GitHub Actions."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise, just print
            format_benchmark_summary.write_summary("Test content")
