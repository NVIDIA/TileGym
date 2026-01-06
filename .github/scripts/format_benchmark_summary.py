#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Parse benchmark results and format them as markdown for GitHub Actions summary.

Reads *_results.txt or *_results.json files and converts to markdown tables.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def parse_benchmark_file(filepath):
    """Parse a benchmark results file and extract tables."""
    with open(filepath, "r") as f:
        content = f.read()

    if content.strip() == "FAILED":
        return None, "FAILED"

    # Split by benchmark sections (lines that end with -TFLOPS: or -GBps:)
    sections = []
    current_section = None
    current_lines = []

    for line in content.split("\n"):
        # Check if this is a section header (benchmark name)
        if line.strip() and (line.endswith("-TFLOPS:") or line.endswith("-GBps:")):
            if current_section:
                sections.append((current_section, "\n".join(current_lines)))
            current_section = line.strip()[:-1]  # Remove trailing ':'
            current_lines = []
        elif line.strip() or current_lines:  # Collect table lines
            current_lines.append(line)

    # Add final section
    if current_section:
        sections.append((current_section, "\n".join(current_lines)))

    return sections, "PASSED"


def table_to_markdown(table_text):
    """Convert pandas-style table to markdown."""
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if not lines:
        return ""

    # First line is the header
    header = lines[0].split()

    # Parse data rows (skip index column)
    data_rows = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            data_rows.append(parts)

    if not data_rows:
        return ""

    # Build markdown table
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"

    for row in data_rows:
        # Align row to header columns (skip first element which is index)
        if len(row) > 1:
            md += "| " + " | ".join(row[1:]) + " |\n"

    return md


def parse_json_results(filepath):
    """Parse JSON benchmark results."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if data.get("status") == "FAILED":
        return None, "FAILED"

    return data.get("benchmarks", []), "PASSED"


def json_benchmark_to_markdown(benchmark_data):
    """Convert JSON benchmark data to markdown table."""
    if not benchmark_data or not benchmark_data.get("configs"):
        return ""

    configs = benchmark_data["configs"]
    if not configs:
        return ""

    # Determine columns: parameter name + all backend names
    param_name = None
    backends = []

    for key in configs[0].keys():
        if key not in ["CuTile", "PyTorch", "Triton", "TorchCompile"]:
            param_name = key
        else:
            backends.append(key)

    if not param_name:
        return ""

    # Build markdown table
    headers = [param_name] + backends
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for config in configs:
        row = [str(config.get(param_name, ""))]
        for backend in backends:
            value = config.get(backend, "")
            if isinstance(value, float):
                row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        md += "| " + " | ".join(row) + " |\n"

    return md


def format_gpu_info(results_dir):
    """Format GPU information section."""
    system_info_file = results_dir / "system_info.json"

    if not system_info_file.exists():
        return ""

    try:
        with open(system_info_file) as f:
            system_info = json.load(f)

        gpu_info = system_info.get("gpu_info", {})
        if not gpu_info.get("available") or not gpu_info.get("gpus"):
            return ""

        summary = "## 🖥️ System Information\n\n"

        for gpu in gpu_info["gpus"]:
            summary += f"**GPU {gpu['index']}:** {gpu['name']}\n"
            summary += f"- **Memory:** {gpu['memory_total_mb']} MB\n"
            summary += f"- **Graphics Clock:** {gpu['clock_graphics_mhz']} MHz\n"
            summary += f"- **SM Clock:** {gpu['clock_sm_mhz']} MHz\n"
            summary += f"- **Memory Clock:** {gpu['clock_memory_mhz']} MHz\n"
            summary += f"- **Driver Version:** {gpu['driver_version']}\n\n"

        return summary
    except Exception as e:
        logger.warning(f"Failed to format GPU info: {e}")
        return ""


def format_benchmark_summary(results_dir):
    """Format all benchmark results as markdown summary."""
    results_dir = Path(results_dir).resolve()  # Get absolute path

    logger.info(f"Looking for results in: {results_dir}")
    logger.info(f"Directory exists: {results_dir.exists()}")

    if not results_dir.exists():
        logger.error("Results directory does not exist")
        return "## Benchmark Results\n\n❌ No benchmark results found (directory does not exist).\n"

    # Find result files - prefer JSON, fall back to TXT
    json_files = sorted(results_dir.glob("*_results.json"))
    txt_files = sorted(results_dir.glob("*_results.txt"))

    result_files = json_files if json_files else txt_files
    file_format = "json" if json_files else "txt"

    logger.info(f"Found {len(result_files)} {file_format} result files")

    if not result_files:
        # List what IS in the directory
        all_files = list(results_dir.glob("*"))
        logger.warning(f"Files in directory: {[f.name for f in all_files]}")
        return "## Benchmark Results\n\n❌ No benchmark results found.\n"

    summary = "# 📊 Benchmark Results\n\n"

    # Add GPU info section if available
    gpu_section = format_gpu_info(results_dir)
    if gpu_section:
        summary += gpu_section

    for result_file in result_files:
        benchmark_name = result_file.stem.replace("_results", "").replace("_", " ").title()
        summary += f"## {benchmark_name}\n\n"

        if file_format == "json":
            benchmarks, status = parse_json_results(result_file)

            if status == "FAILED":
                # Read full data to get error details
                with open(result_file) as f:
                    data = json.load(f)

                summary += "❌ **FAILED**\n\n"

                # Show error type if available
                error_type = data.get("error_type", "Unknown")
                if error_type and error_type != "Unknown":
                    summary += f"**Error Type:** `{error_type}`\n\n"

                # Show error message if available
                error_message = data.get("error_message", "")
                if error_message:
                    summary += f"**Error:** {error_message}\n\n"

                # Show partial results info if available
                partial_results = data.get("partial_results", 0)
                if partial_results > 0:
                    summary += f"**Progress:** {partial_results} configuration(s) completed before failure\n\n"

                # Add expandable section with full error
                full_error = data.get("error", "")
                if full_error:
                    # Limit error output to prevent huge summaries
                    error_preview = full_error[:2000]
                    if len(full_error) > 2000:
                        error_preview += "\n... (truncated)"

                    summary += "<details>\n<summary>📋 View full error output</summary>\n\n```\n"
                    summary += error_preview
                    summary += "\n```\n\n</details>\n\n"

                continue

            if not benchmarks:
                summary += "⚠️ No results captured\n\n"
                continue

            for benchmark_data in benchmarks:
                display_name = benchmark_data["name"].replace("-", " ").replace("_", " ")
                unit = benchmark_data.get("unit", "")
                summary += f"### {display_name}"
                if unit:
                    summary += f" ({unit})"
                summary += "\n\n"

                md_table = json_benchmark_to_markdown(benchmark_data)
                if md_table:
                    summary += md_table + "\n"
                else:
                    summary += "_No data_\n\n"
        else:
            sections, status = parse_benchmark_file(result_file)

            if status == "FAILED":
                summary += "❌ **FAILED**\n\n"
                continue

            if not sections:
                summary += "⚠️ No results captured\n\n"
                continue

            for section_name, table_text in sections:
                # Clean up section name for display
                display_name = section_name.replace("-", " ").replace("_", " ")
                summary += f"### {display_name}\n\n"

                md_table = table_to_markdown(table_text)
                if md_table:
                    summary += md_table + "\n"
                else:
                    summary += "_No data_\n\n"

    return summary


def get_results_directory():
    """Get results directory from command line args."""
    return sys.argv[1] if len(sys.argv) > 1 else "."


def write_summary(summary):
    """Write summary to GitHub Actions or stdout."""
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a") as f:
            f.write(summary)
        logger.info("Benchmark summary written to GitHub Actions summary")
    else:
        # Print to stdout if not in GitHub Actions
        print(summary)


def main():
    results_dir = get_results_directory()
    summary = format_benchmark_summary(results_dir)
    write_summary(summary)


if __name__ == "__main__":
    main()
