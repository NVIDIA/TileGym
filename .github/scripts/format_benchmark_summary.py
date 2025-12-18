#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Parse benchmark results and format them as markdown for GitHub Actions summary.

Reads *_results.txt files and converts pandas-style tables to markdown.
"""

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


def format_benchmark_summary(results_dir):
    """Format all benchmark results as markdown summary."""
    results_dir = Path(results_dir).resolve()  # Get absolute path

    logger.info(f"Looking for results in: {results_dir}")
    logger.info(f"Directory exists: {results_dir.exists()}")

    if not results_dir.exists():
        logger.error("Results directory does not exist")
        return "## Benchmark Results\n\n❌ No benchmark results found (directory does not exist).\n"

    # Find all result files
    result_files = sorted(results_dir.glob("*_results.txt"))
    logger.info(f"Found {len(result_files)} result files")

    if not result_files:
        # List what IS in the directory
        all_files = list(results_dir.glob("*"))
        logger.warning(f"Files in directory: {[f.name for f in all_files]}")
        return "## Benchmark Results\n\n❌ No benchmark results found (no *_results.txt files).\n"

    summary = "# 📊 Benchmark Results\n\n"

    for result_file in result_files:
        benchmark_name = result_file.stem.replace("_results", "").replace("_", " ").title()
        summary += f"## {benchmark_name}\n\n"

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
