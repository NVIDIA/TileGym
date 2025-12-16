#!/usr/bin/env python3
"""
Parse benchmark results and format them as markdown for GitHub Actions summary.

Reads *_results.json files and converts to markdown tables.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)


def parse_benchmark_json(filepath):
    """Parse a benchmark JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check for error
    if isinstance(data, dict) and "error" in data:
        return None, "FAILED"
    
    # Expect list of benchmark results
    if not isinstance(data, list):
        return None, "FAILED"
    
    return data, "PASSED"


def json_to_markdown(benchmark_result):
    """Convert a single benchmark result to markdown table."""
    if not benchmark_result.get("data"):
        return ""
    
    columns = benchmark_result.get("columns", [])
    data_rows = benchmark_result["data"]
    
    if not columns or not data_rows:
        return ""
    
    # Build markdown table
    md = "| " + " | ".join(columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    
    for row in data_rows:
        # Convert row dict to list in column order
        row_values = [str(row.get(col, "")) for col in columns]
        md += "| " + " | ".join(row_values) + " |\n"
    
    return md


def format_benchmark_summary(results_dir):
    """Format all benchmark results as markdown summary."""
    results_dir = Path(results_dir).resolve()  # Get absolute path

    logger.info(f"Looking for results in: {results_dir}")
    logger.info(f"Directory exists: {results_dir.exists()}")

    if not results_dir.exists():
        logger.error(f"Results directory does not exist")
        return "## Benchmark Results\n\n❌ No benchmark results found (directory does not exist).\n"

    # Find all JSON result files
    result_files = sorted(results_dir.glob("*_results.json"))
    logger.info(f"Found {len(result_files)} result files")

    if not result_files:
        # List what IS in the directory
        all_files = list(results_dir.glob("*"))
        logger.warning(f"Files in directory: {[f.name for f in all_files]}")
        return "## Benchmark Results\n\n❌ No benchmark results found (no *_results.json files).\n"

    summary = "# 📊 Benchmark Results\n\n"

    for result_file in result_files:
        benchmark_name = result_file.stem.replace('_results', '').replace('_', ' ').title()
        summary += f"## {benchmark_name}\n\n"

        benchmark_results, status = parse_benchmark_json(result_file)

        if status == "FAILED":
            summary += "❌ **FAILED**\n\n"
            continue

        if not benchmark_results:
            summary += "⚠️ No results captured\n\n"
            continue

        for result in benchmark_results:
            # Section name from benchmark + unit
            section_name = result.get("benchmark", "Unknown")
            unit = result.get("unit", "")
            display_name = f"{section_name} ({unit})" if unit else section_name
            display_name = display_name.replace('-', ' ').replace('_', ' ')
            summary += f"### {display_name}\n\n"

            md_table = json_to_markdown(result)
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
