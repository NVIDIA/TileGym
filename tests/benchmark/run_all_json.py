#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Run all benchmarks and save results as JSON files.

This script executes benchmark files, parses their output, and saves structured
results in JSON format for easier processing and regression detection.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


def parse_benchmark_output(output: str) -> List[Dict[str, Any]]:
    """
    Parse benchmark output and extract structured data.

    Triton benchmarks output tables with headers and data rows.
    Example format:
        benchmark-name-GBps:
                N  CuTile  PyTorch
        0    1024   450.2    320.1
        1    2048   890.5    620.3
    """
    results = []
    lines = output.strip().split("\n")

    current_benchmark = None
    current_unit = None
    header_row = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a benchmark header (e.g., "benchmark-name-GBps:")
        if line and (line.endswith("-TFLOPS:") or line.endswith("-GBps:")):
            # Extract benchmark name and unit
            parts = line[:-1].split("-")  # Remove trailing ':'
            current_unit = parts[-1]  # TFLOPS or GBps
            current_benchmark = "-".join(parts[:-1])  # Everything before unit

            # Next line should be header
            i += 1
            if i < len(lines):
                header_line = lines[i].strip()
                header_row = header_line.split()
                i += 1

                # Following lines are data rows until we hit empty line or next benchmark
                data_rows = []
                while i < len(lines):
                    data_line = lines[i].strip()
                    if not data_line or data_line.endswith("-TFLOPS:") or data_line.endswith("-GBps:"):
                        break
                    data_rows.append(data_line.split())
                    i += 1

                # Structure the data
                if header_row and data_rows:
                    benchmark_data = {
                        "name": current_benchmark,
                        "unit": current_unit,
                        "configs": [],
                    }

                    # header_row[0] is typically the parameter name (e.g., 'N', 'SEQ_LEN')
                    # header_row[1:] are the backend names (e.g., 'CuTile', 'PyTorch')
                    param_name = header_row[0] if header_row else "param"
                    backends = header_row[1:] if len(header_row) > 1 else []

                    for row in data_rows:
                        if len(row) >= 2:
                            config = {
                                param_name: row[1],  # row[0] is index, row[1] is param value
                            }

                            # Add backend results
                            for idx, backend_name in enumerate(backends):
                                if idx + 2 < len(row):  # +2 to skip index and param
                                    try:
                                        value = float(row[idx + 2])
                                        config[backend_name] = value
                                    except ValueError:
                                        config[backend_name] = row[idx + 2]

                            benchmark_data["configs"].append(config)

                    results.append(benchmark_data)
                continue

        i += 1

    return results


def run_benchmark(benchmark_file: Path) -> Dict[str, Any]:
    """Run a single benchmark file and return structured results."""
    print(f"Running {benchmark_file.name}...")

    try:
        result = subprocess.run(
            [sys.executable, str(benchmark_file)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per benchmark
            cwd=benchmark_file.parent,
        )

        if result.returncode != 0:
            return {
                "benchmark_file": benchmark_file.name,
                "status": "FAILED",
                "error": result.stderr,
                "benchmarks": [],
            }

        # Parse the output
        benchmarks = parse_benchmark_output(result.stdout)

        return {
            "benchmark_file": benchmark_file.name,
            "status": "PASSED",
            "benchmarks": benchmarks,
        }

    except subprocess.TimeoutExpired:
        return {
            "benchmark_file": benchmark_file.name,
            "status": "TIMEOUT",
            "error": "Benchmark exceeded 10 minute timeout",
            "benchmarks": [],
        }
    except Exception as e:
        return {
            "benchmark_file": benchmark_file.name,
            "status": "ERROR",
            "error": str(e),
            "benchmarks": [],
        }


def main():
    # Determine output directory
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find benchmark files
    benchmark_dir = Path(__file__).parent
    benchmark_files = sorted(benchmark_dir.glob("bench_*.py"))

    if not benchmark_files:
        print("Error: No benchmark files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(benchmark_files)} benchmark files")
    print(f"Output directory: {output_dir}\n")

    # Run all benchmarks
    all_results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmarks": [],
    }

    failed = False

    for bench_file in benchmark_files:
        result = run_benchmark(bench_file)
        all_results["benchmarks"].append(result)

        # Save individual result file
        output_file = output_dir / f"{bench_file.stem}_results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        # Print status
        status_symbol = "✓" if result["status"] == "PASSED" else "✗"
        print(f"{status_symbol} {result['status']}: {bench_file.name}")
        if result["status"] != "PASSED":
            failed = True
            print(f"  Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"  Results saved to: {output_file}")
        print()

    # Save combined results
    combined_file = output_dir / "all_benchmarks.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 60)
    print("All benchmarks complete!")
    print(f"Combined results: {combined_file}")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
