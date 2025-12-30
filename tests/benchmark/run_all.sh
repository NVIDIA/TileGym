#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run all Python benchmark files and save results
# Usage: ./run_all.sh [OUTPUT_DIR] [--json]

cd "$(dirname "$0")"

OUTPUT_DIR="${1:-.}"
FORMAT="txt"

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--json" ]]; then
        FORMAT="json"
    elif [[ -z "$OUTPUT_DIR" ]] || [[ "$OUTPUT_DIR" == "--json" ]]; then
        OUTPUT_DIR="."
    fi
done

# If --json is first argument, reset OUTPUT_DIR
if [[ "$OUTPUT_DIR" == "--json" ]]; then
    OUTPUT_DIR="${2:-.}"
fi

mkdir -p "$OUTPUT_DIR"

echo "Running benchmarks sequentially (parallel execution disabled to ensure accurate results)..."
echo "Output format: $FORMAT"
echo "Results will be saved to: $OUTPUT_DIR"
echo "Current directory: $(pwd)"
echo "Benchmark files found: $(ls bench_*.py 2>/dev/null | wc -l)"
echo ""

# Check if output directory is writable
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory $OUTPUT_DIR is not writable" >&2
    exit 1
fi

# Use JSON runner if --json flag is set
if [[ "$FORMAT" == "json" ]]; then
    echo "Using JSON output format..."
    if python run_all_json.py "$OUTPUT_DIR"; then
        echo ""
        echo "=========================================="
        echo "All benchmarks complete!"
        echo "Results directory: $OUTPUT_DIR"
        echo "Files created:"
        ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  No result files found"
        echo "=========================================="
        exit 0
    else
        echo "Benchmark execution failed" >&2
        exit 1
    fi
fi

# Original text format runner
for file in bench_*.py; do
    if [[ ! -f "$file" ]]; then
        echo "Warning: No benchmark files matching bench_*.py found" >&2
        continue
    fi

    benchmark_name=$(basename "$file" .py)
    output_file="$OUTPUT_DIR/${benchmark_name}_results.txt"

    echo "=========================================="
    echo "Running $file..."
    echo "=========================================="

    # Ensure output file is created even if benchmark produces no output
    touch "$output_file"

    if python "$file" 2>&1 | tee "$output_file"; then
        echo "✓ PASSED: $file"
        echo "  Results saved to: $output_file"
    else
        echo "✗ FAILED: $file"
        echo "FAILED" > "$output_file"
        exit 1  # Exit with error if any benchmark fails
    fi
    echo ""
done

echo "=========================================="
echo "All benchmarks complete!"
echo "Results directory: $OUTPUT_DIR"
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*_results.txt 2>/dev/null || echo "  No result files found"
echo "=========================================="
