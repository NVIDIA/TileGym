#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run all Python benchmark files and save results
# Usage: ./run_all.sh [OUTPUT_DIR] [--json]

# Enable pipefail to catch errors in piped commands
set -o pipefail

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
echo "Benchmark files found: $(find . -name 'bench_*.py' -not -path './__pycache__/*' | wc -l)"
echo ""

# Check if output directory is writable
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory $OUTPUT_DIR is not writable" >&2
    exit 1
fi

# Use JSON runner if --json flag is set
if [[ "$FORMAT" == "json" ]]; then
    echo "Using JSON output format..."
    if python3 run_all_json.py "$OUTPUT_DIR"; then
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
FAILED_BENCHMARKS=()

# Auto-discover all bench_*.py files recursively (sorted for deterministic order).
# Output file names use the relative path with / replaced by _ to avoid collisions
# (e.g., suites/unsloth/bench_swiglu.py → suites_unsloth_bench_swiglu_results.txt).
while IFS= read -r file; do
    # Derive output name from relative path: strip leading ./, replace / with _
    rel_path="${file#./}"
    safe_name="${rel_path//\//_}"
    safe_name="${safe_name%.py}"
    output_file="$OUTPUT_DIR/${safe_name}_results.txt"

    echo "=========================================="
    echo "Running $rel_path..."
    echo "=========================================="

    if python3 "$file" 2>&1 | tee "$output_file"; then
        chmod 644 "$output_file" 2>/dev/null || true
        echo "✓ PASSED: $rel_path"
        echo "  Results saved to: $output_file"
    else
        (echo "BENCHMARK FAILED"; echo ""; cat "$output_file") > "$output_file.new" 2>/dev/null && \
            mv "$output_file.new" "$output_file" 2>/dev/null || \
            echo "BENCHMARK FAILED" > "$output_file"
        chmod 644 "$output_file" 2>/dev/null || true
        echo "✗ FAILED: $rel_path"
        echo "  Error details saved to: $output_file"
        FAILED_BENCHMARKS+=("$rel_path")
    fi
    echo ""
done < <(find . -name 'bench_*.py' -not -path './__pycache__/*' | sort)

echo "=========================================="
if [ ${#FAILED_BENCHMARKS[@]} -eq 0 ]; then
    echo "All benchmarks complete! ✓"
else
    echo "Benchmarks complete with failures! ✗"
    echo "Failed benchmarks:"
    for failed in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $failed"
    done
fi
echo "Results directory: $OUTPUT_DIR"
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*_results.txt 2>/dev/null || echo "  No result files found"
echo "=========================================="

# Exit with error if any benchmarks failed
if [ ${#FAILED_BENCHMARKS[@]} -gt 0 ]; then
    exit 1
fi
