#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run all Python benchmark files and save results
# Usage: ./run_all.sh [OUTPUT_DIR]

cd "$(dirname "$0")"

OUTPUT_DIR="${1:-.}"
mkdir -p "$OUTPUT_DIR"

echo "Running benchmarks sequentially (parallel execution disabled to ensure accurate results)..."
echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# Run each benchmark and capture output
for file in bench_*.py; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    benchmark_name=$(basename "$file" .py)
    output_file="$OUTPUT_DIR/${benchmark_name}_results.txt"
    
    echo "=========================================="
    echo "Running $file..."
    echo "=========================================="
    
    if python "$file" | tee "$output_file"; then
        echo "✓ PASSED: $file"
        echo "  Results saved to: $output_file"
    else
        echo "✗ FAILED: $file"
        echo "FAILED" > "$output_file"
    fi
    echo ""
done

echo "=========================================="
echo "All benchmarks complete!"
echo "Results directory: $OUTPUT_DIR"
echo "=========================================="
