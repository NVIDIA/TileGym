#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run all Python benchmark files
# Usage: ./run_all.sh [--parallel]

cd "$(dirname "$0")"

run_sequential() {
    echo "Running benchmarks sequentially..."
    for file in bench_*.py; do
        echo "Running $file..."
        python "$file"
        echo "---"
    done
}

# Check if --parallel flag is provided
if [[ "$1" == "--parallel" ]]; then
    echo "Running benchmarks in parallel..."
    
    # Install GNU parallel if not present
    if ! command -v parallel &> /dev/null; then
        echo "GNU parallel not found. Installing..."
        if command -v apt-get &> /dev/null; then
            apt-get update -qq && apt-get install -y -qq parallel 2>/dev/null
        elif command -v yum &> /dev/null; then
            yum install -y -q parallel 2>/dev/null
        fi
    fi
    
    # Try to run in parallel, fall back to sequential if parallel unavailable
    if command -v parallel &> /dev/null; then
        find . -maxdepth 1 -name "bench_*.py" -type f | \
            parallel --will-cite --jobs 0 --tag --line-buffer \
            "echo 'Running {}'; python {} && echo 'PASSED: {}' || echo 'FAILED: {}'"
    else
        echo "Warning: GNU parallel could not be installed. Falling back to sequential execution."
        run_sequential
    fi
else
    run_sequential
fi
