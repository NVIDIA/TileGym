#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run all Python benchmark files
# Usage: ./run_all.sh [--parallel] [--junit-xml OUTPUT_FILE]

cd "$(dirname "$0")"

# Parse arguments
PARALLEL=false
JUNIT_XML=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --junit-xml)
            JUNIT_XML="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

run_sequential() {
    echo "Running benchmarks sequentially..."
    local results=()
    for file in bench_*.py; do
        echo "Running $file..."
        local start_time=$(date +%s.%N)
        if python "$file" > /tmp/bench_output.txt 2>&1; then
            local status="passed"
            echo "PASSED: $file"
        else
            local status="failed"
            echo "FAILED: $file"
        fi
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        results+=("$file:$status:$duration")
        echo "---"
    done
    
    # Generate JUnit XML if requested
    if [[ -n "$JUNIT_XML" ]]; then
        generate_junit_xml "${results[@]}"
    fi
}

generate_junit_xml() {
    local results=("$@")
    local total=0
    local failures=0
    local testcases=""
    
    for result in "${results[@]}"; do
        IFS=':' read -r file status duration <<< "$result"
        ((total++))
        if [[ "$status" == "failed" ]]; then
            ((failures++))
            testcases+="    <testcase classname=\"benchmark\" name=\"$file\" time=\"$duration\">\n"
            testcases+="      <failure message=\"Benchmark failed\"/>\n"
            testcases+="    </testcase>\n"
        else
            testcases+="    <testcase classname=\"benchmark\" name=\"$file\" time=\"$duration\"/>\n"
        fi
    done
    
    cat > "$JUNIT_XML" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="benchmarks" tests="$total" failures="$failures" time="0">
$(echo -e "$testcases")
  </testsuite>
</testsuites>
EOF
    echo "JUnit XML report written to $JUNIT_XML"
}

# Check if --parallel flag is provided
if [[ "$PARALLEL" == true ]]; then
    echo "Running benchmarks in parallel..."
    
    # Check if parallel is available (should be installed in Docker image)
    if ! command -v parallel &> /dev/null; then
        echo "Warning: GNU parallel not found. Falling back to sequential execution."
        run_sequential
    else
        # Run in parallel with result tracking
        TMPDIR=$(mktemp -d)
        
        find . -maxdepth 1 -name "bench_*.py" -type f | \
            parallel --will-cite --jobs 0 --tag --line-buffer \
            'start=$(date +%s.%N); \
             file={}; \
             if python {} > /tmp/bench_{#}_output.txt 2>&1; then \
                 status="passed"; \
                 echo "PASSED: $file"; \
             else \
                 status="failed"; \
                 echo "FAILED: $file"; \
             fi; \
             end=$(date +%s.%N); \
             duration=$(echo "$end - $start" | bc); \
             echo "$file:$status:$duration" > '"$TMPDIR"'/result_{#}.txt'
        
        # Generate JUnit XML if requested
        if [[ -n "$JUNIT_XML" ]]; then
            local results=()
            for result_file in "$TMPDIR"/result_*.txt; do
                if [[ -f "$result_file" ]]; then
                    results+=("$(cat "$result_file")")
                fi
            done
            generate_junit_xml "${results[@]}"
        fi
        
        rm -rf "$TMPDIR"
    fi
else
    run_sequential
fi
