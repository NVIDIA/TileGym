#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Wrapper to run benchmark and save output as JSON"""

import json
import re
import subprocess
import sys


def parse_triton_table(output):
    """Parse Triton benchmark table output into structured JSON"""
    results = []
    
    # Split into sections by benchmark name (lines ending with unit like -TFLOPS: or -GBps:)
    sections = re.split(r'\n(?=[a-zA-Z][\w-]+-(?:TFLOPS|GBps|GB/s):)', output)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # First line is benchmark name and unit
        header_match = re.match(r'([\w-]+)-(TFLOPS|GBps|GB/s):', lines[0])
        if not header_match:
            continue
            
        benchmark_name = header_match.group(1)
        unit = header_match.group(2)
        
        # Second line is column headers
        if len(lines) < 2:
            continue
        headers = lines[1].split()
        
        # Remaining lines are data
        data_rows = []
        for line in lines[2:]:
            if not line.strip() or line.strip().startswith('-'):
                continue
            values = line.split()
            if len(values) >= len(headers):
                row = {headers[i]: float(values[i]) if i > 0 else values[i] 
                       for i in range(len(headers))}
                data_rows.append(row)
        
        if data_rows:
            results.append({
                "benchmark": benchmark_name,
                "unit": unit,
                "columns": headers,
                "data": data_rows
            })
    
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: run_with_json.py <benchmark_file.py> <output.json>")
        sys.exit(1)
    
    benchmark_file = sys.argv[1]
    output_json = sys.argv[2]
    
    # Run the benchmark
    try:
        result = subprocess.run(
            [sys.executable, benchmark_file],
            capture_output=True,
            text=True,
            check=True
        )
        stdout = result.stdout
        
        # Print stdout for debugging (visible in CI logs)
        print(stdout)
        
        # Parse into JSON
        parsed = parse_triton_table(stdout)
        
        # Save JSON
        with open(output_json, 'w') as f:
            json.dump(parsed, f, indent=2)
        
        sys.exit(0)
        
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        
        # Save error as JSON
        with open(output_json, 'w') as f:
            json.dump({"error": str(e), "stderr": e.stderr}, f, indent=2)
        
        sys.exit(1)


if __name__ == "__main__":
    main()

