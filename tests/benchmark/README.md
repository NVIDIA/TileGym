<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!--- SPDX-License-Identifier: MIT -->

# Benchmarks

This directory contains standalone micro-benchmarks for key kernels.

## Prerequisites
- Install dependencies per the project [README](../../README.md).
- Additionally install plotting/data dependencies used by benchmarks:
  ```bash
  pip install matplotlib pandas
  ```

## Run all benchmarks
From this directory:
```bash
# Text format (traditional)
bash run_all.sh

# JSON format (for analysis and regression detection)
bash run_all.sh . --json
# Or directly:
python run_all_json.py .
```
> 💡 **Note**: All benchmarks are validated on **NVIDIA B200** GPUs. If you encounter Out-of-Memory (OOM) errors on other Blackwell GPUs (e.g., RTX 5080, RTX 5090), please reduce the test sizes in the benchmark scripts.

## Check for performance regressions
After running benchmarks in JSON format, you can check for performance regressions against a baseline:
```bash
# Compare current results against baseline
python ../../.github/scripts/check_benchmark_regression.py \
  --current ./current-results \
  --baseline ./baseline-results \
  --threshold 5.0 \
  --fail-on-regression
```

The regression checker will:
- Compare performance metrics between current and baseline runs
- Report any performance drops exceeding the threshold (default: 5%)
- Highlight improvements as well
- Generate detailed JSON reports for further analysis

## Run a single benchmark
Execute the specific Python file, for example:
```bash
python bench_matrix_multiplication.py
```

Available benchmark scripts:
- `bench_fused_attention.py`
- `bench_matrix_multiplication.py`
- `bench_mix_triton_cutile.py`
- `bench_mla.py`
- `bench_mla_decoding.py`
- `bench_persistent_matmul.py`
- `bench_rmsnorm.py`
- `bench_rope.py`
- `bench_silu_and_mul.py`
- `bench_softmax.py`
- `bench_swiglu.py`
