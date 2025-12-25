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
bash run_all.sh
```
> 💡 **Note**: All benchmarks are validated on **NVIDIA B200** GPUs. If you encounter Out-of-Memory (OOM) errors on other Blackwell GPUs (e.g., RTX 5080, RTX 5090), please reduce the test sizes in the benchmark scripts.

## Run a single benchmark
Execute the specific Python file, for example:
```bash
python bench_matrix_multiplication.py
```

Available benchmark scripts:
- `bench_dropout.py`
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
