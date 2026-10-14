<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Supplied optimization solution

This directory contains the best correct kernel from our Codex B200 run and the recorded performance table below. See [the experiment README](../README.md) for environment setup. Delete this entire directory before starting your own agent experiment.

Our optimization run used a 90-minute budget and finished in about 46 minutes.

## Reproduce the supplied solution

`solution/` is a reproducible reference result from our run. Keep it when measuring that implementation; remove it only when starting an independent agent experiment. From the parent package directory (`cd ..` if you are in `solution/`), measure the original kernel, then replace it with the supplied solution:

```bash
mkdir -p results
cd example
python -m pytest test_decode.py
python bench_decode.py --json ../results/baseline.json
cp ../solution/decode_cutile.py decode_cutile.py
python -m pytest test_decode.py
python ../scripts/validate_extended.py
python bench_decode.py --json ../results/final.json
```

Check each validation result before proceeding. This workflow measures the provided solution on your machine. Both original and supplied implementations passed all five pytest tests in our recorded run; the supplied solution additionally passed all 34 extended checks. The table below contains those recorded measurements. Running the commands above produces new data in `results/`.

## Recorded performance

Each benchmark invocation reports median CUDA-event latency over 30 timed iterations after 10 warmups, with 256 MB L2 flushing. Baseline and final were measured in separate invocations; no three-pass interleaved rerun is claimed here. SDPA dense K/V gathering happens before timing. Optimized timing includes the merge where multiple splits are used; single-split paths directly output with one launch.

All latencies below are **ms**. Speed relative to SDPA is `SDPA latency / optimized latency` (higher is better).

| Workload | Original | Optimized | SDPA (final run) | Speedup over original | Speed relative to SDPA |
|---|---:|---:|---:|---:|---:|
| B1_S4096 | 0.266240 | 0.015392 | 0.015360 | 17.30x | 1.00x |
| B1_S32768 | 2.074624 | 0.039936 | 0.037920 | 51.95x | 0.95x |
| B8_S4096 | 0.526336 | 0.037888 | 0.034848 | 13.89x | 0.92x |
| B8_S32768 | 4.148128 | 0.185344 | 0.172048 | 22.38x | 0.93x |
| B64_S2048 | 1.839072 | 0.102432 | 0.105472 | 17.95x | 1.03x |
| B64_S8192 | 7.257088 | 0.329728 | 0.322560 | 22.01x | 0.98x |

Geometric-mean speedup over the original is 21.9x; per-workload speedups are 13.9x–51.9x. Absolute timings and best dispatch choices can vary by GPU and environment.

The solution groups GQA heads, loads multiple KV pages per iteration, uses Tensor Core MMA with FP32 accumulation, partitions KV for small batches, and chooses tiles/splits by shape. A single split bypasses temporary partial outputs and merge. The reference implementation and numerical tolerances are preserved.
