---
id: pattern-compute-bound
kind: pattern
title: Compute-bound kernel (MMA-dominated)
summary: Kernel time tracks FLOPs — SM/tensor-pipe utilization is the limiter while bandwidth has headroom; verify tensor cores are actually engaged, then tune MMA tile shape, 2-CTA MMA, L2 raster order, and the non-MMA math around the loop.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Compute-bound kernel (MMA-dominated)

## Symptom

Kernel time scales with FLOPs, not bytes. Nsight Compute classifies the kernel "Compute Bound": compute
utilization is high (>~80%) while memory throughput has headroom. Typical of GEMM, attention, and other
`ct.mma`-dominated kernels — arithmetic intensity above ~50 FLOPs/byte (performance-model.md). The score that
matters is achieved TFLOPS versus the reference implementation, not GB/s.

## Likely causes

1. **Tensor cores not engaged at all.** cuTile does **not** auto-promote fp32 to tf32 for `ct.mma`; fp32
   operands run on CUDA cores (~8x slower on Blackwell — previously measured on
   linear-attention). Confirm: check operand dtypes at the `ct.mma` call, and check SASS for
   tensor-core (HMMA/QMMA) vs FFMA instructions. Rule and cast guard: `cutile-language`
   core rules.
2. **MMA tile shape mismatched to the hardware.** Tiles too small or thin under-fill the MMA instruction;
   `tileiras --remarks` prints the selected tensor-core shapes — confirm they match what you intended.
3. **Single-CTA MMA where 2-CTA MMA is available.** Dense-dot kernels on Blackwell with large tiles and wide
   accumulators can double effective MMA width with `num_ctas=2`. Confirm: kernel is dense-dot, sm100+, tiles
   are large, and `num_ctas` is currently 1.
4. **Poor L2 reuse from block raster order.** Concurrently resident CTAs walk operand rows/columns with no
   overlap. Confirm: ncu L2 hit rate is low on a large 2D-tiled kernel with a linear block id.
5. **Non-MMA math stealing cycles.** Softmax/tanh/exp/div around the MMA loop consume issue slots. Confirm
   with the isolation experiment: replace the suspect op with `x * constant` and re-benchmark; if time drops
   sharply, the math — not the MMA — is the bottleneck.

## Candidate techniques

Ordered by expected value (after the tf32 check in cause 1, which dominates everything else when it
applies) — a reference list, not the full candidate space; explorations beyond it are encouraged:

1. `tech-tile-size` — M/N/K tile shape is the primary MMA knob: it sets instruction
   selection, accumulator registers, and K-loop length; sweep per architecture.
2. `tech-num-ctas` — `num_ctas=2` enables 2-CTA MMA on Blackwell dense-dot kernels,
   doubling effective tile width for large-tile GEMM-shaped work.
3. `tech-group-swizzle` — GROUP_SIZE_M block swizzle keeps resident CTAs on
   overlapping operand rows/columns in L2; cheap add for large 2D grids.
4. `tech-ftz-approx` — flush-to-zero and APPROX rounding move the non-MMA math
   (tanh/exp/div) onto fast hardware paths when cause 5 is confirmed.
5. `tech-exp2-activation-forms` / `tech-transcendental-replica` — when the confirmed non-MMA math
   is an activation or transcendental chain and flags are not enough, respell it: division-free
   exp2 forms; range-reduction + polynomial replicas.

## Caveats

- `num_ctas=2` is unsupported pre-sm90 and measured 1.76x slower on a thin-tile (N extent 16) MMA
  decode kernel — check the accumulator shape first (`tech-num-ctas`).
- The tf32 cast and APPROX math trade precision; re-validate numerics with a stated tolerance after either
  change.
- Tile size, num_ctas, and occupancy interact — sweep them jointly rather than fixing one and
  tuning another.
- A "compute-bound" classification taken from a tight, L2-resident timing loop can be an artifact; confirm the
  classification with an L2-flushed timer before spending iterations here.
- If tensor utilization is high but SMs still idle between waves, the limiter is scheduling, not compute — see
  `pattern-tail-effect` and `pattern-low-sm-utilization`.
