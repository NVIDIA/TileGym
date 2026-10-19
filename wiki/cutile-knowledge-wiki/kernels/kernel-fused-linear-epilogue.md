---
id: kernel-fused-linear-epilogue
kind: kernel
title: Fused linear + epilogue (pointwise tail fused into the GEMM)
summary: GEMM with the pointwise tail fused into the accumulator tile — the epilogue changes register pressure, saved-output traffic, and the best per-shape launch configuration.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Fused linear + epilogue (pointwise tail fused into the GEMM)

## What it computes

`y = epilogue(x @ W^T + b)` where the epilogue is a pointwise tail applied to the accumulator tile
before it is stored, such as an activation or a gate that combines two projections.
Variants in the repo:

- **linear_gluact_linear** — two input projections form a GLU gate
  (`act(x@W_a^T) * (x@W_b^T)`), followed by an output projection.

## Computational shape

The GLU forward kernel uses a dense-matmul K loop with two fp32 accumulators and a GROUP_M
pid swizzle (`src/tilegym/ops/cutile/linear_gluact_linear.py`). It loads an input tile once
and multiplies it by both weight tiles. Activation and gating happen before the final stores;
the output projection is a separate `matmul` call. What changes the shape of the kernel:

- **Extra live tiles.** A GLU epilogue needs two accumulators at once.
  Register pressure is strictly higher than the plain GEMM at the same tile size.

See `reference/gluact-accumulator-init.md`.

- **Output traffic.** The cuTile implementation has no backward. Its forward writes
  `Act_out`, `Noact_out`, and `Mul_out`, but only `Mul_out` is read by the output projection.
  `Act_out` and `Noact_out` are dead stores; include their traffic and allocations in the cost model.

## What dominates performance

- **The GEMM carries the FLOPs; the epilogue decides the gap between implementations.** Unfused,
  each epilogue stage is a separate launch plus a full (M, N) round trip through HBM; fused, that
  traffic disappears but the tile/occupancy optimum moves. Fusing is the reliable structural win
  (`tech-epilogue-fusion`), after which the kernel behaves like a GEMM with a register handicap.
- **Why per-shape config selection dominates.** Because the epilogue eats registers, the best
  (BLOCK_M, BLOCK_N, BLOCK_K, num_ctas, occupancy) is not the plain-GEMM winner and does not
  transfer across shapes or dtypes. The GLU wrapper keeps a per-problem autotune cache
  (`_cutile_autotune_linear_gluact`, `src/tilegym/ops/cutile/linear_gluact_linear.py`).
  Treat recorded winners as perishable; keep the space wide and re-tune per shape.
- **Host-overhead traps.**

  - *Materialized temporaries*: `grad_weight.addmm_(a.t(), b)` avoids the explicit
    `a.t() @ b` temporary needed by a separate `add_` (liger fused_linear_jsd).
  - *Host dispatch*: the cuTile GLU wrapper calls the fused gate forward, then a separate
    `matmul` for the output projection.

- **Epilogue size itself.** When the fused tail is large (many live fp32 tiles), emitting the
  epilogue in column slices bounds live registers: moe_actgrad_bwd writes its epilogue in
  32-column slices; the flashinfer GEMM alpha_beta epilogue was subtiled the same way.

## Applicable techniques

- `tech-epilogue-fusion` — the defining technique: pointwise tail into the accumulator tile; subtiling
  when the tail is register-heavy; chunked fusion when the intermediate
  would not fit memory (fused_linear_cross_entropy).

- `tech-exp2-activation-forms` — division-free respellings of the activation tail
  (sigmoid/SiLU/GELU-tanh/softplus) when the epilogue's math paces the kernel.
- `tech-tile-size` — same regime logic as kernel-gemm, shifted by epilogue register pressure; the GLU
  variants keep two accumulators live, increasing register demand relative to a single GEMM.
- `tech-group-swizzle` — GROUP_M pid remap in the GLU forward kernel.
- `tech-num-ctas` / `tech-occupancy` — the GLU Blackwell config space includes `num_ctas=2`
  alongside 256-wide tiles and keeps occupancy in the config.
- `tech-copy-batching` — avoid materialized temporaries in gradient accumulation.

## Where it lives

- `src/tilegym/ops/cutile/linear_gluact_linear.py` — fused input projections, GLU activation
  and gating; per-problem autotuning; separate output projection via `matmul`.
- Dispatch: `"linear_gluact_linear"` (`src/tilegym/ops/ops.py`).
