---
id: kernel-block-scaled-gemm
kind: kernel
title: Block-scaled GEMM (per-block quantization scales)
summary: GEMM whose operands carry per-block quantization scales — the scale-load path and where scaling is applied decide performance alongside tile geometry and scheduling.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Block-scaled GEMM (per-block quantization scales)

## What it computes

`C = (A · scale_A) @ (B · scale_B)` where A and B are narrow-precision (FP8 e4m3/e5m2) and the
scales are one factor per block of VEC_SIZE consecutive K elements.
Variants in the repo:

- **w8a8_block_fp8_matmul** — FP8 weights with per-block scales, activations with per-token-group
  scales (`src/tilegym/ops/cutile/fp8_quantization_matmul.py`).
- **ragged_block_scaled_bmm** — batch of segments packed along M with `m_indptr` offsets
  (MoE/grouped inference: per-expert token counts differ).

## Computational shape

The GEMM body is the dense tile loop (see kernel-gemm) plus a second, much smaller operand stream:
the scales. Per K-tile each CTA needs a `(TILE_M, TILE_K/VEC_SIZE)` sliver of A-scales and the
matching B-scales, and must apply them at the right granularity. The w8a8 and ragged kernels
compute an FP8 dot product for each K tile, multiply it by the combined A/B scales, and add
the scaled result to the fp32 accumulator.

Ragged variants add an outer per-segment loop with a runtime `valid_m` bound and `Array.slice`
views into the packed A (`_ragged_block_scaled_bmm_kernel`,
`src/tilegym/suites/flashinfer/cutile/gemm/ragged_block_scaled_bmm.py`).

## What dominates performance

- **The scale-load path.** Scales are tiny, so how they are fetched matters more than how much:

  - The w8a8 kernel gathers per-row A scales and a scalar B scale. Its TMA operand path
    uses `latency=4` on both scale gathers — see `reference/fp8-quant-matmul-scale-gather.md`.
    Operand loads have separate TMA and gather paths selected by `use_tma`; the scale
    gathers remain in both paths (`src/tilegym/ops/cutile/fp8_quantization_matmul.py`).

  - **Where scalar loads hurt**: the ragged kernel fetches `b_scale` as a single scalar per
    (K-tile, N-tile) inside the K loop and broadcasts it — a serialized dependent load per
    iteration whose latency is exposed when segments are small — see `reference/ragged-bmm-scale-load.md`.

    The config space varies both tile sizes and layout (`swap_ab`) to match segment sizes
    (`ragged_block_scaled_bmm.py`).
- **Ragged-batch tile geometry.** Per-segment M varies, so a fixed BLOCK_M wastes tail work on
  short segments. The autotune space varies tile sizes, `swap_ab`, GROUP_SIZE_M, occupancy,
  and launch variant. The per-expert persistent kernel schedules each segment's actual tile
  count; uniform and swapped variants mask tiles past each segment's `valid_m`.
  The `swap_ab` layout computes `(B@A^T)^T`, exchanging which operand occupies the MMA M axis.
  Re-measure the available layouts for the segment-length distribution rather than treating
  one tile or threshold as a universal choice (`ragged_block_scaled_bmm.py`).

## Applicable techniques

- `tech-tile-size` — BLOCK_M vs segment-length matching in ragged batch;
  BLOCK_N/BLOCK_K pinned to quantization block sizes in w8a8 so scale indexing stays
  one-per-tile (`fp8_quantization_matmul.py`).
- Layout choices belong in the autotune space: `swap_ab` × BLOCK_SIZE_M × occupancy in
  w8a8; swap_ab + GROUP_SIZE_M variants in ragged bmm
  (`ragged_block_scaled_bmm.py`).

- `tech-group-swizzle` — GROUP_SIZE_M tile ordering in the w8a8 and ragged kernels.
- `tech-tma-load` — tune the w8a8 TMA operand path against its gather path.
- `tech-latency-hint` — on scale gathers (`latency=4`, snippet `reference/fp8-quant-matmul-scale-gather.md`).
- `tech-persistent-grid` — distribute actual per-expert tile counts across a bounded launch grid
  in `_ragged_block_scaled_bmm_kernel`.

## Where it lives

- `src/tilegym/suites/flashinfer/cutile/gemm/ragged_block_scaled_bmm.py` — ragged batch, normal +
  swap_ab kernels.

- `src/tilegym/ops/cutile/fp8_quantization_matmul.py` — w8a8 gather and TMA kernels.
- Dispatch: `"w8a8_block_fp8_matmul"` (`src/tilegym/ops/ops.py`),
  `"flashinfer.gemm.ragged_block_scaled_bmm"` (`src/tilegym/suites/flashinfer/ops.py`).
