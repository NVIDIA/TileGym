---
id: kernel-gemm
kind: kernel
title: Dense GEMM (matmul / bmm / group_gemm)
summary: Tiled dense matrix multiply — the arithmetic-intensity regime set by M/N/K decides whether MMA throughput, memory bandwidth, or wave/tail effects dominate, and which techniques pay.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Dense GEMM (matmul / bmm / group_gemm)

## What it computes

`C = A @ B` for dense 2D operands: A is (M, K), B is (K, N), C is (M, N). Accumulation is fp32
regardless of input dtype; fp32 inputs are cast to tf32 so the MMA runs on tensor cores — see `reference/matmul-tf32-dtype-select.md`.

Variants in the repo:

- **matmul** — single 2D GEMM, optional transposed operands, non-persistent and static-persistent kernels.
- **bmm** — the same tile body with a leading batch dimension.
- **group_gemm** — a list of independent GEMMs with heterogeneous shapes, walked by one persistent grid.

## Computational shape

Each CTA owns one `TILE_M x TILE_N` output tile and runs a serial K loop: load a `TILE_M x TILE_K`
slice of A and a `TILE_K x TILE_N` slice of B, `ct.mma` into the accumulator, then one store of the
finished tile. Two grid shapes exist:

- **Non-persistent**: grid = `cdiv(M,TILE_M) * cdiv(N,TILE_N)` CTAs, one tile each
  (`_matmul_kernel`, `src/tilegym/ops/cutile/matmul.py`).
- **Static persistent**: grid = `min(NUM_SMS // num_ctas, total_tiles) * occupancy` CTAs, each
  striding `for tile_id in range(start, num_tiles, num_programs)` over tiles
  (`_static_persistent_matmul_kernel`, `src/tilegym/ops/cutile/matmul.py`). The loop is a
  `for range(...)` on purpose: the forOp compiles to faster code than whileOp in persistent GEMMs
  (cuTile GEMM; same conversion applied to bmm on B200).

Both map the linear block id to (tile_m, tile_n) through a GROUP_SIZE_M swizzle
(`_swizzle_2d` / `_compute_bid`, `src/tilegym/ops/cutile/matmul.py`; the computation is embedded
on `tech-group-swizzle`) so consecutive CTAs share B columns in L2.

Arithmetic intensity is the whole story of this kernel: FLOPs = 2·M·N·K against roughly
(M·K + K·N + M·N)·bytes/elem of mandatory traffic, improved by tile reuse — each loaded A/B tile is
reused TILE_N/TILE_M times. Bigger output tiles mean more reuse per byte, bounded by registers and
(on Blackwell) by what a 2-SM MMA pair can hold.

## What dominates performance

- **Large M, N, K (compute-bound regime)**: tensor-core issue rate dominates. The levers are the
  largest tile that fits (sm100 configs go up to 256x256 and 512x256 with `num_ctas=2/4`,
  `src/tilegym/ops/cutile/matmul.py`), K-loop load pipelining, and L2 reuse from the group
  swizzle. On sm90, exposing the `ct.load` cost as a tunable mattered on the critical path — the
  in-file note at the A-tile load (H100 sm90) — see `reference/matmul-a-load-critical-path.md`.
- **Small M or N (skinny GEMM, memory-bound regime)**: reuse collapses — with M ≤ TILE_M every B
  element is used O(M) times and the kernel streams the big operand at bandwidth. Large square
  tiles only add tail waste; the per-arch spaces therefore carry narrow tiles (sm120 space includes
  128x64 and 64x64, `src/tilegym/ops/cutile/matmul.py`). Operand-swap (compute `(B@A^T)^T`
  so the large dimension lands on the MMA M axis) is the standard fix in the quantized siblings;
  when M is tiny and the weight matrix is huge, restructure to a
  weight-stationary streaming form (stream the big operand once, keep the small one resident).
- **Few output tiles (small M·N)**: the grid can't fill the SMs and wave quantization dominates;
  shrinking tiles to create CTAs, raising `occupancy` on the persistent grid, or landing the block
  count on a whole residency wave (`tech-wave-exact-cover`) is worth more than
  per-tile efficiency. K only sets the serial loop length — there is no split-K in the cuTile
  matmul path (grid has no K axis), so small-M·N/large-K shapes serialize inside few CTAs.
- **Grouped/batched shapes**: group_gemm walks all problems with one persistent grid sized
  `NUM_SMS // num_ctas * occupancy` (`src/tilegym/ops/cutile/group_gemm.py`), so per-GEMM
  launch overhead and inter-GEMM load imbalance replace raw throughput as the limiter; the unsloth
  grouped variant additionally deleted host-side tile mapping and its GPU→CPU sync.

## Applicable techniques

- `tech-tile-size` — the primary knob in every regime; per-arch families differ hard: the pre-sm90
  space sweeps 64/128 tiles only, always `num_ctas=1` (A100 space explicitly
  excludes 256x256 tiles and `num_ctas=2`), while sm100+ keeps 256x256.
- `tech-num-ctas` — 2-SM MMA: sm100 configs pair 256x256x64 tiles with `num_ctas=2` (and 4)
  (`src/tilegym/ops/cutile/matmul.py`); unsupported pre-sm90, so the sm80 space pins
  `num_ctas=1`.
- `tech-group-swizzle` — GROUP_SIZE_M=8 pid remap for L2 locality, present in both kernels
  (`src/tilegym/ops/cutile/matmul.py`).
- `exhaustive_search` over the per-arch config iterator, cached by
  (M, N, K, dtype, device) (`src/tilegym/ops/cutile/matmul.py`); sm90 entries added after
  H100 runs; Triton-TileIR autotune winners were replayed as cuTile seeds.
- `tech-latency-hint` — LOAD_LATENCY as an autotune dimension (1..10, -1 = compiler-inferred), tuned on
  sm90 only.
- `tech-persistent-grid` — static-persistent kernel plus the for-not-while loop rule.
- `tech-occupancy` — carried by every config; multiplies the persistent grid size.

## Where it lives

- `src/tilegym/ops/cutile/matmul.py` — `_matmul_kernel`, `_static_persistent_matmul_kernel`,
  per-arch config iterators, tune caches.
- `src/tilegym/ops/cutile/bmm.py` — batched variant.
- `src/tilegym/ops/cutile/group_gemm.py` — persistent grouped GEMM over heterogeneous shapes.
- Dispatch: `"matmul"`, `"bmm"`, `"group_gemm"` (`src/tilegym/ops/ops.py`).
