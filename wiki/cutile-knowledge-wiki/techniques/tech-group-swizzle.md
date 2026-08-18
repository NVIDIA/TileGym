---
id: tech-group-swizzle
kind: technique
basis: ungraded-batch-1
title: GROUP_SIZE_M block swizzle for L2 reuse
summary: Remap the linear block id of a 2D-tiled kernel so groups of GROUP_SIZE_M M-tiles walk the N dimension together, keeping concurrently resident CTAs on overlapping operand rows/columns in L2.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# GROUP_SIZE_M block swizzle for L2 reuse

## What it is

A launch-order remap for 2D-tiled dot-family kernels (matmul, bmm, attention-like GEMMs). With a naive
row-major tile order, the CTAs resident at any instant span one long row of output tiles, so they share B
columns but almost no A rows — the L2 working set is maximal. The swizzle instead processes tiles in groups of
`GROUP_SIZE_M` consecutive M-tiles that advance through N together: resident CTAs then share both a small band
of A rows and a small band of B columns, and both operands stay hot in L2. The output is unchanged; only the
block-id → (bid_m, bid_n) mapping changes.

## Pattern

Host- or kernel-side delinearization:

```python
def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)  # clamp the last group
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n
```

The shipped bmm kernel computes the same thing kernel-side, with `ct.minimum` for the boundary clamp
(the extra `bid_q` handles the batch axis) — see `reference/bmm-group-swizzle.md`.
In a persistent kernel, apply the swizzle to each tile index produced by the
tile-stride loop, not once to `ct.bid(0)`.

Treat `GROUP_SIZE_M` as a tunable: sweep {4, 8, 16}. The repo default is 8 across the persistent-matmul and
bmm config generators (`src/tilegym/ops/cutile/matmul.py`, `src/tilegym/ops/cutile/bmm.py`;
the non-persistent matmul configs carry no `GROUP_SIZE_M` field).

## When to use

- 2D tile grid over a dot-style kernel where each output tile re-reads full rows of A and columns of B, and
  the grid has many more tiles than can be resident at once — otherwise there is nothing to reorder.
- Operand footprints large relative to L2, so cross-CTA reuse is the difference between L2 hits and DRAM
  traffic.
- Adding TMA or growing tile counts on large matrices: swizzle is the standard companion to a TMA matmul.

## Caveats

- The optimal `GROUP_SIZE_M` depends on matrix shape and L2 capacity; there is no universal value — sweep it,
  or put it in the autotune space.
- No effect (only index-math overhead) when the whole grid fits residency, e.g. small matrices or heavily
  persistent launches with few programs.
- The last group along M is ragged; the mapping must clamp with
  `min(num_bid_m - first_bid_m, GROUP_SIZE_M)` or tiles are skipped/duplicated (the `ct.minimum`
  line in `reference/bmm-group-swizzle.md`).
- Swizzle interacts with CTA count: changes that multiply the number of concurrent CTAs (smaller tiles, higher
  occupancy) change what "resident together" means, so re-measure the swizzle after re-tiling.
- Reuse effects are shape-heterogeneous across a workload matrix; retain a swizzle change from the
  full shape matrix, not from one flattering shape.

## Evidence
- unsloth w8a8 block-fp8 matmul (B200): TMA kernel added with a `GROUP_SIZE_M` pid swizzle as part of the suite fill-in. [2026-07]
- cuTile persistent matmul and bmm ship GROUP_SIZE_M=8 in every autotune config across all arch branches (sm80/sm90/sm100/sm120); the non-persistent matmul configs carry no GROUP_SIZE_M field — see `reference/matmul-autotune-configs.md` and `reference/bmm-group-swizzle.md`. [2026-07]
