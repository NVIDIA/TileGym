---
id: tech-persistent-grid
kind: technique
basis: ungraded-batch-1
title: Persistent grid (grid-stride over work items)
summary: Launch min(NUM_SM * occupancy, n_items) blocks that grid-stride over work items instead of one block per item; wins when waves or launch overhead dominate, and enables fused partial reductions and host-work elimination.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Persistent grid (grid-stride over work items)

## What it is

Replace a one-block-per-work-item launch (`grid = (n_items,)`) with a capped grid whose blocks loop over
multiple work items via a grid-stride loop. Beyond scheduling,
persistence changes what the kernel *can* do: a block that visits many items can accumulate partial reductions
across them, and a fixed-size grid removes per-item host work (index mapping, per-launch allocations,
GPU-to-CPU syncs) and is CUDA-Graph-friendly.

## Pattern

```python
# Before: one block per row
@ct.kernel
def kernel(X, Y, N: ct.constexpr):
    row = ct.bid(0)
    ...

grid = (n_rows, 1, 1)

# After: persistent — use a for-range grid-stride loop (not while)
@ct.kernel
def kernel(X, Y, n_rows: ct.constexpr, N: ct.constexpr):
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    for row in range(pid, n_rows, num_programs):
        ...

NUM_SM = torch.cuda.get_device_properties(device).multi_processor_count
grid = (min(NUM_SM * occupancy, n_rows), 1, 1)   # occupancy from autotune cfg
```

## When to use

- `n_work_items > NUM_SM * 2` on memory-bound ops with many small work items.
- Launch or host overhead is visible in a profile: persistence can delete host-side per-item tile mapping and
  device syncs, not just amortize launches.
- The op wants a cross-item reduction (e.g. a weight-gradient accumulated over rows): each persistent block
  accumulates a partial into a `(grid, TILE)` buffer, finalized by a small second kernel.
- Grouped/ragged GEMM-like ops where per-group grids would otherwise force host-side grid computation per
  group.

## Caveats

- **Write the loop as `for ... in range(pid, total, num_programs)`, never `while`** — the cuTile forOp
  compiles to faster code than whileOp.
- **Query the live SM count** (`torch.cuda.get_device_properties(...).multi_processor_count`, as in the
  Pattern above and in `src/tilegym/ops/cutile/rms_norm.py`); hardcoded SM constants go stale across
  parts — one in-repo attention heuristic carried `num_sms = 132` for B200
  although B200 parts have 148 SMs, silently shifting every wave estimate.
- **Not a monotonic win**: the grid-stride loop adds control overhead, holds registers and SM residency for
  the whole kernel, and the persistent variant can spill where the simple launch did not. Compare against the
  simple launch on the op's full shape matrix before keeping it.
- **On heavy mma bodies the loop tax is small and recoverable — not the folklore half-rate.** A standalone
  A/B on a bf16 4096-cubed GEMM measured: one-tile-per-CTA 1133 TF/s; persistent single-accumulator
  988 TF/s (13% loop tax); persistent with TWO live accumulators sharing each A load 1114 TF/s (parity —
  the second accumulator keeps the mma pipe fed across tile boundaries). Dynamic-trip persistent GEMM
  loops have separately measured 25% FASTER than a static full unroll on another mma-heavy body. Race
  persistent forms with a dual-accumulator variant included; do not price them out on single-shot folklore.
- The grid-cap multiplier and the kernel `occupancy` hint interact (the same number sizes the launch and the
  register budget) — tune them together; see `tech-occupancy`.
- Accumulator shape matters inside the loop: keep it as low-dimensional as the math allows (see the bmm
  evidence — a 2D accumulator replaced a 3D one during the persistent conversion).

## Evidence

- Persistent GEMM loop-tax A/B (bf16 4096^3, TM/TN/TK 128/256/64, 148-CTA persistent grid, all forms bit-exact vs torch): flat 1133 TF/s, persistent single-acc 988 TF/s (0.87x), persistent dual-acc 1114 TF/s (0.98x). Previous measurement. [2026-08, B200, cuda-tile 1.2.0, N=1 repro]
- Dynamic-trip persistent GEMM 25% faster than the static full unroll; nested dynamic-in-dynamic loops compiled and pipelined (two mma-heavy kernels). Previous measurements. [2026-07, B200, cuda-tile 1.2.0, N=2 kernels]

- unsloth grouped_gemm fwd/dX: true persistent pattern (grid=NUM_SMS, tile-stride loop) eliminating host-side tile mapping, its caching, and the GPU-to-CPU sync from `m_sizes.tolist()`; CUDAGraph benchmark added in the same change.
- rms_norm backward replaced by a persistent grid-stride kernel that accumulates per-block dW partials into a (grid, TILE_N) buffer.
- cuTile GEMM persistent loop: `while` converted to `for current_pid in range(pid, total_tiles, NUM_PROGRAMS)` — the change exists specifically because forOp compiles to faster code than whileOp.
- B200, bmm: same while-to-for persistent conversion, plus a 2D accumulator replacing a 3D one and transposed-A handled by a 3D load + `ct.permute`. [2026-07]
- recurrent_gated_delta_rule: persistent variant added alongside its autotune space.
