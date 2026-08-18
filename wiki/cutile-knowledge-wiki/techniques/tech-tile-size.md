---
id: tech-tile-size
kind: technique
basis: ungraded-batch-1
title: Tile size tuning
summary: Tile size sets each block's work assignment and drives the elements-per-thread → registers → occupancy chain; sweep several sizes per architecture and chunk instead of widening past the register budget.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Tile size tuning

## What it is

Tile size parameters (`TILE_M`, `TILE_N`, `TILE_K`, `BLOCK_SIZE`, ...) determine how much of the tensor each
block processes — the primary knob for data granularity, register/shared-memory utilization, and memory
transaction efficiency.

The governing mechanism is a chain: tile size → elements per thread → registers per thread → achievable
occupancy. Larger tiles raise per-block work and memory-transaction width but consume registers; past the
register budget the kernel spills and performance falls off a cliff. Smaller tiles allow more concurrent
blocks but add launch/loop overhead. The right response to hitting the budget is usually **chunking** (loop
over the row in fixed-size chunks) rather than one ever-wider tile.

## Pattern

```python
@ct.kernel
def my_kernel(X, Y, TILE_M: ct.constexpr, TILE_N: ct.constexpr):
    ...

# Sweep tile sizes via autotune rather than hardcoding one value
configs = [SimpleNamespace(TILE_M=m, TILE_N=n, occupancy=occ)
           for m in (32, 64, 128) for n in (32, 64, 128) for occ in (1, 2, 4)]
result = ct.tune.exhaustive_search(configs, stream=stream, grid_fn=..., kernel=my_kernel,
                                   args_fn=lambda cfg: (X, Y, cfg.TILE_M, cfg.TILE_N),
                                   hints_fn=lambda cfg: {"occupancy": cfg.occupancy})
```

## When to use

- Always: optimal tile sizes are hardware- and kernel-specific; benchmark several plausible sizes rather than
  trusting any recorded value (on Blackwell 2D problems, cover roughly 16x16 up to 128x128).
- A wide-row kernel (norms, softmax, cross-entropy) whose single-tile width would exceed the register budget:
  cap the tile and process the row in chunks.
- Per-architecture dispatch: cap or select tile sizes by `torch.cuda.get_device_capability()` — register
  files, shared memory, and supported tile shapes differ across sm80/sm90/sm100/sm120.
- Latency-bound loop-carried computation (scans, recurrences): a *narrower* tile in the carried dimension can
  win even when bandwidth arguments favor wide tiles.

## Caveats

- Register-spill cliffs make the search non-convex: a tile 2x larger can be far more than 2x slower. Sweep,
  do not extrapolate. On f32 ELEMENTWISE bodies the cliff has a measured danger band: 4-8K live f32
  elements per tile (two kernels hit the wall at 4K and 8K on different bodies) — race across that
  boundary explicitly rather than interpolating through it.
- Per-arch validity limits prune the space: e.g. sm80 supports neither 256x256 tiles nor `num_ctas=2`
  — see `tech-num-ctas`.
- Tile size interacts with the occupancy hint (bigger tiles need lower occupancy to fit) — tune them together;
  see `tech-occupancy`.
- Tile choice can change which masking code compiles out: when the tile exactly covers the dimension
  (`TILE_N == n_cols`, `BLOCK_SIZE == VOCAB_SIZE`), bounds masks can be skipped at compile time.
- Do not copy tile sizes across backends or architectures; the same op's Triton config is evidence of nothing
  about the cuTile register budget.

## Evidence

- f32 live-tile-area cliff: two elementwise kernels measured hard spill walls at 8K and 4K f32 elements per tile respectively; below the band, tile size was a smooth knob. Previous measurements. [2026-07, B200, cuda-tile 1.2.0, N=2 kernels]

- B200, liger fused_add_rms_norm bwd: 2-chunk persistent kernel with `CHUNK_SIZE=BLOCK_SIZE//2`; previously measured: caps 32 f32/thread at CHUNK_SIZE=4096, reducing peak register pressure from 87% to 56% of budget, -50% bwd time. [2026-07]
- B200, cuTile softmax: single- vs multi-chunk dispatch, previously measured — N=4096: 32 elem/thread, 64 regs, occ=8; N=16384: 128 elem/thread, 255 regs, occ=2; empirical cutoff `_SINGLE_CHUNK_MAX_N=16384`. [2026-07]
- Other in-repo instances (B200/sm80/sm_100): fused_add_rms_norm fwd 2-chunk cap; cross_entropy per-capability BLOCK cap; the A100 pass pinning 128-tile pre-sm90 defaults; grpo_loss BLOCK_N raise; unsloth swiglu/geglu per-sweep FWD/BWD split. [2026-07]
