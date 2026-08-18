---
id: pattern-low-sm-utilization
kind: pattern
title: Low SM utilization (thin grid / too few CTAs)
summary: The launch grid puts fewer CTAs on the machine than it has SMs to feed — most of the GPU sits idle while a handful of blocks serially absorb the work; fix the grid, not the inner loop.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Low SM utilization (thin grid / too few CTAs)

## Symptom

Most SMs are idle while the kernel runs. Nsight Compute shows low SM busy / achieved occupancy despite no
resource limit; the launch grid is smaller than (or barely above) the SM count. A telltale scaling signature:
kernel time grows linearly with input along some axis while the grid stays fixed — a few blocks are absorbing
the growth in longer serial loops instead of the machine getting wider.

## Likely causes

1. **Grid formula produces too few blocks.** `grid = (n_items,)` with few items, or a fixed small grid.
   Confirm: print the launch grid and compare against
   `torch.cuda.get_device_properties(device).multi_processor_count` — a grid width below NUM_SM cannot fill
   the machine.
2. **Grid tied to a small metadata domain while the real work domain grows.** The grid is sized by expert
   count, head count, or segment count, while tokens/rows grow with input and are consumed by per-program
   loops. Confirm: read the grid formula; if time scales with tokens but the grid scales with heads, the
   decomposition is the cause.
3. **Naive 1:1 block-to-item launch at the other extreme misdiagnosed.** If `n_items >> NUM_SM` the problem is
   not thinness — check `pattern-tail-effect` instead. This page is for grids that are
   *small*; measure `grid_width / NUM_SM` before choosing a fix.
4. **Persistent grid sized from a wrong SM count.** A grid computed from a hardcoded SM constant for a
   different board undershoots the real machine. Confirm: grep the launch path for integer SM
   constants and compare against the queried `multi_processor_count`.

## Candidate techniques

Ordered by expected value — a reference list, not the full candidate space; explorations beyond it
are encouraged:

1. `tech-persistent-grid` — size the grid to `min(NUM_SM * occupancy, n_items)` and
   grid-stride over work; the highest-impact scheduling fix when there are many independent items.
2. Work-domain decomposition: when the grid is bound to a small metadata domain, give the growing
   domain (tokens, rows, K-chunks) its own grid dimension, or re-express the op on a decomposition
   that parallelizes it (a 1x1 conv re-expressed as a plain matmul is the canonical case).
3. `tech-occupancy` — the occupancy hint multiplies resident CTAs per SM and is the
   multiplier in the persistent-grid width formula; autotune it rather than hardcoding.

## Caveats

- Persistent scheduling does not help when `n_items < NUM_SM` — there is simply not enough independent work;
  the fix is a finer decomposition (technique 2) or accepting the kernel is latency-bound.
- Query the SM count at runtime (`torch.cuda.get_device_properties(...).multi_processor_count`);
  constants baked for one board silently misfit every other.
- Splitting a work domain adds combining cost (partial reductions, extra buffers); measure both ends of the
  shape range, not just the shape that motivated the split.
- Thin grids at tiny shapes are often really launch-overhead-bound — check the device-time vs wall-time gap
  before restructuring the kernel (`pattern-host-overhead-bound`).
