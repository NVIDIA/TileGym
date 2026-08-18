---
id: pattern-tail-effect
kind: pattern
title: Tail effect (uneven last wave)
summary: Block count lands just past a multiple of the SM count, so the last wave runs nearly empty — performance is discontinuous in problem size; remove the wave boundary or move the block count.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Tail effect (uneven last wave)

## Symptom

Performance is discontinuous in problem size: a slightly larger input is disproportionately slower, and a
sweep over sizes shows a sawtooth in achieved throughput. The kernel launches `ceil(work / tile)` blocks; when
that count is just past a multiple of the SM-resident capacity, the final wave runs with most SMs idle. Nsight
Compute shows utilization high through the body of the kernel and collapsing at the end.

## Likely causes

1. **Wave quantization.** With B blocks and W CTAs resident machine-wide per wave, the kernel takes
   `ceil(B / W)` waves and the last wave holds `B mod W` blocks. Confirm arithmetically: compute B from the
   grid formula, get the SM count from `torch.cuda.get_device_properties(device).multi_processor_count`, and
   check whether the shapes that regress are exactly the ones whose last wave is nearly empty.
2. **Tile size sets the block count on the wrong side of a boundary.** The tile size, not the input, decides
   B. Confirm: recompute B under a neighboring tile size and see whether the boundary moves off the affected
   shapes.
3. **Grid computed from a stale SM count.** A hardcoded SM constant from a different board silently shifts
   every wave boundary (e.g. 132 assumed vs 148 actual on B200 variants). Confirm: grep the launch path for
   integer SM constants and compare against the queried `multi_processor_count`.
4. **Uneven per-block work, not uneven block count.** Ragged segments make some blocks straggle so every wave
   has a tail. Confirm: per-block work is data-dependent; the sawtooth is absent but utilization still decays
   toward the end of the kernel.

## Candidate techniques

Ordered by expected value — a reference list, not the full candidate space; explorations beyond it
are encouraged:

1. `tech-persistent-grid` — a grid of `min(NUM_SM * occupancy, n_items)` blocks that
   grid-stride over tiles has no wave boundary at all; stragglers and remainders amortize across the loop.
2. `tech-wave-exact-cover` — choose block counts (or split K) to land on exact wave multiples;
   scoped — real on row-loop and split-K bodies, absent on self-pipelining single-shot MMA grids.
3. `tech-tile-size` — change the block count directly; a different tile size can fill the
   last wave or eliminate it for the shapes that matter.
4. `tech-occupancy` — the occupancy hint changes CTAs resident per SM and therefore the
   wave width W; it both shifts the boundary and shrinks the cost of a partial wave.

## Caveats

- **Never hardcode the SM count** when computing grids or wave arithmetic — query it at runtime;
  grids sized to one board's constant silently misfit every other.
- Tail effect only matters at moderate wave counts; when `B >> W` the partial wave is amortized and this
  pattern is the wrong diagnosis — measure waves before acting.
- Fixing the boundary for one shape moves it onto another: tile-size and occupancy fixes here are
  shape-heterogeneous by construction, so measure the full matrix and gate
  (`pattern-shape-heterogeneity`).
- Wave arithmetic assumes the occupancy the hardware actually grants, not the hint you asked for; confirm
  achieved occupancy in ncu when the arithmetic and the measurement disagree.
