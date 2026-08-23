---
id: tech-wave-exact-cover
kind: technique
basis: measured, N=5 kernels (+2 falsifying-scope kernels)
title: Size grids to whole residency waves
summary: When every CTA runs the same-length body, throughput is quantized by waves — at occupancy k the wave is SMs·k CTAs and rate scales with CTAs/(SMs·k·ceil(CTAs/(SMs·k))) — so a grid one CTA over a wave boundary pays a whole extra wave; choose block counts (or split K) to land on exact wave multiples. Scoped: real on row-loop and split-K bodies, absent on self-pipelining single-shot MMA grids.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Size grids to whole residency waves

## What it is

With `occupancy=k`, the device runs `SMs * k` CTAs per wave. If every CTA's body takes the same time,
a grid of `N` CTAs completes in `ceil(N / (SMs*k))` wave-times regardless of how full the last wave
is: 149 same-length CTAs on 148 SMs take exactly as long as 296. The effective rate is

    rate ≈ ideal * N / (SMs*k * ceil(N / (SMs*k)))

The technique is to treat the block count as a knob and land it on a wave multiple:

- **Trim or merge tiles** so the grid is `SMs*k` (or an exact multiple) — e.g. slightly larger tiles
  that reduce N from 160 to 148.
- **Split K (or rows) to fill one wave exactly**: a reduction split across
  `floor(SMs*k / output_tiles)` slices per output uses the whole wave with no remainder.
- **Persistent grids sidestep the cliff** by construction (`tech-persistent-grid`): `SMs*k` CTAs loop
  over tiles, so uneven work distributes at tile granularity instead of wave granularity.

Always compute from the QUERIED SM count (`torch.cuda.get_device_properties(0).multi_processor_count`),
never a hardcoded constant — grids sized for one chip silently misfit every other.

## Pattern

```python
sms = torch.cuda.get_device_properties(0).multi_processor_count
wave = sms * occupancy
# split a K-reduction so output_tiles * ksplit fills one wave (exact when output_tiles divides it)
ksplit = max(1, wave // output_tiles)
grid = (output_tiles * ksplit,)
```

## When to use

- The profile shows a short, uniform body and the grid is within ~2 waves — the last-wave remainder is
  then a first-order term (a 1.5-wave grid wastes a third of the machine).
- Row-loop, reduction, and split-K bodies where per-CTA work is genuinely uniform.
- Shape suites with many small workloads: per-shape block counts (via tile-size or split choices) can
  hold every shape on an exact cover where one global tile config leaves ragged waves.

## Caveats

- **Body-class scoping is first-class — this effect can be ABSENT.** On single-shot MMA grids whose
  bodies self-pipeline across CTAs (long in-flight memory phases overlapping compute), measured wave
  arithmetic had no predictive power: two boards' best grids sat mid-wave and exact-cover variants
  did not move them. Verify the effect exists on YOUR body with one off-by-a-wave A/B before
  optimizing for it.
- Boards whose natural CTA counts are powers of two on 148-SM parts never land near wave multiples
  anyway — the arithmetic tells you up front the knob has no reachable win; skip it.
- Uniformity is the premise: if per-CTA work varies (ragged rows, data-dependent trips), the last wave
  is not the bottleneck and a persistent/backfill form beats exact-cover accounting.
- Interacts with `tech-occupancy`: the wave size is `SMs*k`, so re-derive covers when the occupancy
  hint changes — an exact cover at occ2 is a half-filled wave at occ4.

## Evidence

- Wave exact-cover block counts (K-split-to-one-wave and merged-tile covers) were the deciding grid
  choices on three reduction/row-loop boards and materially improved two more; on the clearest case
  the split-K exact cover reached parity where the ragged grid stalled a partial wave per output
  band. Previous measurements. [2026-07, B200, cuda-tile 1.2.0, N=5 kernels]
- Falsifying scope, same campaign: on two single-shot MMA-grid boards the exact-cover variants moved
  nothing (bodies self-pipelined across the wave boundary); recorded here so the technique is not
  applied by arithmetic alone. Previous measurements. [2026-07, B200, cuda-tile 1.2.0,
  N=2 kernels]
