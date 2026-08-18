---
id: pattern-register-pressure
kind: pattern
title: Register pressure (spills and occupancy collapse on big tiles)
summary: Per-thread register demand from wide tiles or deep fusion hits the ceiling — the compiler spills to local memory or demotes occupancy, and performance falls off a cliff past a tile width; chunk the work instead of widening.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Register pressure (spills and occupancy collapse on big tiles)

## Symptom

Performance degrades sharply — not gradually — past a particular tile width, hidden dimension, or fusion
depth. Nsight Compute shows registers/thread at or near the ceiling (255), achieved occupancy far below the
hint, and local-memory traffic where there should be none. In SASS, `LDL`/`STL` instructions are spills. The
same kernel is fine at smaller widths.

## Likely causes

1. **Elements-per-thread grew past the register budget.** Tile elements divide across a fixed thread count, so
   widening the tile raises registers/thread roughly linearly until spill or occupancy demotion. Confirm: do
   the arithmetic (elements per thread x live values per element), then check ncu registers/thread and
   achieved occupancy at the failing width. A recorded chain of this shape: softmax at N=4096 runs 32
   elem/thread at 64 regs and occupancy 8; at N=16384 it is 128 elem/thread at 255 regs and occupancy 2
   (B200, 128 threads/block; previous measurements).
2. **Too much live state from fusion.** Fused epilogues or multi-quantity kernels hold several fp32 tiles live
   simultaneously. Confirm: count live tiles across the loop body in the source; spills appear only in the
   fused variant.
3. **Occupancy/worker-warp hints forcing the budget down.** Higher occupancy shrinks the per-CTA register
   allocation; a hint that is fine at small N spills at large N. Confirm: drop the occupancy hint and see the
   spill disappear (spill-safe spaces are shape-dependent).

**IR-dump diagnosis**: dump bytecode and run
`tileiras --remarks=schedule` — a very high initiation interval (II > ~1000) indicates register pressure or a
long dependency chain; simplify the inner loop or shrink the tile and re-check whether II drops.

## Candidate techniques

Ordered by expected value — a reference list, not the full candidate space; explorations beyond it
are encouraged:

1. `tech-tile-size` — cap the tile and process the row in chunks instead of widening;
   the elements-per-thread → registers → occupancy chain is the core of this pattern, and chunking caps it
   (fused_add_rms_norm bwd moved to a 2-chunk kernel with CHUNK_SIZE = BLOCK_SIZE//2, cutting peak register
   pressure from 87% to 56% of the B200 budget per previous measurements).
2. `tech-occupancy` — trade the direction the kernel needs: lower the occupancy hint to
   give each CTA a bigger register budget, or raise worker warps to spread elements across more threads; the
   optimum is tile-config-dependent, so sweep, don't reason.
3. Split the work: divide one monolithic kernel into passes with smaller live sets (splitting a
   register-bound backward into one sweep per output group has let `latency` pipelining
   engage at all), or dispatch single-chunk vs multi-chunk variants at a measured width.

## Caveats

- Tile size and occupancy interact through the same register budget — sweep them jointly; tuning
  one at a fixed value of the other finds a false optimum.
- Register pressure is shape-dependent: a fix that only large widths need should be gated on width, not made
  the global default.
- Spills do not always appear as ncu spill counters; high II in `tileiras --remarks` or a sudden occupancy
  demotion may be the only visible signal — use the IR-dump route above.
- Splitting a kernel adds passes and intermediate traffic; verify the split wins on the small shapes too, not
  just the spilling ones.
