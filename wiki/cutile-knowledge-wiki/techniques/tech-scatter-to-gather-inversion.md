---
id: tech-scatter-to-gather-inversion
kind: technique
basis: measured, N=4 kernels
title: Invert scatter-accumulate into binned gather-reduce
summary: Duplicate-index atomic accumulation (index_add, embedding backward, MoE combine) inverts into a two-phase form — build the inverse index map once, then each OUTPUT row gathers its contributors and reduces in registers — trading atomic contention for a cheap ranking pass plus contention-free reads.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Invert scatter-accumulate into binned gather-reduce

## What it is

The natural spelling of "accumulate row `g[i]` of the output with contribution `v[i]`" is a scatter
with `ct.atomic_add` — and its cost is set by the WORST output row's contention: thousands of
contributors to one hot row serialize on that row. The inversion flips who drives: first build the
inverse map (for each output row, the list of contributing input indices), then launch one CTA/tile
per OUTPUT row that gathers its contributors and reduces them in registers, storing once with no
atomics at all.

Building the inverse map is itself cheap and parallel, in one of three forms:

- **Rank pass**: one relaxed `ct.atomic_add(counts, g[i], 1)` sweep whose RETURN VALUE is input i's
  slot within its bin — writes `slots[g[i]][rank] = i`. Order within a bin is nondeterministic, which
  is fine: addition is the reduce.
- **Counting sort**: histogram → exclusive scan → placement (stage patterns on `kernel-scan-histogram`)
  when bin offsets must be contiguous for a later stage.
- **Capacity-bounded bins**: when the index distribution has a provable per-row bound, a fixed
  `(rows, cap)` slot table skips the scan entirely; certify the bound on real index data.

## Pattern

```python
# phase 1: inverse map (rank pass) — one light kernel over inputs
r = ct.atomic_add(counts, (g,), one)          # r = my slot in bin g (relaxed order)
ct.scatter(slots, (g, r), idx)

# phase 2: one tile per output row — gather contributors, reduce in registers, ONE store
n   = ct.load(counts, index=(row,), shape=(1,))            # true contributor count for this row
acc = ct.full((1, D), 0.0, dtype=ct.float32)
for j in range(cap_tiles):                                 # capacity-bounded loop, tail masked
    ids  = ct.load(slots, index=(row, j), shape=(1, TJ))
    live = (j * TJ + ct.arange(TJ, dtype=ct.int32)) < n    # zero out slots past the true count
    vals = ct.where(live, gathered_rows(V, ids), 0.0)      # (TJ, D); dead slots contribute 0
    acc  = acc + ct.sum(vals, axis=0, keepdims=True)
ct.store(Out, index=(row, 0), tile=acc)                    # no atomics anywhere in phase 2
```

## When to use

- Duplicate-index accumulation where the duplication factor is high or skewed: embedding backward,
  MoE expert-output combine, `index_add`-class ops, histogram-weighted reductions.
- The output rows are reused across calls (the inverse map amortizes) OR the map build is measurably
  cheaper than the contention it removes — on skewed inputs it usually is, because the rank pass's
  atomics are one per INPUT (uniform) while the scatter's are one per input SERIALIZED per hot row.
- Determinism matters: the gather-reduce's in-register accumulation has a fixed order per row, unlike
  atomic scatter — this form can be made bit-reproducible.

## Caveats

- Capacity bounds must come from the REAL index distribution, not the mean: one over-capacity row
  silently drops contributions if the loop is not masked against the true count. Validate the bound
  and assert it host-side.
- On UNIFORM low-duplication indices the plain atomic scatter can win — the inversion pays two passes
  and the map traffic; race both once per op family.
- The rank pass's nondeterministic within-bin order is only safe because the reduce is commutative;
  if a later stage needs stable order, rank explicitly in phase 2 instead (a precede-mask comparison
  is enough) rather than making phase 1 ordered.
- Inter-CTA handoff between phases belongs at a kernel boundary, not inside one kernel — the launch
  is the only inter-CTA synchronization point (see `cutile-language`).

## Evidence

- Expert-output weighted accumulation (MoE-combine class, heavy index skew): binned gather-reduce
  with a relaxed-atomic rank pass replaced the atomic-scatter form and was the shipped architecture;
  the win class on that board was 1.9x+ over the contended scatter. Previous measurement.
  [2026-07, B200, cuda-tile 1.2.0, N=1 kernel]
