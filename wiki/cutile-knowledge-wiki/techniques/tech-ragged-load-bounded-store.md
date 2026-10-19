---
id: tech-ragged-load-bounded-store
kind: technique
title: Whole-tensor loads with bounded ragged stores
summary: For row-independent operators, keep input descriptors fixed and enforce sequence ownership on output stores.
basis: source-semantics
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Whole-tensor loads with bounded ragged stores

## What it is

For row-independent operators, load from fixed full-tensor views and bound only
output stores with `Array.slice`. Extra input rows may be read, but their results
are discarded. This removes per-chunk input views; a device-built output
descriptor may still be needed. The optimization applies when the compiler builds
per-chunk input descriptors; the source transformation alone implies no speedup.

Existing [ragged GEMM coverage](../kernels/kernel-block-scaled-gemm.md) describes
sliced views and tile geometry; [scale-load coverage](../reference/ragged-bmm-scale-load.md)
addresses scalar loads. This page adds the input-view transformation.

See `tech-ragged-tma-fixed-descriptor` when input boundaries must also be enforced.

## Pattern

The concrete operation is `output = (input_a * input_b) @ weights`, with rows independent and `weights` shared
across chunks. After reading `start` and `end` for a chunk:

```python
# Full allocation bounds remain available to each input load.
input_a_view = input_a.tiled_view((BLOCK_M, INPUT_WIDTH), padding_mode=ct.PaddingMode.ZERO,
                      traversal_steps=(1, INPUT_WIDTH))
input_b_view = input_b.tiled_view((BLOCK_M, INPUT_WIDTH), padding_mode=ct.PaddingMode.ZERO,
                      traversal_steps=(1, INPUT_WIDTH))
input_a_tile = input_a_view.load((start, 0), allow_tma=True)
input_b_tile = input_b_view.load((start, 0), allow_tma=True)
weight_tile = ct.load(weights, index=(0, column_tile_id), shape=(INPUT_WIDTH, BLOCK_N), allow_tma=True)
output_tile = ct.mma(input_a_tile * input_b_tile, weight_tile, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
# Only the rows belonging to this chunk may be written.
output_slice = output.slice(axis=0, start=start, stop=end)
ct.store(output_slice, index=(0, column_tile_id), tile=output_tile.astype(output.dtype), allow_tma=True)
```

`traversal_steps=(1, INPUT_WIDTH)` makes `start` an element-row offset even when it is
not divisible by `BLOCK_M`. In contrast, `ct.load`'s usual block index would multiply
it by the tile height. The output slice starts at `start`, so its row tile index
is zero. The core load/store operations are shown in
`ragged-row-load-bounded-store`.

Metadata loads can use `ct.load(..., allow_tma=False)` or `ct.gather` independently
of tensor TMA. For persistence, distribute `chunk_bounds.shape[0] * NUM_N_TILES` tasks
by grid stride, with `chunk_id = task_id // NUM_N_TILES` and
`column_tile_id = task_id % NUM_N_TILES`. Each task constructs its output slice; the loop
alone proves neither descriptor reuse nor cross-iteration overlap.
See `tech-persistent-grid` for launch sizing.

## When to use

- The consumer is row-independent: elementwise work followed by a matmul whose
  reduction dimension is the feature dimension, not the ragged row dimension.
- Input rows outside a chunk but inside the allocation may legally be fetched.
- The selected compiler path spends work constructing per-chunk input descriptors.

## Caveats

- Discarding output rows cannot undo contamination of valid rows by a cross-row
  reduction, attention, scan, convolution, normalization, or row-dependent weight
  selection. Such operations need explicit input-boundary semantics.
- Every chunk owns `[start, min(start + BLOCK_M, end))`, with
  `0 <= start <= end <= min(input_a.shape[0], input_b.shape[0], output.shape[0])`.
  Require `INPUT_WIDTH == input_a.shape[1] == input_b.shape[1] == weights.shape[0]` with power-of-two
  `INPUT_WIDTH` loaded as a single tile (no K loop), and `output.shape[1] == weights.shape[1]`.
  Empty slices were not separately validated for this variant; empty chunk entries
  can be omitted on the host. Owned intervals must not overlap, and the complete
  intended output must be covered. Removing the bounded store can create overlapping
  writes across CTAs even when the rows happen to compute the same values.
- Extra input lanes may read the next sequence. That is acceptable only if those
  reads are permitted by the allocation and the application's access contract.
  Zero padding on the full view protects the allocation end, not the sequence end.
- Bases, row strides and tile layouts must remain TMA-compatible. A unit-row
  traversal supports non-tile-aligned starts; it does not waive byte alignment.
- The output slice still depends on runtime metadata. Persistence does not remove
  that dependency or imply descriptor reuse across tasks with different bounds.
- Skip an empty task list on the host; for persistence, use a positive grid cap
  no larger than the task count. Source loops do not prove descriptor overlap.

## Evidence

- `ragged-row-load-bounded-store` shows the row-wise matmul, full input views, sliced output and scalar metadata path; valid output rows depend only on their corresponding input rows and the shared weight matrix.
