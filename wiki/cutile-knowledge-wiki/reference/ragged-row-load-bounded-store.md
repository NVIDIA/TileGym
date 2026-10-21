---
id: ragged-row-load-bounded-store
kind: reference
title: Whole-tensor loads with bounded ragged stores (code snapshot)
summary: Core cuTile operations for full-input loads and bounded ragged output stores.
used_by: [tech-ragged-load-bounded-store]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->


# Whole-tensor loads with bounded ragged stores

Kernel-body excerpt for `output = (input_a * input_b) @ weights`; `weights` is
shared across chunks. `ct` denotes `cuda.tile`; tile sizes are compile-time constants.

```python
start = ct.load(chunk_bounds, index=(ct.bid(0), 0), shape=(), allow_tma=False)
end = ct.load(chunk_bounds, index=(ct.bid(0), 1), shape=(), allow_tma=False)
column_tile_id = ct.bid(1)
input_a_view = input_a.tiled_view(
    (BLOCK_M, INPUT_WIDTH), padding_mode=ct.PaddingMode.ZERO,
    traversal_steps=(1, INPUT_WIDTH))
input_b_view = input_b.tiled_view(
    (BLOCK_M, INPUT_WIDTH), padding_mode=ct.PaddingMode.ZERO,
    traversal_steps=(1, INPUT_WIDTH))
input_a_tile = input_a_view.load((start, 0), allow_tma=True)
input_b_tile = input_b_view.load((start, 0), allow_tma=True)
weight_tile = ct.load(weights, index=(0, column_tile_id),
                      shape=(INPUT_WIDTH, BLOCK_N), allow_tma=True)
output_tile = ct.mma(input_a_tile * input_b_tile, weight_tile,
                     ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
output_slice = output.slice(axis=0, start=start, stop=end)
ct.store(output_slice, index=(0, column_tile_id),
         tile=output_tile.astype(output.dtype), allow_tma=True)
```

Reads may cross segment boundaries within each input allocation; writes own only
`[start, min(start + BLOCK_M, end))`. Full-view zero padding protects the allocation
end, not the segment end. See `tech-ragged-load-bounded-store` for shape/alignment
requirements, non-overlapping ownership and the unvalidated empty-slice case.
