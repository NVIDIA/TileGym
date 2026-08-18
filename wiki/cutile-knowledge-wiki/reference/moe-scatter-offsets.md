---
id: moe-scatter-offsets
kind: reference
title: moe-scatter-offsets (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-gather-scatter
used_by: [kernel-gather-scatter]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        # Write back the tile of the output.
        # Use 2D indexing so cuTile bounds-checks the N dimension. Otherwise,
        # with TILE_SIZE_N > N (e.g. down-projection GEMM where N=hidden_size),
        # out-of-range column offsets silently alias into the next row of a
        # flattened C buffer and corrupt neighbouring outputs.
        offs_cn = bid_n * TILE_SIZE_N + ct.arange(TILE_SIZE_N, dtype=ct.int32)
        accumulator = ct.astype(accumulator, c_ptr.dtype)
        ct.scatter(
```
