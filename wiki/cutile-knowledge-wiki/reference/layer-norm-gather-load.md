---
id: layer-norm-gather-load
kind: reference
title: layer-norm-gather-load (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-norms
used_by: [kernel-norms]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
            x_tile = ct.gather(x, offsets, padding_value=0)
            x_tile = ct.astype(x_tile, ct.float32)
            # Mask out OOB C/W lanes: ct.gather returns the padding value only
            # when the computed offset is outside the buffer, but for OOB
            # col_offsets (>= C) the offset stride_c * col_offsets still lands
            # inside the buffer — in the next row's data — and gather returns
            # that real (wrong) value. Explicit mask mirrors what the
            # BLOCK_SIZE_W == 1 branch and the _var path already do.
            x_tile = ct.where(mask, x_tile, ct.zeros((BLOCK_SIZE_C, BLOCK_SIZE_W), dtype=ct.float32))
```
