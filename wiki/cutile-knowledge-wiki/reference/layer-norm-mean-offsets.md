---
id: layer-norm-mean-offsets
kind: reference
title: layer-norm-mean-offsets (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-norms
used_by: [kernel-norms]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        mean_offsets = row * W + tub_offsets
        # Redirect OOB W lanes (tub_offsets >= W) to an out-of-bounds sentinel
        # so they don't corrupt the next row's mean slot in the flat (N*W,) buffer.
        safe_mean_offsets = ct.where(mask_W, mean_offsets, ct.full((BLOCK_SIZE_W,), STAT_N_ELEMENTS, dtype=ct.int32))
        ct.scatter(mean, safe_mean_offsets, mean_val)
```
