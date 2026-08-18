---
id: rms-norm-load
kind: reference
title: rms-norm-load (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-norms
used_by: [kernel-norms]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        # Load input tile.
        # padding_mode=ZERO is required when N is not a power of two so that
        # the out-of-range columns [N, TILE_SIZE_N) contribute 0 to the
        # sum-of-squares reduction below; otherwise uninitialized memory
        # inflates the variance and compresses the normalized output.
        x = ct.load(
            X,
            index=(current_bid, 0),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=10,  # +2% perf from this hint
        )
        x = ct.astype(x, ct.float32)
```
