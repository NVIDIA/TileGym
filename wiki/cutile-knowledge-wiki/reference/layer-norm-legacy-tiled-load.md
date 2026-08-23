---
id: layer-norm-legacy-tiled-load
kind: reference
title: layer-norm-legacy-tiled-load (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-norms
used_by: [kernel-norms]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        x_tile = ct.load(X, index=(current_pid, 0), shape=(BLOCK_N, BLOCK_D), padding_mode=PAD_ZERO, latency=4)
        x = ct.astype(x_tile, ct.float32)

        if COMPUTE_MEAN_AND_RSTD:
            x_squared = x * x
            avg_square = ct.sum(x_squared, axis=1) / D
            mean = ct.sum(x, axis=1) / D
            var = avg_square - mean * mean
            rstd = ct.rsqrt(var + EPS)
            if TRAINING:
                ct.store(Mean, index=(current_pid,), tile=mean, allow_tma=False)
                ct.store(Rstd, index=(current_pid,), tile=rstd, allow_tma=False)
```
