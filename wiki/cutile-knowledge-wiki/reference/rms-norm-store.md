---
id: rms-norm-store
kind: reference
title: rms-norm-store (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-norms
used_by: [kernel-norms]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        ct.store(
            Y,
            index=(current_bid, 0),
            tile=y,
            allow_tma=False,  # +30% perf
            latency=3,  # +3% perf from this hint
        )
```
