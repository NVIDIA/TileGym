---
id: gluact-accumulator-init
kind: reference
title: gluact-accumulator-init (code snapshot)
summary: Frozen code snapshot (2026-10); cited by: kernel-fused-linear-epilogue
used_by: [kernel-fused-linear-epilogue]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from `src/tilegym/ops/cutile/linear_gluact_linear.py` — illustrative evidence, not canonical source.

```python
    # Initialize accumulators for both branches
    acc_act = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    acc_noact = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
```
