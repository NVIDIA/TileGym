---
id: gluact-linear-sigmoid
kind: reference
title: gluact-linear-sigmoid (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-ftz-approx
used_by: [tech-ftz-approx]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
def _sigmoid_ct(x, BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int]):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    one = ct.full((BLOCK_M, BLOCK_N), 1.0, dtype=ct.float32)
    neg_x = -x
    exp_neg_x = ct.exp(neg_x)
    denom = one + exp_neg_x
    return ct.truediv(one, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)
```
