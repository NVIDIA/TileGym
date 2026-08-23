---
id: attention-truediv-approx
kind: reference
title: attention-truediv-approx (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-ftz-approx
used_by: [tech-ftz-approx]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```
