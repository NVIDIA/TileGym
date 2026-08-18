---
id: attention-exp2-ftz
kind: reference
title: attention-exp2-ftz (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-ftz-approx
used_by: [tech-ftz-approx]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
```
