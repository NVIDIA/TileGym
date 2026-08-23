---
id: sm100_matmul_num_ctas_gating
kind: reference
title: sm100_matmul_num_ctas_gating (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-num-ctas
used_by: [tech-num-ctas]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    else:
        # sm100+ (Blackwell)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=4, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1)
```
