---
id: bmm-group-swizzle
kind: reference
title: bmm-group-swizzle (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-group-swizzle
used_by: [tech-group-swizzle]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
        num_bid_m = ct.cdiv(M, TILE_M)
        num_bid_n = ct.cdiv(N, TILE_N)
        bid_q = current_bid // (num_bid_m * num_bid_n)
        num_bid_in_group = GROUP_SIZE_M * num_bid_n

        current_bid_2d = current_bid % (num_bid_m * num_bid_n)
        group_id = current_bid_2d // num_bid_in_group
        first_bid_m = group_id * GROUP_SIZE_M
        group_size_m_temp = num_bid_m - first_bid_m
        group_size_m = ct.minimum(group_size_m_temp, GROUP_SIZE_M)
        bid_m = first_bid_m + (current_bid_2d % group_size_m)
        bid_n = (current_bid_2d % num_bid_in_group) // group_size_m
```
