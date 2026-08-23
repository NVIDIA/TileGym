---
id: softmax-bounds-check
kind: reference
title: softmax-bounds-check (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-softmax
used_by: [kernel-softmax]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    check_bound = TILE_SIZE != N

    row = ct.gather(input, (row_idx, offsets), check_bounds=check_bound, padding_value=-math.inf)
```
