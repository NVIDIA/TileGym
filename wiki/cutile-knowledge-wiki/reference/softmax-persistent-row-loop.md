---
id: softmax-persistent-row-loop
kind: reference
title: softmax-persistent-row-loop (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-softmax
used_by: [kernel-softmax]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    for row_idx in range(pid, N_ROWS, num_programs):
        # Load the row tile using index-based access
        row = ct.gather(input, (row_idx, offsets), check_bounds=True, padding_value=-math.inf)
        # Convert to float32 for computation
        row = ct.astype(row, ct.float32)
```
