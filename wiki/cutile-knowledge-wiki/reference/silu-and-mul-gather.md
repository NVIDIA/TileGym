---
id: silu-and-mul-gather
kind: reference
title: silu-and-mul-gather (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-epilogue-fusion
used_by: [tech-epilogue-fusion]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Implement sigmoid for SiLU
    denom = 1 + ct.exp(-a_tile)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # Perform SiLU(a) * b
    silu_a = a_tile * sigmoid_a
    result = silu_a * b_tile
    result = ct.astype(result, output.dtype)
    # output is also 2D: (batch_size, hidden_size)
    out_col_idx = offsets
    ct.scatter(output, (row_idx, out_col_idx), result, check_bounds=True)
```
