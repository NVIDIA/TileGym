---
id: ragged-bmm-scale-load
kind: reference
title: ragged-bmm-scale-load (code snapshot)
summary: Frozen code snapshot (2026-10); cited by: kernel-block-scaled-gemm
used_by: [kernel-block-scaled-gemm]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from `src/tilegym/suites/flashinfer/cutile/gemm/ragged_block_scaled_bmm.py` — illustrative evidence, not canonical source.

```python
            b_scale_block = ct.load(
                b_scale,
                index=(pid_q, offs_bsn, k),
                shape=(1, 1, 1),
                order=(0, 1, 2),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_scale_val = ct.reshape(b_scale_block, (1, 1))
```
