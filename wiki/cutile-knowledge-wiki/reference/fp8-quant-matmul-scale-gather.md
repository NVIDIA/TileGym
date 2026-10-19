---
id: fp8-quant-matmul-scale-gather
kind: reference
title: fp8-quant-matmul-scale-gather (code snapshot)
summary: Frozen code snapshot (2026-10); cited by: kernel-block-scaled-gemm
used_by: [kernel-block-scaled-gemm]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from `src/tilegym/ops/cutile/fp8_quantization_matmul.py` — illustrative evidence, not canonical source.

```python
            # Per-block scales via gather (latency=4 tuned for scale loads)
            a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0, latency=4)
            b_s = ct.gather(Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0, latency=4)
```
