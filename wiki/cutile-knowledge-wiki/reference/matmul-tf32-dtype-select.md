---
id: matmul-tf32-dtype-select
kind: reference
title: matmul-tf32-dtype-select (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-gemm
used_by: [kernel-gemm]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
```
