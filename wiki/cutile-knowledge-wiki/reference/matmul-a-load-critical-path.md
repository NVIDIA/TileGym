---
id: matmul-a-load-critical-path
kind: reference
title: matmul-a-load-critical-path (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-gemm
used_by: [kernel-gemm]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
            # Load A tile (tuned: A's load cost is also on the critical path — tuning
            # both A and B reaches 6.74 ms vs 7.27 ms for B-only, ~7-8% better)
```
