---
id: matmul-autotune-configs
kind: reference
title: matmul-autotune-configs (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: tech-group-swizzle
used_by: [tech-group-swizzle]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
def _matmul_autotune_configs():
    """
    Iterator of autotune configurations for matmul kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
```
