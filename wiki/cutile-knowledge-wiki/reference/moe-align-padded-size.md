---
id: moe-align-padded-size
kind: reference
title: moe-align-padded-size (code snapshot)
summary: Frozen code snapshot (batch-1 mining, 2026-07); cited by: kernel-moe-align
used_by: [kernel-moe-align]
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

Snapshot from a production cuTile codebase — illustrative evidence, not canonical source.

```python
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
```
