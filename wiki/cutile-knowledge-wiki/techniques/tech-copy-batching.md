---
id: tech-copy-batching
kind: technique
basis: ungraded-batch-1
title: Copy batching and host-overhead elimination
summary: When device time is small, launches and host-side prep dominate — batch small copy kernels into fewer launches and delete per-call host work (dead allocations, zero-inits that empty_like avoids, fresh placeholder tensors, GPU-to-CPU syncs).
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Copy batching and host-overhead elimination

## What it is

A family of host-side moves for ops whose kernels finish in microseconds: at that scale the launch count, the
allocator, and Python-side prep are the bottleneck, not the device code. The family has two wings:

- **Copy batching** — collapse many small, similar launches into fewer, larger ones: fixed-slot batched copy
  kernels, per-block token batching, and page-level (rather than element-level) gather transactions.
- **Host-overhead elimination** — delete work the device never needed: allocate with `torch.empty`/
  `empty_like` instead of `zeros` when every element is written, cache read-only placeholder tensors instead
  of allocating them per call, use in-place ops instead of materialized temporaries, vectorize host index
  prep, and remove GPU→CPU syncs from the launch path.

## Pattern

**Fixed-slot copy batching**: give the kernel one input
view + metadata tuple per slot, branch on `ct.bid(2)` to select the active slot, keep the `ct.load`/`ct.store`
tile shape fixed after the branch; on the host, pack entries per launch up to a fixed slot count (a `ct.Constant`), pad unused slots
with a valid dummy view, and sweep the slot count over {2, 4, 8}.

**Token batching**: process TOKENS_PER_BLOCK tokens per block and make that factor an autotune dimension
(rope_quantize_fp8). **Page-level gather**: load paged KV with `ct.load_advanced_indexing`
(sparse_dim=0) so the copy issues one transaction per page instead of one per token.

**Host-side deletions**:

```python
# Every (block_id, col) is written exactly once by the kernel -> zero-init is dead work.
dW_partial = torch.empty(sm_count, n_cols, ...)        # was torch.zeros(...)

# Read-only placeholders: allocate once per (device, dtype), not per forward.
@functools.lru_cache(maxsize=None)
def _zero_placeholder(device, dtype):
    return torch.zeros(1, device=device, dtype=dtype)

# In-place accumulate instead of materializing a temporary matmul result.
grad_weight.addmm_(a.t(), b)                           # was add_(a.t() @ b)
```

Compute host-side constants arithmetically, not by loop (`2**ceil(log2(n))`); build tile mappings
with torch ops/cumsum instead of Python loops; pre-pack layouts the kernel's fast path consumes so
the kernel reinterprets without a copy; make the grid independent of data so no `.tolist()`
GPU→CPU sync sits on the launch path.

## When to use

- Profiles show gaps between kernels, or a shape's total time is dominated by many tiny launches
  (launch-bound decode/small-batch shapes are the classic case).
- The op launches one similar copy kernel per segment/input, each doing little work, and the copies are
  regular enough for `ct.load`/`ct.store` — batch them.
- Host code allocates zeroed buffers that the kernel fully overwrites, or re-creates constant dummy/placeholder
  tensors every call — each such allocation is an extra fill/alloc launch.
- The launch path calls `.item()`/`.tolist()` or otherwise syncs the GPU to compute the grid.

## Caveats

- `empty` instead of `zeros` is correct only when every element is provably written (exactly-once scatter or
  full-tile store); a partially-written buffer silently reads garbage. Document the invariant at the alloc
  site.
- Cached placeholder tensors must be read-only; a kernel that writes into a shared cached dummy corrupts every
  later caller. Key the cache by (device, dtype).
- In-place `addmm_` requires matching dtypes; keep the out-of-place fallback branch.
- Batched copy kernels can lose store vectorization: dynamic output slices may drop the alignment facts that
  kept stores at `STG.E.128`, scalarizing them to `STG.E.U16`. If the host can prove slice bounds divisible by
  the needed alignment, pass the divisor as a constant and re-materialize the bounds before the dynamic
  `Array.slice`; benchmark both — launch savings can be canceled by scalarized stores.
- Branch-selected slot views can hit type-compatibility checks; split incompatible cases into host buckets.
- These wins are per-shape-regime: overhead savings that dominate a tiny launch-bound shape are noise on large
  shapes, so measure the full matrix.

## Evidence
- unsloth grouped_gemm fwd/dX: persistent grid removed host-side tile mapping, its caching, and the GPU→CPU sync from `m_sizes.tolist()`; benchmarked under CUDA Graph.
- flashinfer prefill paged KV (B200): `ct.cat` of per-page loads replaced by `ct.load_advanced_indexing` page-level gather — NUM_PAGES transactions instead of BLOCK_N token-level ones; same pattern on 3D/4D decode caches. [2026-07]
- Other in-repo instances (B200): liger rms_norm bwd (`torch.empty` for a fully-written partial buffer), fused_linear_jsd (in-place `addmm_`), liger FLCE (one upstream d_logits scale replacing per-gradient launches), unsloth suite (vectorized tile mapping, arithmetic BLOCK_SIZE), rope_quantize_fp8 (TOKENS_PER_BLOCK batching), block-scale matmul (host 5D scale pre-pack). [2026-07]
