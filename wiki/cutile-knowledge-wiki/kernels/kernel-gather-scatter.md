---
id: kernel-gather-scatter
kind: kernel
title: Gather / Scatter (embedding, index-select, paged access)
summary: Index-driven data movement — perf is decided by coalescing along the dense axis, index materialization cost, and picking gather vs TMA per access pattern.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Gather / Scatter (embedding, index-select, paged access)

## What it computes

Data movement where the addresses come from another tensor:

- **Embedding lookup:** rows of a weight table selected by an id tensor — BERT's three-table lookup fused
  with layer norm (`ct.gather(word_embedding_weight, (input_weight_index, col_offsets))`).
- **Index-select / permuted rows:** MoE expert GEMM reads token rows of A through `sorted_token_ids`
  and scatters output rows back through the same permutation
  (`src/tilegym/ops/cutile/moe.py`).
- **Paged access:** KV-cache blocks addressed through a page table — decode/prefill attention gathers
  whole pages via `ct.load_advanced_indexing` with the page-id vector indexing dim 0 (the sparse dim;
  cuTile infers it from which index is a vector vs a `ct.Slice` — there is no `sparse_dim` kwarg)
  (`src/tilegym/suites/flashinfer/cutile/fmha_decode_bsr.py`,
  `src/tilegym/suites/flashinfer/cutile/fmha_prefill_bsr.py`).
- **Scalar picks:** one element per row at a data-dependent column, e.g. cross-entropy label logits.

## Computational shape

- Arithmetic intensity is near zero: performance *is* achieved bytes per second plus whatever the
  indirection costs. There are two address streams — the (small) index tensor and the (large) payload —
  and the payload's row order is arbitrary while each row is internally contiguous.
- The dense axis is the feature/row dimension: indices pick *which* row; lanes within the tile should cover
  the row contiguously. Both embedding and MoE kernels build 2D index tiles as
  `row_index[:, None] * stride + col_offsets[None, :]` so the fast-moving lane dimension stays coalesced.
- Scatter is the same shape with write hazards added: disjoint index sets (a permutation, as produced by
  `moe_align_block_size`) need no atomics; colliding keys need `ct.atomic_add`, a pre-sort that converts
  the problem into dense tiles (see `kernel-scan-histogram`), or inversion into a binned
  gather-reduce (`tech-scatter-to-gather-inversion`).
- Paged access adds one indirection level: page id from the page table, then a dense block within the page.
  The natural transfer unit is the page, not the token.

## What dominates performance

- **Coalescing along the contiguous dimension.** The gather index tile fixes the memory transaction
  pattern. Keep the per-row access dense and the tile's lane dimension on the row axis; the row-selection
  axis can be arbitrary without penalty because each row is a separate transaction anyway.
- **Index materialization cost.** Building a `(TILE_M, TILE_K)` flat-index tile costs register work per
  element and can exceed the payload cost for tiny payloads. Scalar picks should be scalar gathers: unsloth
  cross-entropy label extraction replaced a mask-and-`ct.sum` over a BLOCK_SIZE=32768 tile with one scalar
  `ct.gather((row, label))` — O(1) instead of O(BLOCK_SIZE).
- **Gather vs TMA is a per-kernel measurement, not a rule.** Both migration directions are real, measured moves:
  - TMA loses when descriptor setup dominates small transfers: unsloth RoPE replaced 4D/5D TMA loads with
    1D gather/scatter flat addressing at half_head_dim=32, taking all RoPE cases from 1.13–1.18x to 1.00x
    vs Triton on B200.
  - Converting a gather to a TMA load changes padding semantics: TMA cannot pad `-inf`, so the
    cross-entropy fwd conversion used `PaddingMode.ZERO` plus `ct.where` re-injection.
- **Batch the indirection at the coarsest grain.** For paged KV, gather *pages*, not tokens:
  `ct.cat` of per-page loads → `ct.load_advanced_indexing` with the page-id vector as the dim-0 index issues NUM_PAGES
  transactions instead of BLOCK_N token-level ones (B200, flashinfer prefill); decode uses the
  same page-level gather TMA on 3D/4D caches, with `allow_tma=True` + `latency=2` on the paged
  loads.
- **Bounds handling has a measurable cost.** `check_bounds=True` on gather/scatter replaced a
  manual offset-clamp + `ct.where` masking pattern in unsloth swiglu/geglu with large measured swings on B200
  (bf16 (2,2048,4096): swiglu_fg 0.93x→1.36x, swiglu_bwd 0.50x→1.00x vs Triton). For scatter,
  bounds-checking must see the real tensor shape: MoE keeps C two-dimensional because a flattened buffer
  lets out-of-range column offsets alias into the next row silently — the in-file comment at the
  output scatter — see `reference/moe-scatter-offsets.md`.
- **Latency hiding is the only remaining lever.** With no math to overlap, throughput comes from
  outstanding loads: persistent row loops with gather `latency` hints pipeline the next row's lookup behind
  the current row's compute (`latency=3` on B200 liger row kernels; gather `latency=10` in
  grpo_loss), and embedding-style kernels run persistent grids so index loads for row i+1
  overlap the normalize/store of row i.

## Applicable techniques

- `tech-tma-load` — try TMA where indices are affine after a reshape; expect it to lose to gather on
  small transfers behind high-dimensional descriptors.
- `tech-copy-batching` — page-level / multi-row gathers over element-level ones; index preprocessing
  (sorting, tile maps, cumsum offsets) belongs in vectorized torch ops on the host, not Python loops.
- `tech-scatter-to-gather-inversion` — duplicate-index atomic accumulation inverts into a two-phase
  binned gather-reduce when contention or determinism matters.
- `tech-latency-hint` — deep pipelining on gather loads; memory-bound kernels tolerate large hints.
- `tech-occupancy` — bandwidth-bound kernels want warps in flight, not registers; higher-occupancy
  spaces paid off for the large-vocab CE gather kernels.
- `tech-persistent-grid` — amortize index/weight setup across rows (the persistent row loop above).

## Where it lives

| What | Path |
|---|---|
| MoE expert GEMM: gathered A rows, scattered C rows | `src/tilegym/ops/cutile/moe.py` |
| Token sorting that feeds the MoE gather | `src/tilegym/ops/cutile/moe_align_block.py` |
| Paged-KV decode / prefill attention (page-level gather TMA) | `src/tilegym/suites/flashinfer/cutile/fmha_decode_bsr.py`, `src/tilegym/suites/flashinfer/cutile/fmha_prefill_bsr.py` |
