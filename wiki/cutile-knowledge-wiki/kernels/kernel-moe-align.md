---
id: kernel-moe-align
kind: kernel
title: MoE align-block-size (token-routing auxiliaries)
summary: Integer histogram/cumsum/scatter pipeline that groups MoE token ids by expert and pads per-expert counts to the GEMM block size; launch count, per-stage grid choice, and atomics-vs-tiled-reduction dominate.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# MoE align-block-size (token-routing auxiliaries)

## What it computes

`moe_align_block_size(topk_ids, block_size, num_experts)` prepares the routing metadata that a fused
MoE GEMM consumes. Input is `topk_ids` of shape `[num_tokens, top_k]` (expert index per token slot);
outputs are:

- `sorted_token_ids` — flat token-slot indices grouped by expert, each expert's group padded up to a
  multiple of `block_size`; padding slots hold the sentinel value `numel` so the GEMM can skip them.
- `expert_ids` — for each output block of `block_size` rows, the expert that owns it.
- `num_tokens_post_pad` — total row count after padding.
- `cumsum` — exclusive prefix sums of *padded* per-expert counts (per-expert write offsets).
- `max_expert_cnt` — max pre-padding token count over experts.

Downstream contracts constrain the kernel: `sorted_token_ids` is allocated at the worst case and
sentinel-filled on the host, so no output shape depends on data (CUDA-graph friendly) — see `reference/moe-align-padded-size.md`.

and `block_size` must equal the fused MoE GEMM's `BLOCK_SIZE_M` — a fixed interface contract
between the two ops (the downstream GEMM statically asserts the match).

## Computational shape

A four-stage integer pipeline with data dependencies that force global synchronization (separate
launches) between stages:

```
topk_ids [T, K]  (flattened to NUMEL = T*K int32 ids)
   |
   v
1. histogram        counts[e] = #slots routed to expert e
   |
   v
2. padded cumsum    cumsum[e] = sum of ceil(counts[<e]/BS)*BS; also total, max   (O(E), serial)
   |
   v
3. block table      expert_ids[b] = e  for blocks b in [cumsum[e]/BS, cumsum[e+1]/BS)
   |
   v
4. rank + scatter   sorted_token_ids[cumsum[e] + rank(i within e)] = i   (stable: ascending i)
```

There is no floating point and no data reuse; the entire working set is `NUMEL` int32 ids plus
`O(E)` counters. The op is launch- and latency-bound rather than bandwidth-bound — though a
decomposition can self-inflict traffic (an earlier stage-4 form re-read all `NUMEL` ids per expert).
The cuTile backend runs exactly four launches (`_moe_align_block_size` host launcher,
`src/tilegym/ops/cutile/moe_align_block.py`).
The stable-order requirement in stage 4 (tokens within an expert must stay in ascending token-index
order, matching the reference) is what makes stage 4 the structurally hardest stage: a scatter needs
each slot's *rank* within its expert, which is itself a prefix computation.

## What dominates performance

**Decomposition, not instruction tuning.** With almost no arithmetic, the only real levers are how the
four stages map onto grids, how the histogram and rank computations are routed (atomics vs tiled
reduction), and how many launches the pipeline needs. The cuTile backend's current choices:
stage 1 counts each program's `ceil(NUMEL/E)`-token chunk into a private counts row by serial
gather-modify-scatter at grid `(E,)` (no atomics); stage 2 prefix-sums the per-chunk counts per
expert (grid `(E,)`, `ct.cumsum` down the chunk axis); stage 3 runs the serial `O(E)` padded cumsum
at grid `(1,)`; stage 4 (grid `(E,)`) fills each expert's block-table range and scatters each
chunk's tokens, ranked by chunk-prefix count plus the expert's padded cumsum. The bullets below map the design space each of those choices sits in:

- **Atomics vs tiled reduction (histogram).** One-CTA-per-token atomics maximize parallelism but
  serialize on hot experts and pay a `NUMEL`-sized grid of near-empty CTAs. The tiled route
  partitions tokens into chunks (one program per chunk, `tokens_per_thread = ceil(NUMEL/E)`), builds
  a private histogram with a broadcast compare `ids[:, None] == expert_idx[None, :]` reduced along
  the token axis, and writes one row of a counts matrix — E-fold redundant compares per token, zero
  contention. Which side wins depends on token count and expert skew; the routing question recurs in
  any histogram-shaped kernel.
- **Stable rank (scatter).** The stable-order contract forces either a per-expert serial scan of all
  ids (simple, `O(E * NUMEL)` read traffic, every CTA re-reads the whole id array) or per-chunk
  exclusive cumsum ranks over the match matrix combined with cross-chunk running offsets (more index
  algebra, one read per id per program). The chunk width bounds the per-program working set
  (cap it at a power of two around 128; wider chunks inflate the match matrix past register budgets).
- **Launch floor.** At small token counts every stage finishes in the launch shadow; wall time is
  the number of launches times per-launch overhead. Gains there come from fusing stages (e.g. block
  table into the expert-grid scatter stage), not from faster kernels.
- **Sentinel/padding contract.** Host pre-fills the output with the sentinel and sizes it worst-case
  (the allocation snippet under "What it computes"); a kernel that fills only each expert's real
  slots must keep the padding-slot contract intact for the consuming GEMM.

## Applicable techniques

- **Kernel decomposition / grid shaping** — pick the grid axis (tokens, token chunks, experts,
  output blocks) per stage; the axis determines both parallelism and traffic.
- `tech-scatter-to-gather-inversion` — the rank pass (relaxed `atomic_add` return value as bin
  slot) and the binned gather-reduce cover stages 1 and 4; atomics-vs-tiled-reduction is the same
  routing decision, with per-partition private histograms plus a reduce stage replacing contended
  global atomics. The `(chunk, E)` broadcast-compare match matrix turns scalar per-id control flow
  into dense tile math (histogram and rank both use it).
- `kernel-scan-histogram` — the histogram and prefix-sum stage patterns (private histograms,
  chunk carries).
- Counting-sort collapse: when the key domain is small (expert ids), the sort degenerates to one
  histogram + scan + scatter pass — see `kernel-scan-histogram` for the stage patterns.
- `tech-copy-batching` — launch-count reduction: fuse adjacent stages that share a grid (e.g. fold
  the block-table stage into the scatter stage's per-expert program) once per-stage kernels are near
  the launch floor; keep host prep vectorized.
- **Masked scatter with sentinel redirect** — a masked scatter writes only owned slots while
  preserving the host sentinel fill.

## Where it lives

- Dispatch: `moe_align_block_size` in `src/tilegym/ops/moe_interface.py`.
- cuTile implementation: `src/tilegym/ops/cutile/moe_align_block.py`.
- Consumer: fused MoE GEMM, `src/tilegym/ops/moe_interface.py` (the consuming GEMM statically
  asserts the block-size match).
