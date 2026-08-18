---
id: kernel-scan-histogram
kind: kernel
title: Scan / Histogram (prefix-sum, cumsum, binning)
summary: Loop-carried prefix sums and contended bin counts — latency-bound shapes where tile width and carry strategy matter more than bandwidth.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Scan / Histogram (prefix-sum, cumsum, binning)

## What it computes

Two related primitives that appear on their own and as stages inside larger kernels:

- **Scan (prefix sum / cumsum):** `out[i] = f(out[i-1], in[i])` along one axis — gated log-decay
  accumulators in linear attention (`g_cum = ct.cumsum(g_raw, axis=0)` in
  `src/tilegym/ops/cutile/chunk_gated_delta_rule.py`).
- **Histogram (binning / bincount):** count occurrences of each key into a small bin array, e.g. tokens per
  expert in MoE routing (`_moe_align_block_size_stage1_kernel`,
  `src/tilegym/ops/cutile/moe_align_block.py`).

The canonical composite is `moe_align_block_size` (`src/tilegym/ops/cutile/moe_align_block.py`): histogram
over experts → padded exclusive prefix sum over bins → bucketed scatter of token ids. This
count/prefix-sum/reorder pipeline is what converts an irregular scatter problem into the dense tiled GEMM
that `moe.py` then consumes.

## Computational shape

- **Scan is a serial dependence chain, not a reduction.** Work is O(N) and data is O(N), but every element
  depends on its predecessor along the scan axis. Within a tile, `ct.cumsum` (forward or `reverse=True`)
  resolves the chain in one op; across tiles, a carry must be propagated. The in-repo pattern is a
  **chunk-carry loop**: one CTA owns a row (or feature slice), walks it in BT-sized chunks, and carries the
  running total `b_z` across chunk iterations.

- **Histogram is data-parallel reads with contended writes.** Each input element issues one atomic
  increment on a bin.

  Contention scales
  with key skew: uniformly routed tokens hit E bins evenly; a hot expert serializes on one address.
- **Bin-array phases are tiny and serial.** The padded prefix sum over E expert bins is a single-CTA
  O(E) loop with scalar gather/scatter (`_moe_align_block_size_stage3_kernel`,
  `src/tilegym/ops/cutile/moe_align_block.py`). E (expert count) is orders of magnitude smaller than the
  token count, so this stage never dominates.
- **Parallelism comes from independent rows, not from within the scan.** Grid axes are batch, heads,
  feature-blocks, chunks — the scan axis itself contributes no grid parallelism unless a multi-kernel scan
  decomposition (local scans → scan-of-block-sums → add offsets) is used.

## What dominates performance

- **Loop-carried latency, when the parallel dimensions are thin.** The scan chain is exposed latency: the
  next chunk's cumsum cannot retire until the carry arrives. On GB200, the D64 suffix (reverse) scan in the
  linear-attention suite is latency-bound on its loop-carried cumsum and runs best with the scan tile
  narrowed to BT=32, while forward scans and wider feature dims run BT=128 — both retuned from a uniform
  BT=64.

  Wider tiles amortize loads but
  lengthen the serial critical path per iteration — the tradeoff flips with scan direction and feature width.
- **Grid shape for the histogram stage.** One CTA per input element doing a single atomic maximizes
  parallelism; a serial per-CTA gather-modify-scatter loop over an `O(NUMEL/E)` chunk trades that
  parallelism for zero contention (the shipped stage 1's form). Which side wins is a measured routing
  decision: atomic throughput on a small, hot bin array is the atomic route's ceiling, not DRAM
  bandwidth.
- **Whether the carry loop is hidden.** In chunk-carry scans the loads of chunk i+1 are independent of the
  carry of chunk i; load latency hints let the compiler software-pipeline the next chunk's memory behind the
  serial arithmetic (pattern measured on persistent row loops with `latency=3` in the liger B200 batch).
- **When parallel-scan decompositions pay:** only when a *single* scan is long and the surrounding grid is
  too thin to fill the machine. If batch × heads × feature-blocks already covers the SMs (the common LLM
  case — the global-cumsum kernel launches `(cdiv(S, BS), N*H)` CTAs), a per-row sequential chunk-carry
  loop is simpler and avoids the extra kernel round-trips of reduce-then-scan. Conversely, for small bin
  arrays (E experts) a single-CTA serial pass is always cheaper than a decomposition — the whole stage is a
  few microseconds of launch-bound work.
- **Padding hygiene.** Zero the padded tail before it enters the running sum, or the carry poisons every
  later chunk (the `ct.where(m_t[:, None], b_s, 0.0)` line in the chunk-carry snippet).

## Applicable techniques

- `tech-tile-size` — scan-tile width (BT) trades load amortization against loop-carried latency;
  narrow the tile on latency-bound (reverse / small-feature) scans.
- `tech-latency-hint` — pipeline next-chunk loads behind the carry chain in chunk-carry loops.
- `tech-occupancy` — thin-grid scans leave SMs idle; extra occupancy on the surviving CTAs is the
  only latency-hiding lever left.
- `tech-copy-batching` — multi-stage pipelines (histogram → cumsum → scatter) are several tiny
  launches back-to-back; keep host prep (buffer init, index math) off the critical path.
- `tech-persistent-grid` — a persistent CTA per row keeps the carry in registers across the whole
  axis instead of re-reading state between kernels.
- `tech-scatter-to-gather-inversion` — the relaxed-atomic rank pass and binned gather-reduce build
  directly on this page's histogram/scan stages when the consumer is a duplicate-index accumulation.

## Where it lives

| What | Path |
|---|---|
| MoE align pipeline (histogram → padded cumsum → bucketed scatter) | `src/tilegym/ops/cutile/moe_align_block.py` |
| Gated-decay cumsum inside chunked linear attention | `src/tilegym/ops/cutile/chunk_gated_delta_rule.py` |
