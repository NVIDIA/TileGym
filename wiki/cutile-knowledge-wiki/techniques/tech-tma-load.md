---
id: tech-tma-load
kind: technique
basis: ungraded-batch-1
title: TMA loads and stores (replace gather/scatter)
summary: Route contiguous or block-aligned memory access through ct.load/ct.store (hardware TMA) instead of software gather/scatter; the single most impactful cuTile memory choice.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# TMA loads and stores (replace gather/scatter)

## What it is

`ct.load`/`ct.store` issue Tensor Memory Accelerator (TMA) copies: the hardware unit performs address
computation, coalescing, and out-of-bounds padding for a whole tile in one transaction. `ct.gather`/`ct.scatter`
compute per-element addresses in software. This is the single most impactful cuTile memory choice —
measured swings across in-repo conversions span 2-78x.

The two APIs have different index semantics: TMA `index=` is a **block index** (which tile), while gather takes
**element offsets**. TMA requires block-aligned, contiguous tiles and has a hardware limit of ~16K elements per
load; gather has no such limit but is software-computed.

## Pattern

```python
# Before: software gather/scatter
indices = bid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
x = ct.gather(X, indices, check_bounds=True)
ct.scatter(Y, indices, result, check_bounds=True)

# After: TMA — index is the BLOCK index, NOT an element offset
x = ct.load(X, index=(bid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
ct.store(Y, index=(bid,), tile=result)

# Ragged/variable-length: Array.slice makes TMA legal on a runtime sub-range
seg = X.slice(axis=0, start=start, stop=start + length)
x = ct.load(seg, index=(bid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)

# Paged/indirect: extract a scalar page id, then TMA
page_id = ct.gather(block_table, (bid,), padding_value=0).item()
x = ct.load(X, index=(page_id, 0), shape=(1, BLOCK), allow_tma=True)

# Many small gathers over ONE contiguous region (sub-blocks / recurrence rows,
# e.g. a triangular-solve merge): ONE bulk TMA load, then extract sub-tiles —
# do NOT recompute the gathered values in-register on the dependency chain
blk = ct.load(A, index=(0, 0), shape=(64, 64))            # one transaction
sub = ct.extract(blk, index=(i, j), shape=(16, 16))       # per sub-block
# ... serial recurrence over sub-tiles ...
# assemble pairwise — ct.cat concatenates exactly TWO tiles per call
ct.store(Y, index=(0, 0), tile=ct.cat((left, right), axis=1))  # one assembled store
```

## When to use

- Default choice — TMA-first. Fall back to gather/scatter only for truly sparse or random
  per-element access.
- Block-aligned contiguous tiles: direct `ct.load`/`ct.store`.
- Ragged segments: `Array.slice` with runtime start/stop, then TMA on the sliced view.
- Paged or indirect access (KV caches, block tables): `ct.gather(...).item()` to get the scalar page id, then
  `ct.load(..., allow_tma=True)` on the page.
- Multi-page loads in one kernel: prefer a page-level gather TMA (`ct.load_advanced_indexing` with a sparse
  dim) over concatenating per-page loads.
- Many small gathers over one contiguous region (sub-blocks, recurrence rows): ONE bulk load + `ct.extract`
  per sub-tile, `ct.cat` (pairwise — it takes exactly two tiles) to assemble the store — not per-row
  gathers, and **not in-register recomputation** of the gathered values.

## Caveats

- **Padding values**: TMA pads via `padding_mode` and cannot inject arbitrary values such as -inf directly;
  pad `ZERO` and re-inject the sentinel with `ct.where` (see the cross_entropy evidence below).
- **Descriptor overhead**: high-dimensional (4D/5D) TMA descriptor setup can dominate small loads; a flat 1D
  gather/scatter path can win (see the RoPE evidence below).
- **Ampere (sm80/sm86)**: no TMA hardware — `allow_tma=True` falls back to `cp.async` emulation with ~8-15%
  overhead, and the emulation does **not** zero-fill out-of-bounds lanes. Either pass
  `padding_mode=ct.PaddingMode.ZERO` on any load that can go OOB, or route Ampere through the
  non-TMA path (silent-corruption risk otherwise).
- **Stores are a separate decision**: TMA-vs-direct for scalar and thin row stores flips sign per kernel; see
  `tech-tma-store-disable`.
- A TMA conversion that wins in one benchmark run may not hold up across the full matrix or CI — treat each
  conversion as a measured change, not a rule: a gather-B load converted to a transposed TMA
  `ct.load(order=(1,0))` won its first benchmark and was reverted after CI (fp8 quantized GEMM, B200).
- **A gather is only worth eliminating if it costs anything**: a recurrence-independent gather that is
  already pipelined behind the dependency chain can be effectively free — replacing it with in-register
  recomputation *on* the serial chain can be slower than leaving the gathers in place. Check the
  kernel_times profile before rewriting; the enemy is time on the critical path, not the word "gather".
- Per-load hardware limit ~16K elements: split larger tiles or chunk the loop.
- **Odd-row-count bf16 matrices: load 16-byte-aligned 4-row (or 8-row) GROUPS** instead of single rows —
  grouped loads run at full TMA rate and remove the odd-parity penalty; the measured grouped-load vs
  gather crossover sat at ~30-50 MB of matrix footprint.

## Evidence
- Quad-row and octo-row grouped loads on odd-row bf16 weights: full TMA rate restored on two kernels; grouped-vs-gather crossover ~30-50 MB. Previous measurements. [2026-07, B200, cuda-tile 1.2.0, N=2 kernels]
- B200, unsloth cross_entropy_loss fwd: `ct.gather` replaced by TMA `ct.load` with `PaddingMode.ZERO` + `ct.where` re-injecting -inf, because TMA cannot pad -inf directly; a later pass put this row load back to a bounds-check-eliminated gather — the -inf padding pattern stands, but each TMA row-load conversion stays a per-kernel measured choice. [2026-07]
- B200, unsloth RoPE: 4D/5D TMA loads replaced by flat 1D gather/scatter because descriptor setup dominated at `half_head_dim=32`; previously measured: all RoPE cases went from 1.13-1.18x to 1.00x vs Triton. [2026-07]
- Other in-repo instances (B200): flashinfer paged loads (`allow_tma=True`, latency=2) and the page-level gather TMA replacing `ct.cat` of per-page loads; liger 2D row loads/stores and 1D weight loads; unsloth w8a8 TMA kernel and CE backward load. [2026-07]
