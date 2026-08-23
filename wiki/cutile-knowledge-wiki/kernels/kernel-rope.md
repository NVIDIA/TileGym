---
id: kernel-rope
kind: kernel
title: RoPE (rotary position embedding)
summary: Paired 2D rotation of Q/K feature pairs by position-dependent cos/sin — elementwise-with-permutation, purely memory-bound; wins come from load/store vectorization, occupancy, and fusing into quantize/cache-append.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# RoPE (rotary position embedding)

## What it computes

Rotary position embedding rotates each (x1, x2) feature pair of Q and K by a position-dependent angle before
attention: `y1 = x1*cos - x2*sin`, `y2 = x2*cos + x1*sin`, with `cos`/`sin` precomputed per (position,
rotation index) and shared across heads. Two pairing conventions exist and are **not** interchangeable:

- **Half-split** (HuggingFace `rotate_half`): the partner of element `d` is `d + rope_dim/2`. This is what
  `src/tilegym/ops/cutile/rope.py` implements — the kernel loads the two halves as separate tiles via
  tile-space indexing (`dim_tile=0/1`) and never does a strided element shuffle.
- **Interleaved** (stride-2 pairs): partner of element `2i` is `2i+1`
  (`_apply_rope_interleave_batched` in `src/tilegym/suites/flashinfer/cutile/rope_quantize_fp8.py`).

Options carried by the tilegym kernels: partial RoPE (`rope_dim < head_dim`; the tail of the head passes
through untouched), broadcast or per-batch cos/sin tables, in-place update of Q and K in one launch, and
model-specific variants (Llama4, Qwen2-VL mRoPE with 3D multimodal position ids).

## Computational shape

Elementwise with a permutation: every output element reads exactly two input elements (itself and its pair
partner) plus one cos and one sin; there is no reduction and no reuse of x. Flop count is 4 multiplies + 2
adds per pair — negligible. The kernel is a pure memory shape: read Q and K once, write them once, stream the
(much smaller) cos/sin table.

- Grid: one CTA per token row — `grid = (batch * seq_len,)` in `_rope_forward`
  (`src/tilegym/ops/cutile/rope.py`); each CTA covers all Q heads (`TILE_QH`), all K heads (`TILE_KH`), and
  half the rotary width (`TILE_RD = next_power_of_2(rope_dim // 2)`) in two load/rotate/store rounds (Q then
  K), operating in place.
- The half-split partner sits `rope_dim/2` elements away in a head_dim-contiguous layout, so both halves are
  wide contiguous tiles — the "permutation" costs nothing when expressed as two tile loads instead of a
  gather.
- cos/sin rows are loaded once per (batch, position) and broadcast across the head axis of the tile.
- The fused serving variant processes `TOKENS_PER_BLOCK` tokens per CTA and appends the rotated+quantized K/V
  into the paged cache in the same kernel (`_rope_quantize_fp8_kernel`,
  `src/tilegym/suites/flashinfer/cutile/rope_quantize_fp8.py`).

## What dominates performance

- **Load/store efficiency.** All time is in moving Q/K through the memory system; anything that degrades the
  width or coalescing of those accesses shows up 1:1. Descriptor overhead is a real term at this size: the
  unsloth RoPE kernels replaced 4D/5D TMA loads with flat 1D gather/scatter addressing because
  high-dimensional TMA descriptor setup dominated at `half_head_dim=32` — previous measurements took all RoPE cases
  from 1.13–1.18x vs Triton down to 1.00x on B200. Conversely, plain 2D row loads/stores did
  move to TMA profitably in the liger Llama4 RoPE batch (B200). TMA-vs-gather is a
  per-shape measurement, not a rule.
- **Occupancy.** Per-CTA work is tiny, so latency hiding comes from CTA count and occupancy, not from
  intra-CTA pipelining. The ops-level kernel autotunes occupancy over (1, 7, 9, 12) — a recorded search space, i.e. a seed to re-measure on your board, not an answer — with cached
  `replace_hints`, static default 9 (B200; config tuple `_ROPE_OCCUPANCY_CONFIGS` in
  `src/tilegym/ops/cutile/rope.py`).
- **Boundary handling for odd head shapes.** Tiles are padded to powers of two (`TILE_QH`, `TILE_KH`,
  `TILE_RD`); with non-power-of-2 head dims (96, 160) unmasked lanes wrote into adjacent heads until explicit
  lane masking was added — a correctness cliff, not just a perf one (unsloth RoPE). The
  same change skips the cos/sin dtype cast when it already matches Q's dtype.
- **Launch overhead at decode.** At `seq_len=1` the whole op is a few microseconds; host-side prep
  (reshapes, table slicing, allocations) can rival the kernel. The in-place, no-host-reshape design of
  `_rope_forward` (tile-space indexing instead of host slice/cat for partial RoPE) exists for this reason.
- **Being a separate kernel at all.** RoPE's bytes are a pure tax adjacent to attention; the biggest wins are
  fusions that make the pass free (below).

## Applicable techniques

- **tech-occupancy** — high-occupancy sweep for a tiny-CTA elementwise kernel.
- **tech-tma-load** — with the explicit caveat that high-dimensional descriptors can cost more than gather at
  small tile widths; measure both paths per shape.
- **tech-copy-batching** — batch multiple tokens per CTA when tokens are small (`TOKENS_PER_BLOCK` x occupancy
  autotune space in the fused FP8 kernel; B200); tile-space partial-RoPE indexing instead of host
  reshape/slice/cat (`_rope_forward` docstring, `src/tilegym/ops/cutile/rope.py`).
- **tech-epilogue-fusion** — the high-leverage direction:
  - RoPE + FP8 quantize + paged KV-cache append in one kernel
    (`flashinfer.rope.rope_quantize_fp8`, `src/tilegym/suites/flashinfer/cutile/rope_quantize_fp8.py`).
  - Attention kernels that consume rotary dims directly (MLA prefill/decode take separate `qpe/kpe`
    inputs — `src/tilegym/ops/cutile/mla_decoding.py`) remove the standalone K-side pass.
  - In-place Q/K update avoids output allocation and a copy.
- Padded-tile stores must be masked on non-power-of-2 head dims — `padding_mode=ZERO` on loads does
  not protect the store side (see `cutile-language`, boundary rules).

## Where it lives

- `src/tilegym/ops/cutile/rope.py` — `apply_rope_base` and `get_apply_rope_func` (model-keyed wrapper);
  unified in-place kernel for full and partial RoPE (`_rope_kernel`).
- `src/tilegym/suites/liger/cutile/rope.py` (`liger.rope`),
  `src/tilegym/suites/liger/cutile/llama4_rope.py` (`liger.llama4_rope`),
  `src/tilegym/suites/liger/cutile/qwen2vl_mrope.py` (`liger.qwen2vl_mrope`) — Liger-compatible forward +
  backward variants.
- `src/tilegym/suites/unsloth/cutile/rope_embedding.py` (`unsloth.rope_embedding`,
  `unsloth.rope_embedding_qk`).
- `src/tilegym/suites/flashinfer/cutile/rope_quantize_fp8.py` (`flashinfer.rope.rope_quantize_fp8`) — fused
  RoPE + FP8 quantize + cache append.
- Triton twin: `src/tilegym/ops/triton/rope.py`.
- Tests/benchmarks: `tests/ops/test_rope.py`, `tests/suites/liger/test_rope.py`,
  `tests/suites/liger/test_llama4_rope.py`, `tests/suites/unsloth/test_rope_embedding.py`,
  `tests/suites/flashinfer/test_rope_quantize_fp8.py`, `tests/benchmark/bench_rope.py`.
