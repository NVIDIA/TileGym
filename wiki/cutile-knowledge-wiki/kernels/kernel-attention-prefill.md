---
id: kernel-attention-prefill
kind: kernel
title: Attention prefill (FMHA forward)
summary: Fused multi-head attention forward over full query sequences — online softmax over a KV-tile loop; two MMAs per iteration; compute-bound at long context with softmax as the non-MMA tax.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Attention prefill (FMHA forward)

## What it computes

`O = softmax(Q @ K^T * scale) @ V` per (batch, head), over a full query sequence — the prompt-processing
phase of LLM inference and the forward pass of training. Alongside `O` it emits the per-row log-sum-exp
(`L`/LSE), which the backward pass and split/merge schemes consume. Standard options are all handled in the
same kernel family: causal and sliding-window masks, GQA/MQA (`QUERY_GROUP_SIZE` maps query heads onto shared
KV heads), bias variants (vector/matrix/alibi), per-batch actual sequence lengths (varlen/ragged), soft-cap
variants (tanh-capped logits, e.g. Gemma), and paged KV caches for serving (chunked prefill against an
existing cache via `block_tables` indirection).

## Computational shape

Two chained GEMMs with a reduction (softmax) welded between them, tiled so the intermediate `S = QK^T` matrix
never touches global memory:

- Grid: one CTA per (query tile, batch x head). Dense form:
  `grid = (ceil(seq_len_q / TILE_M), batch * num_heads_q, 1)` (dense-form
  host launcher). The paged form adds the sequence block as a third axis and looks sequence lengths up per
  batch from `actual_seq_lens_q/kv`.
- Each CTA pins a `[TILE_M, D]` query tile in registers and loops over KV in `[TILE_N, D]` chunks. Per
  iteration: MMA #1 `S = q @ k^T` (`[TILE_M, TILE_N]` fp32 accumulator), online-softmax update, MMA #2
  `acc += p @ v` (`[TILE_M, D]` accumulator).
- Online softmax carries a running row max `m_i` and running row sum `l_i`; each iteration rescales the
  accumulator by `alpha = exp2(m_i - m_ij)` with the `1/log(2)` factor folded into the QK scale so all
  exponentials are `exp2` (`_prefill_attention_paged_body` in
  `src/tilegym/suites/flashinfer/cutile/fmha_prefill_bsr.py`).
- Work per (batch, head) is O(S_q x S_kv x D) MMA flops — quadratic in context length; causal masking halves
  it. Q is read once per CTA; K and V are re-read once per query tile, so KV traffic scales with
  `seq_len_q / TILE_M`.
- Paged-KV prefill resolves `block_tables` per KV block and gathers whole pages
  (`_load_page_prefill` / `_load_page_wrapper_prefill` in `fmha_prefill_bsr.py`).

## What dominates performance

- **MMA throughput at long context.** The quadratic QK^T/PV term swamps everything else; the kernel is
  compute-bound once S_kv is large. The exponent/rescale chain is the fixed non-MMA tax on the critical loop —
  implementations keep it in `exp2` form with `flush_to_zero=True` and use approximate rounding on the final
  reciprocal (`RMd.APPROX` epilogue in `fmha_prefill_bsr.py`). Soft-cap variants put a `tanh` in the inner
  loop; running it with `rounding_mode=APPROX` targets the hardware MUFU path (gemma attention).
- **Causal-mask specialization.** Only diagonal query/KV tiles need the mask. The paged prefill body runs one
  unified KV loop with a uniform scalar branch (`curr_n >= start_m`) so the unmasked prefix pays zero mask
  overhead and there is no warp divergence (comment block in `_prefill_attention_paged_body`,
  `fmha_prefill_bsr.py`).
- **KV feed rate.** The KV loop must stay ahead of the MMAs. Paged loads carry `allow_tma=True` with
  `latency=2` (B200, flashinfer decode/prefill suite). Multi-page KV tiles are
  fetched as a page-level gather (`ct.load_advanced_indexing` with the page-id vector as the dim-0 index —
  the sparse dim is inferred from which index is a vector, not passed as a kwarg) issuing NUM_PAGES transactions
  instead of BLOCK_N token-level ones — replacing a `ct.cat` of per-page loads (B200, fmha_prefill_bsr).
- **Tile shape vs register pressure.** `TILE_M x TILE_N` trades accumulator/register footprint against how
  often KV is re-read and how well the diagonal is amortized. Config families are per-architecture: the fmha op's space
  (`_fmha_autotune_configs` in `src/tilegym/ops/cutile/attention.py`) reaches 256x128 on sm100+ and got its
  own sm90 64/128-tile set with `num_ctas=1`, occupancy=2; pre-sm90 caps the default
  prefill tiles at 128x64 and its autotune branches pin `num_ctas=1` because CGA is
  unsupported there.
- **Grid fill at short/ragged sequences.** With few query tiles the grid underfills the GPU; the paged suite
  ships a longest-processing-time variant (`_prefill_attention_paged_lpt_kernel`, `fmha_prefill_bsr.py`) that
  reorders work for ragged batches, and occupancy is part of the autotune space rather than fixed.
- **Loop structure vs warp specialization.** Keeping the K/V loads inside the `ct.mma` loop (rather than
  hoisting/prefetching them manually) avoided a warp-specialization hang in the fused
  neighborhood-attention kernel (B200, liger FNA).

## Applicable techniques

- **tech-tma-load** — TMA for Q/O tile loads/stores and paged KV loads (`allow_tma=True` throughout
  `fmha_prefill_bsr.py`).
- **tech-latency-hint** — `latency=2` on paged KV/Q/O accesses to software-pipeline the KV loop.
- **tech-copy-batching** — page-level gather TMA for multi-page KV tiles.
- **tech-tile-size** — TILE_M/TILE_N selection per arch and per problem size; register-pressure ceiling on the
  fp32 accumulators.
- Per-arch config families (sm80/sm90/sm100) keep occupancy and tile shape as the search dimensions;
  `num_ctas` stays 1 everywhere except one 256x128 `num_ctas=2` sm100+ config in the fmha op's
  space (`src/tilegym/ops/cutile/attention.py`); the sm90 and pre-sm90 spaces pin `num_ctas=1`.
- **tech-occupancy** — swept over {1, 2, 4} jointly with tile shape; introduced with the paged prefill kernel's
  space.
- **tech-softmax-max-elision** — the running max exists only to bound the exp2 argument; on
  range-proven inputs the elision rungs delete the per-tile rescale chain.
- Cross-notes: the thin-tile `num_ctas` measurement on the attention-decode page (prefill tiles are
  wide enough that it does not apply there); keep K/V loads inside the `ct.mma` loop — hoisting them
  produced a warp-specialization hang on one fused kernel (liger FNA, B200).

## Where it lives

- `src/tilegym/ops/cutile/attention.py` — `fmha` / `fmha_backward` ops.
- `src/tilegym/suites/flashinfer/cutile/fmha_prefill_bsr.py` — serving prefill:
  `flashinfer.attention.prefill_attention_kv_paged` and `...kv_ragged`, plus the LPT load-balanced variant.
- Soft-cap / variant kernels: `src/tilegym/ops/cutile/gemma_attention.py`,
  `src/tilegym/ops/cutile/attention_sink.py`, `src/tilegym/ops/cutile/mla.py` (MLA prefill).
