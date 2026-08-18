---
id: kernel-attention-decode
kind: kernel
title: Attention decode (single/few-token FMHA)
summary: Autoregressive decode attention — one or few query tokens against a long cached KV; memory-bound with thin MMA tiles; split-KV for grid fill; paged-KV gather and operand-swap tricks decide throughput.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Attention decode (single/few-token FMHA)

## What it computes

The per-step attention of autoregressive generation: for each sequence, one (or a few) new query tokens attend
to the entire cached KV history — `O = softmax(q @ K_cache^T * scale) @ V_cache` with `q` of shape
`[num_q_heads, head_dim]` per sequence. Flavors share the loop but change the head/KV geometry:

- **GQA/MQA grouped decode** — `QUERY_GROUP_SIZE` query heads share each KV head; the group is packed into
  the tile as `BLOCK_H` rows.
- **MLA decode** — all query heads attend to a single compressed latent KV (a "nope" latent part plus a
  separate small RoPE part, loaded and matmul'd separately per block); with weight absorption the score/value
  matmuls run in the latent dimension (`_naive_absorb_mla_transpose_kernel`,
  `src/tilegym/ops/cutile/mla_decoding.py`).
- **Paged-KV serving decode** — KV lives in a paged cache addressed through `block_tables`, with per-batch
  `actual_seq_lens`.
- **Variant heads**: soft-cap decode (`gemma_attention_decode`), attention-sink decode
  (`attention_sink_decode`).

Output is `[batch, num_q_heads, head_dim]` plus optional LSE (needed whenever partial results are merged).

## Computational shape

A dot-product reduction over the KV axis wearing an MMA costume:

- Grid: `(batch, head blocks, kv splits)` — `_decode_attention_kv_paged_kernel` reads
  `batch_id = ct.bid(0)`, `head_block_id = ct.bid(1)`, `kv_split_id = ct.bid(2)`
  (`src/tilegym/suites/flashinfer/cutile/fmha_decode_bsr.py`).
- Each CTA loads the thin query tile `[BLOCK_H, BLOCK_D]` once, then streams KV in `[BLOCK_N, BLOCK_D]`
  chunks with the same online-softmax recurrence as prefill (running max, running sum, `exp2` with
  `flush_to_zero=True`).
- Work is O(S_kv x D) MACs per head against O(S_kv x D) bytes of mandatory KV reads — arithmetic intensity is
  roughly `BLOCK_H` MACs per KV element, far below the compute/memory ridge on any current part. The kernel is
  memory-bound by construction; the right success metric is achieved KV-read bandwidth, not TFLOPS.
- **Split-KV**: the KV range is divided into `NUM_KV_SPLITS` slices, each CTA emits a partial `(acc, lse)`,
  and a second reduction merges them (`_splitk_reduce_kernel` in `fmha_decode_bsr.py`;
  `src/tilegym/ops/cutile/splitk_reduce.py` for the ops-level decode). Host fast paths skip the kernel for 1–2
  splits (`_splitk_reduce_with_seq_len`, `fmha_decode_bsr.py`).
- **Paged gather**: per KV block the page ids come from `block_tables`; multi-page tiles use a page-level
  gather (`_load_page` / `_load_page_mla` in `fmha_decode_bsr.py`), single-page tiles share one resolved
  `page_id` between the K and V loads.

## What dominates performance

- **KV-cache bandwidth.** Every cached byte must be read once per step; nothing amortizes it. Load path
  quality — TMA on paged loads with a pipelining hint (`allow_tma=True, latency=2`; B200, flashinfer suite)
  and page-level gather instead of token-level gather (NUM_PAGES vs BLOCK_N transactions;
  B200, fmha_decode_bsr) — moves the kernel directly.
- **Thin MMA tiles.** The natural score matmul has M = `BLOCK_H` (16/32 for GQA groups), far below the
  efficient MMA M extent. The shipped fix is `TRANS_QK`: swap the MMA operands so
  `qk = ct.mma(k_tile, q^T)` runs with M = `BLOCK_N` = 128 and the small extent lands on N
  (in-file comment at the `TRANS_QK` accumulator setup, `fmha_decode_bsr.py`; the GQA kernel already carried
  the flag, and it was later extended to MLA decode). The softmax reduction axis flips accordingly and the accumulator is kept transposed
  (`[BLOCK_D, BLOCK_H]`) until the epilogue.
- **`num_ctas` on thin tiles.** 2-CTA MMA pairs pay their pairing overhead per instruction; with an
  accumulator N extent of 16 there is not enough work to amortize it — measured 1.76x *slower* than
  `num_ctas=1` on B200 MLA decode. Note that the
  Triton-TileIR twin of the same operator uses `num_ctas=2` profitably — the knob does not transfer because the
  cuTile tile is thinner. Decode autotune spaces do not sweep `num_ctas`; the only shipped `num_ctas=2` use is
  a host-side hint on the MLA path gated on a wide tile (`num_batch >= 16 and BLOCK_H >= 64`,
  `fmha_decode_bsr.py`) — consistent with the thin-tile analysis — and pre-sm90 cannot use `num_ctas=2` at all.
- **Grid fill.** `batch x kv_heads` CTAs rarely covers the SM count at serving batch sizes; split-KV is the
  lever that restores parallelism, at the price of the merge pass and LSE traffic. `KV_LEN_PER_SPLIT`
  interacts with page size and `BLOCK_N`.
- **Occupancy and block shape.** Decode occupancy is autotuned jointly with `(BLOCK_H, BLOCK_N)` via
  `exhaustive_search` (occupancy [1,2]; config generator `_get_gqa_decode_autotune_configs`
  in `fmha_decode_bsr.py` walks `BLOCK_H` in [8,16,32,64] divisors of the query group size).
- **Launch/host overhead.** Steps are microseconds-long, so per-launch host work (split planning, reduce
  dispatch, dummy allocations) is visible; keep host paths allocation-free and the 1–2-split fast paths in
  place.

## Applicable techniques

- **tech-copy-batching** — page-level gather TMA over the paged cache (token batching only at page
  granularity); split-KV merge fast paths, no per-step allocations.
- **tech-tma-load** + **tech-latency-hint** — `allow_tma=True, latency=2` on Q and paged KV loads.
- **tech-occupancy** — occupancy x block-shape search per query-group size; spaces exclude
  `num_ctas=2` (see the thin-tile note above) and prune per arch.
- **tech-softmax-max-elision** — the online-softmax max/rescale chain is per-KV-tile overhead here
  too; when a range proof bounds the scores, the elision rungs apply.
- **tech-num-ctas** — compare actual MMA tile shapes before copying the knob across backends (the
  thin-tile measurement above).
- Operand-order/thin-MMA note — `TRANS_QK` swap belongs with tile-size/MMA-shape reasoning, not with launch
  knobs: fix the tile shape first, then tune.

## Where it lives

- `src/tilegym/ops/cutile/flash_decode.py` — `fmha_decode` grouped decode; core loop factored out
  as `attention_decode_kernel_grouped_impl` for reuse; split-K reduce in
  `src/tilegym/ops/cutile/splitk_reduce.py`.
- `src/tilegym/suites/flashinfer/cutile/fmha_decode_bsr.py` — serving decode:
  `flashinfer.attention.decode_attention_kv_paged` (GQA) and `flashinfer.attention.decode_mla_kv_paged`
  (MLA), `TRANS_QK`, split-KV reduce kernel, page-gather helpers.
- `src/tilegym/ops/cutile/mla_decoding.py` (`mla_decoding`) and
  `src/tilegym/ops/cutile/mla_decoding_split_kv.py` (`mla_decoding_split_kv`) — absorbed-MLA decode, dense and
  split-KV forms.
- `src/tilegym/ops/cutile/gemma_attention_decode.py` (`gemma_attention_decode`),
  `src/tilegym/ops/cutile/attention_sink_decode.py` (`attention_sink_decode`) — variant heads on the same
  loop.
- Tests/benchmarks: `tests/ops/test_flash_decode.py`, `tests/ops/test_mla_decoding.py`,
  `tests/ops/test_mla_decoding_split_kv.py`, `tests/benchmark/bench_mla_decoding.py`.
