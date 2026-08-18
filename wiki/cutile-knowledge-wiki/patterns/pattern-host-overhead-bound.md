---
id: pattern-host-overhead-bound
kind: pattern
title: Host-overhead-bound op (launch/alloc-dominated small kernels)
summary: End-to-end op time is far above summed device kernel time — launches, allocations, zero-init fills, Python prep, and GPU-to-CPU syncs dominate; delete host work and batch launches instead of tuning the kernels.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Host-overhead-bound op (launch/alloc-dominated small kernels)

## Symptom

The op's wall-clock time is much larger than the sum of its device kernel times. A profile (nsys timeline or
`torch.profiler`) shows gaps between kernels, many tiny launches per op call, fill kernels you did not write
(zero-inits), or a stream stall at a synchronization point. Op time stays flat as the shape shrinks — it has
hit the launch floor — and per-kernel tuning moves nothing. Classic on decode/small-batch shapes.

## Likely causes

Each cause is confirmed the same way: count and name the kernels per op call
(`torch.profiler` `key_averages()`), and compare device time to wall time.

1. **Zero-init allocations the kernel fully overwrites.** `torch.zeros`/`zeros_like` launches a fill kernel;
   `torch.empty`/`empty_like` does not. Confirm: FillFunctor-style kernels in the profile next to buffers every
   element of which is written by your kernel (rms_norm dW_partial).
2. **Fresh placeholder/dummy tensors allocated per call.** Read-only dummies rebuilt every forward each cost an
   alloc + fill launch (linear-attention dense path: 6 tiny launches, ~7 us pure overhead per forward,
   previously measured).
3. **One small copy launch per segment/input.** The op loops on the host, launching a near-identical tiny copy
   kernel per item (the motivating case for fixed-slot copy batching).
4. **GPU-to-CPU sync on the launch path.** `.item()`/`.tolist()` to compute the grid stalls the stream and
   blocks CUDA-graph capture. Confirm: cudaStreamSynchronize in the nsys trace before the launch
   (grouped_gemm `m_sizes.tolist()`, since removed).
5. **Python-loop host prep and materialized temporaries.** Index/tile mappings built element-by-element in
   Python, or out-of-place ops where an in-place op would do.

## Candidate techniques

Ordered by expected value — a reference list, not the full candidate space; explorations beyond it
are encouraged:

1. `tech-copy-batching` — the whole family: fixed-slot batched copy kernels,
   `empty_like`-vs-`zeros`, cached placeholder tensors, dead-alloc elimination, in-place ops, vectorized host
   prep, and sync removal. This is the primary route; most of the causes above map one-to-one to its moves.
2. `tech-persistent-grid` — a persistent kernel replaces per-item launches with one
   launch and can eliminate host-side tile mapping plus its GPU-to-CPU sync entirely.
3. **CUDA-graph awareness** — capture-and-replay caps per-launch CPU cost, and benchmarking under CUDA Graph
   is the cleanest way to separate launch overhead from device time (grouped_gemm added a CUDAGraph benchmark
   alongside the persistent rewrite). Graph capture requires a sync-free, allocation-stable launch
   path — which is exactly what techniques 1 and 2 produce — so treat graph-compatibility as a design
   constraint even when you do not ship graphs.

## Caveats

- Overhead wins are per-shape-regime: microseconds saved dominate a launch-bound shape and are noise on large
  shapes — measure the full matrix and gate accordingly.
- `empty` instead of `zeros` is correct only when every element is provably written; document the invariant at
  the alloc site (`tech-copy-batching` caveats).
- Cached placeholder tensors must be read-only and keyed by (device, dtype), or one caller corrupts the next.
- Wall-clock benches carry a launch floor that *hides* small device-time regressions while you chase overhead;
  run a device-time check at final acceptance.
- Batched copies can lose store vectorization and give back the launch savings — inspect SASS store widths
  after batching.
