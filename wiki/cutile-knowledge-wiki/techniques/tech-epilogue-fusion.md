---
id: tech-epilogue-fusion
kind: technique
basis: ungraded-batch-1
title: Epilogue fusion
summary: Apply the elementwise tail (bias, activation, residual/elementwise combine, dropout, quant/dequant scales) to the producing kernel's accumulator tile instead of launching separate elementwise kernels — the intermediate never round-trips through HBM.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Epilogue fusion

## What it is

Folding the elementwise operations that consume a kernel's output — bias add, activation, elementwise
combine with a second tensor, residual add, dropout, quantization/dequantization scaling, and small trailing
reductions — into the kernel that produces the output, applied to the accumulator tile while it is still on-chip
(registers, or TMEM on Blackwell — cuTile leaves residency to the compiler). Unfused, every stage is its
own launch plus a full write and re-read of the intermediate
through HBM; fused, the tail runs between the last `ct.mma` (or reduction step) and the single `ct.store`,
at near-zero marginal memory cost.

This page covers the elementwise-tail case. Fusing whole sibling kernels (dual gate/up GEMMs, multi-stage
MoE pipelines) is the same idea at coarser granularity. Hand-scheduled CUDA epilogues additionally
overlap the epilogue with the next tile's MMA; in cuTile the win to reach for is the fusion itself —
eliminated launches and round trips — with instruction overlap left to the compiler.

## Pattern

**GEMM + fused tail**: after the K loop, every epilogue stage operates on the accumulator tile,
guarded by compile-time `ct.Constant` flags so unused stages compile out.

Variants of the same move:

- **Fuse adjacent elementwise ops even without a GEMM producer**: `silu_and_mul` computes
  `silu(x[..., :H]) * x[..., H:]` in one kernel — one load pair, one store — instead of an activation
  kernel feeding a multiply kernel — see `reference/silu-and-mul-gather.md`.
- **Subtile a register-heavy tail**: emit the epilogue in column slices so only a slice of the fp32
  intermediate is live at once, instead of shrinking the whole GEMM tile — 32-wide activation slices,
  or the accumulator split into two N/2 halves to cut epilogue shared memory.
- **Chunk a fusion whose intermediate cannot be materialized**: fused_linear_cross_entropy runs per-chunk
  GEMM → loss → grad accumulation so the full logits tensor never exists.
- **Fuse a framework-op tail into one kernel**: a partials reduction that ran as eager `torch.sum(0)`
  becomes a single fused reduce kernel that also folds in the fp32-to-output-dtype cast.

## When to use

- The kernel's output feeds one or more elementwise launches over the same tensor. Each unfused stage costs
  one launch plus one extra full read+write of the tensor; both disappear when the tail moves onto the
  on-chip tile.
- Both ends of the shape spectrum qualify, for different reasons: launch-bound small shapes win by dropping
  launches (one upstream scale on d_logits replacing per-gradient elementwise-multiply launches);
  memory-bound large shapes win by dropping HBM round trips.
- The intermediate is too large to materialize at all (logits in fused linear + cross-entropy): fusion plus
  chunking is the only route that bounds peak memory.
- A trailing reduction or cast runs as eager framework ops (`torch.sum(0)`, `.to(dtype)`) — fusing them
  into one custom kernel beats the eager sequence.

## Caveats

- **The tuning optimum moves.** The epilogue's extra live tiles (a second accumulator for GLU gates, RNG
  offsets and mask for dropout) raise register pressure, so the plain-GEMM best config is no longer the
  winner — re-tune per shape rather than inheriting the producer's configs (see `tech-tile-size`).
- **Pointwise work is 10-100x cheaper on the accumulator than on mma operands** (measured across two
  independent board families) — operand transforms belong in a producer kernel or folded into the
  weights; only accumulator-side work rides free in the epilogue.
- **Do not shrink the whole tile for the tail's sake**: when the fused tail is register-heavy, subtile the
  epilogue in column slices and keep the GEMM tile.
- **Fusion can trade away backward inputs.** A fused forward that skips saving the pre-activation is only
  legal in inference; gate fused-vs-unfused dispatch on `requires_grad`, never default to the fused path.
  Training variants must store intermediates (pre-activation, dropout mask) from inside the
  kernel, turning one (M, N) store into two or
  three — the traffic advantage shrinks in training.
- **Compile-time epilogue flags multiply specializations.** Each combination of the compile-time flags
  is a separate compilation — the flags are `ct.Constant` kernel parameters.

  Combined with a GEMM-sized autotune space this compounds compile time — keep the shipped
  flag-times-config product small.
- **Optional arguments breed host overhead.** Fused signatures accumulate optional tensors (bias, saved
  activation, second input, mask); filling them with fresh `torch.zeros(1)` placeholders per call puts
  allocations and fill launches on the hot path — cache dummies per (device, dtype) instead
  (`tech-copy-batching`).
- **Mask/dtype friction**: boolean masks must be stored as int8 because TMA descriptors do not support bool.

## Evidence

- Prologue-vs-epilogue asymmetry: identical elementwise ops measured 10-100x costlier applied to mma operands than applied to the accumulator, across two independent board families (an elementwise census band and a computed-operand GEMM family). Previous measurements. [2026-07, B200, cuda-tile 1.2.0, N=2 board families]
- fused_linear_cross_entropy (liger suite, cuTile): chunked backward-in-forward — per-chunk GEMM → CE with in-place dlogits → grad accumulation — keeps peak logit memory `O(chunk_size x V)` once BT x V x sizeof(dtype) crosses ~4 GB. [2026-07, B200]
- Other in-repo instances (B200): multi_token_attention (fused softmax-bwd + causal mask, ~45% claimed), liger FLCE/JSD (deferred-grad path + 256 MB chunked fallback). [2026-07]
