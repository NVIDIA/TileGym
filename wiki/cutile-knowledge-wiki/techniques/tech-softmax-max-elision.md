---
id: tech-softmax-max-elision
kind: technique
basis: measured, N=6 kernels (+1 boundary kernel)
title: Softmax max-elision under a range proof
summary: The online-softmax running max exists only to prevent exp overflow — when the score range is provably bounded, drop the max/rescale machinery entirely (exp2 directly on scaled scores); when headroom is tight but the tail is benign, a static first-tile max shift is the cheap middle rung; heavy-tailed inputs are the hard boundary where the full online max is mandatory.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Softmax max-elision under a range proof

## What it is

Textbook online softmax carries a running row max `m_i`, and every KV tile pays the update chain:
new max, `alpha = exp2(m_i - m_ij)` rescale of the running sum AND the accumulator, then the shifted
exp. That machinery exists for exactly one reason — keeping `exp` arguments in range. It is not part
of the math; softmax is shift-invariant. If you can PROVE the scores are bounded (softmax scale folded
upstream, normalized Q/K, bounded logit magnitudes), the entire chain deletes:

- **Rung 1 — no max at all**: `p = exp2(qk * S_LOG2E)` directly; accumulate `l += sum(p)`; divide once
  at the end. Deletes the per-tile max reduce, the alpha rescale multiplies on the accumulator, and the
  dependency chain they serialize.
- **Rung 2 — static m0 shift**: when fp16/bf16 intermediate headroom is tight but the input is not
  adversarial, use the FIRST KV tile's row max as a fixed shift for all tiles. One reduce total,
  no per-tile rescale, and it buys ~the full exponent range back.
- **Rung 3 — full online max**: mandatory when the score distribution is heavy-tailed (hot keys,
  un-normalized projections) — see Caveats.

## Pattern

```python
# scale and log2(e) folded into Q upstream (one multiply, or folded into the QKV projection weights)
qk = ct.mma(q_tile, k_tile, qk_zero)          # scores already in exp2 units
p  = ct.exp2(qk)                              # rung 1: no max, no rescale
l_i = l_i + ct.sum(p, axis=-1, keepdims=True) # running denominator only
acc = ct.mma(p.astype(ct.bfloat16), v_tile, acc)
# epilogue: out = acc / l_i
```

The range proof is a certification step on REAL inputs, not an assumption: compute
`max |qk * scale|` over the target workload's actual data at wrapper level once, and compare against
the exp2 overflow bound of the P dtype (f32: ~127; fp16 P: ~15 — which is why fp16 P usually needs
rung 2).

## When to use

- Attention or softmax kernels where the score scale is known and inputs pass the range census —
  normalized-QK attention variants are the canonical case.
- Any softmax whose inputs are outputs of a bounded op (post-norm, post-tanh, quantized logits).
- fp16 P tiles with tight exponent headroom: rung 2 (m0 shift) instead of abandoning the elision — an
  fp16 P overflow presents as sudden Inf/NaN rows on exactly the workloads with the largest scores.

## Caveats

- **The range proof is the technique.** Eliding the max on unproven inputs converts a perf trick into a
  silent-overflow bug: exp2 of an unbounded score saturates to Inf, and Inf/Inf produces NaN rows that a
  loose tolerance check can miss. Certify on the real data distribution, and re-certify when the
  upstream model changes.
- **Heavy-tailed inputs are a hard boundary, not a tuning knob.** On one attention board with hot-key
  score outliers, every elision rung failed correctness and the full online max was the only correct
  form — the census must look at tails (max), not moments (mean/std).
- The division by `l_i` moves to the epilogue; on very long rows check the running `l` against the
  accumulation dtype's range (f32 running sums are safe in practice; fp16 running sums are not).
- Pairs with `tech-ftz-approx` (exp2 flush-to-zero) and the exp2-rebase entry in
  `tech-transcendental-replica`; the scale·log2e fold into projection WEIGHTS (zero per-call cost) is an
  instance of composing constants into an upstream operator.

## Evidence
