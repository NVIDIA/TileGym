---
id: tech-ftz-approx
kind: technique
basis: ungraded-batch-1
title: Flush-to-zero and approximate math
summary: Per-call-site flush_to_zero and rounding_mode=APPROX (plus the CUTILE_ENABLE_FTZ / CUTILE_ENABLE_APPROX env toggles) trade denormal handling and last-ULP accuracy for faster math in exp/div/tanh-heavy kernels.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Flush-to-zero and approximate math

## What it is

cuTile math ops accept numeric-mode arguments that relax IEEE behavior for speed:

- `flush_to_zero=True` — skip denormal (subnormal) handling: denormal inputs/outputs are treated as zero.
  Supported on `ct.exp2`, `ct.rsqrt`, `ct.truediv`, `ct.sqrt`, `ct.add`, `ct.sub`, `ct.mul`.
  `ct.exp()` does NOT accept it — use `ct.exp2` instead.
- `rounding_mode=ct.RoundingMode.APPROX` — use the hardware approximation (MUFU-class instructions) instead
  of the IEEE-rounded software sequence. Supported on `ct.truediv` and `ct.tanh`.

Both exist as environment toggles too — `CUTILE_ENABLE_FTZ=1` and `CUTILE_ENABLE_APPROX=1`
(`CLAUDE.md`, TileGym debug env table) — which flip the behavior globally rather than per call site.

This is a numerics tradeoff, not a free win: the payment is in ULPs and denormal semantics, and whether that
is acceptable is a property of the op's tolerance contract, not of the kernel.

## Pattern

The shipped attention kernel is the canonical per-call-site form. The softmax exponent and
online-softmax rescale in the inner loop — see `reference/attention-exp2-ftz.md`.

and the final normalize after the loop — see `reference/attention-truediv-approx.md`.

Approximate tanh in an inner loop, with the cost/accuracy note kept at the call site:

```python
# APPROX tanh: ~1.6x faster, 2-4 ULP off; well within bwd tolerance 1e-2.
tanh_x = ct.tanh(alpha * x, rounding_mode=RMd.APPROX)
```

Prefer the per-call-site parameters over the env toggles in shipped kernels: call sites keep the tradeoff
reviewable, testable, and scoped to the ops that tolerate it. The env toggles are for whole-kernel A/B triage
("would FTZ/APPROX matter here at all?") before editing.

## When to use

- Math-heavy inner loops — softmax/attention exponentials, normalization reciprocals and rsqrt, tanh-based
  activations and soft caps — where profiles show MUFU/SFU or FP pipe pressure rather than memory.
- The op's accuracy contract has slack: activation/attention paths validated at tolerances of 1e-2..1e-3
  absorb 2-4 ULP; strict-parity ops (loss values feeding convergence checks, fp64 reference tests) do not.
- Inputs are known to be far from the denormal range (post-softmax probabilities, normalized activations), so
  FTZ changes nothing numerically on real data and only removes the hardware's denormal slow path.

## Caveats

- Whole-kernel impact is typically single-digit percent on math-heavy kernels; measured per-op
  effects can be larger when a single transcendental dominates the loop (the ~1.6x tanh claim
  is per-instruction, not per-kernel).
- May fail tight test tolerances. Loosening atol/rtol to make a test pass is only legitimate after confirming
  the precision loss is acceptable for the op's users.
- FTZ zeroes gradients that pass through denormal magnitudes; training ops that legitimately traffic in tiny
  values (loss scaling edge cases) can silently lose signal.
- Support is op- and dtype-specific: `ct.exp()` rejects `flush_to_zero`; fp16 lacks FTZ on at least the
  `ct.mul` path. Check the API before sprinkling flags.
- The env toggles change numerics globally for every kernel in the process — results measured under
  `CUTILE_ENABLE_FTZ=1`/`CUTILE_ENABLE_APPROX=1` do not certify per-call-site edits, and vice versa.
- Keep the accuracy note at the call site: an APPROX flag without its ULP/tolerance
  justification is unreviewable.
- Related but distinct: rewriting math onto hardware-approximated instructions (sigmoid as
  `0.5 + 0.5*tanh(0.5x)` → MUFU.TANH) changes the *formula*, not just the rounding mode — treat
  it as an algorithmic change with its own accuracy audit.

## Evidence

- liger dyt bwd (B200): `ct.tanh(..., rounding_mode=APPROX)` with in-file note "~1.6x faster, 2-4 ULP off; well within bwd tolerance 1e-2". [2026-07]
- gemma_attention soft cap: `ct.tanh(..., rounding_mode=APPROX)` and APPROX truediv in the attention inner loop.
- ops/cutile attention family ships `flush_to_zero=True` on softmax exp2/rescale and `flush_to_zero=True, rounding_mode=APPROX` on the final truediv across attention, varlen, sink, and decode variants — see `reference/attention-exp2-ftz.md` and `reference/attention-truediv-approx.md`. [2026-07, B200]
