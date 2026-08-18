---
id: tech-transcendental-replica
kind: technique
basis: measured, N=3 kernels
title: Transcendental replicas in tile ops
summary: Replace sin/cos/exp-class device transcendentals with explicit range-reduction + polynomial forms built from tile arithmetic (Cody-Waite two-word reduction, pi-parity, exp2 rebase, magic-add rounding) — for bit-accuracy control and speed on transcendental-hot paths.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Transcendental replicas in tile ops

## What it is

Instead of calling the built-in transcendental (`ct.sin`, `ct.cos`, `ct.exp`), reimplement the function
as explicit tile arithmetic: reduce the argument to a small primary range, evaluate a short polynomial
there, and reconstruct the result from the range-reduction bookkeeping. The classical scalar-math
toolkit, spelled in tile ops:

- **Cody-Waite range reduction**: subtract `k * C` where the constant `C` (e.g. pi/2) is split into two
  (or three) floats whose sum is exact to double-ish precision — `r = x - k*C_hi - k*C_lo` keeps the
  reduced argument accurate even for large `x`, entirely in f32 tiles.
- **Magic-add rounding**: `k = (x * (1/C) + 1.5*2^23) - 1.5*2^23` extracts round-to-nearest-integer
  without an int conversion — one FMA and one subtract, tile-friendly.
- **Pi-parity reconstruction**: for sin/cos, the quadrant index `k mod 4` selects sign and sin/cos swap;
  when only parity matters (rotations by `n*pi + r`), a cheap `k mod 2` sign flip suffices.
- **exp2 rebase**: `exp(x) = exp2(x * log2(e))` — fold the `log2(e)` into an upstream scale (e.g. the
  softmax `1/sqrt(d)`), then use the hardware `exp2` path (see `tech-ftz-approx` for the flag side).

## Pattern

```python
# sin(pi * t) with parity reconstruction, f32 tiles; C = pi split hi/lo
k  = (t + MAGIC) - MAGIC                      # round-to-nearest integer part
r  = (t - k) * PI_HI + (t - k) * PI_LO        # Cody-Waite reduced argument
r2 = r * r
p  = r * (S1 + r2 * (S3 + r2 * S5))           # short odd polynomial
# parity of k flips the sign: sin(pi*t) = (-1)^k * sin(pi*(t-k))
sign = 1.0 - 2.0 * ((k + MAGIC2) - MAGIC2_PARITY_TRICK)
out = p * sign
```

The polynomial degree sets the accuracy; degree 5-7 in the reduced range reproduces f32 sin/cos to
1-2 ulp. Coefficients are compile-time constants — precompute them host-side, never derive in-kernel.

## When to use

- The hot path evaluates sin/cos/exp of a position- or index-derived phase per element (rotary
  embeddings, positional encodings, frequency kernels) and the built-in is either slow on the target
  or numerically mismatched against the reference you must be bit-close to.
- You need to control WHERE the rounding happens to match a torch/cuDNN reference's error profile —
  a replica gives you the exact operation order; a built-in gives you whatever the library does.
- Arguments can be large (positions x frequencies), where naive `x - round(x/C)*C` in f32 loses the
  reduced argument entirely — Cody-Waite is the fix, not higher polynomial degree.

## Caveats

- This is the opposite trade of `tech-ftz-approx`: replicas buy accuracy control and per-op cost on a
  transcendental-HOT path; approx flags buy speed where accuracy is already slack. Read both.
- Keep every coefficient a compile-time constant. Runtime-scalar coefficients drag the whole chain
  into runtime-value paths and per-element extract costs.
- Verify against an fp64 oracle over the REAL argument distribution (including the largest positions),
  not unit-range samples; range-reduction bugs only fire at large arguments.
- A replica is more instructions than one SFU op; on paths that are memory-bound anyway it changes
  nothing — profile first.

## Evidence

- Two rotation-heavy kernels whose hot path is sin/cos of a position-derived phase: pi-parity +
  Cody-Waite + magic-add replicas matched the framework reference's numerics where the built-in path
  missed tolerance, and carried both kernels to their final shipped forms; the same replica library transferred between the two boards unchanged. Previous
  measurement. [2026-07, B200, cuda-tile 1.2.0, N=2 kernels]
- Attention-class softmax: exp2 rebase with log2(e) folded into the upstream scale (plus flush-to-zero)
  is the standard fast-softmax spelling; adopted as the winning form on a flash-attention board.
  Previous measurement. [2026-07, B200, cuda-tile 1.2.0, N=1 kernel]
