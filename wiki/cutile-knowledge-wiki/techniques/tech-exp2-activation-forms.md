---
id: tech-exp2-activation-forms
kind: technique
basis: measured, N=8 kernels
title: Division-free exp2 spellings of activations
summary: Sigmoid/SiLU/GELU-tanh/softplus all reduce to exp2 plus adds and multiplies — fp32 divides serialize the dependent chain while SFU exp2 issues async-cheap, so respell the activation around exp2 and remove or restructure the divide; race the approx-divide flag first, adopt exact forms only when tolerance demands them.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Division-free exp2 spellings of activations

## What it is

Activation-heavy elementwise and epilogue bodies bottleneck on two scalar-unit costs: `exp` (fine —
rebase to `exp2`, whose SFU path is cheap and overlaps) and fp32 DIVISION, which lowers to a long
dependent sequence that serializes the chain. The activations of practice are all one exp2 and one
reciprocal away from closed forms, and the reciprocal can be removed, weakened, or restructured:

- **Sigmoid**: `sigma(z) = 1 / (1 + exp2(z * -LOG2E))`. Division-weakening options, in order:
  approx-divide flag (`tech-ftz-approx`), reciprocal-multiply replication when matching a reference,
  or the rsqrt-of-square respelling
  `sigma = rsqrt(u*u)` with `u = 1 + exp2(-z*LOG2E)` — two rsqrt-class ops replacing the full divide,
  exact to ~1-2 ulp.
- **SiLU / gated MLP**: keep the product form `x * sigma(z)` and share ONE sigmoid across gate uses;
  in `p * sigma` epilogues the sigmoid rides an existing multiply slot.
- **GELU (tanh approximation)**: `x * sigma(1.702*x)`-class forms restructure as
  `x - x / (exp2(2*a*x*LOG2E) + 1)` — one exp2, one divide, and the divide's numerator is `x` (no
  extra dependency); measured cheaper than the tanh-chain spelling on epilogue bodies.
- **Softplus**: the stable form `softplus(z) = max(z, 0) + log1p(exp(-|z|))`, with the `log1p` on the
  bounded argument replaced by a short minimax polynomial when no `log1p` primitive exists — bounded
  domain makes a low-degree fit exact to fp32.

## Pattern

```python
# gated-MLP epilogue: silu(x) * y with one exp2 and one weakened divide
u = 1.0 + ct.exp2(x * NEG_LOG2E)      # NEG_LOG2E = -1.4426950408889634, compile-time
s = ct.rsqrt(u * u)                   # sigma(x) without fp32 divide
out = x * s * y
```

All constants (`LOG2E`, polynomial coefficients) are compile-time; fold upstream scales into them
where the algebra allows.

## When to use

- Elementwise or epilogue bodies where the profiler shows the activation chain pacing the kernel —
  typical on gated MLPs, discretization steps, and activation-dense pointwise fusions.
- The op sits under an SFU-friendly load path: exp2 overlaps memory on gather/LDG-lowered bodies;
  the win shrinks when the body is purely bandwidth-bound (the chain hides under loads either way).

## Caveats

- **Race the cheap rung first**: on most tolerance budgets the approx-divide/FTZ flag alone matches
  these respellings at zero code cost — the exact algebraic forms earned their keep only on the
  tightest-tolerance boards. Try flags, then respell (this ordering was confirmed against reference
  solutions that shipped the flag where the campaign shipped algebra).
- Bit-compatibility with a torch reference is a different problem than being fast: torch's own
  sigmoid/gelu lower through specific rounding chains; if the gate is bit-strict, replicate the
  reference chain (`tech-transcendental-replica` covers controlling where the rounding happens).
- Stability edges live at large |z|: the naive `log(1+e^z)` overflows where the max/log1p form does
  not; the minimax fit is only valid on the bounded post-reduction domain — state and assert the
  domain.
- fp16/bf16 intermediate rounding inside the respelled chain shifts results by 1-2 ulp vs the
  reference spelling — fine for tolerance gates, visible to bit gates.

## Evidence

- Division-free exp2 respellings (sigmoid family, GELU-tanh restructure, p-times-sigma epilogues,
  stable softplus with minimax log1p) shipped as the winning activation forms across eight
  elementwise/epilogue boards in one campaign; the fp32-divide serialization was the bisected
  mechanism on the clearest board (chain time tracked divide count, not op count). Previous
  measurements. [2026-07, B200, cuda-tile 1.2.0, N=8 kernels]
- Ordering caveat receipt: reference solutions on overlapping boards shipped the APPROX-divide flag
  instead of algebra and matched tolerance — the exact forms are the tight-tolerance rung, not the
  default. [2026-08, B200, construct comparison]
