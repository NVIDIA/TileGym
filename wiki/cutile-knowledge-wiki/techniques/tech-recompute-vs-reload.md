---
id: tech-recompute-vs-reload
kind: technique
basis: measured, N=2 kernels + 2 negative instances
title: Recompute vs reload on backward kernels
summary: Backward kernels that load stored forward intermediates should race a recompute variant — arithmetic is cheaper than a second input stream, and a rounded (bf16/fp16) forward chain is idempotent, so recompute can be BIT-exact — but first verify on real data that the stored tensor equals the recomputable expression at all; on two boards it did not.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Recompute vs reload on backward kernels

## What it is

A backward kernel's signature typically includes forward intermediates the framework saved for it
(gate activations, normalized values, generated basis tensors). Loading them is a whole extra input
stream; recomputing them from inputs already in registers is a handful of FMAs. On memory-bound
backward bodies the recompute deletes a stream and usually wins.

The numerics are better than intuition suggests: if the forward stored a ROUNDED value (bf16/fp16),
recomputing the same chain with the same ops and rounding points reproduces it bit-for-bit —
rounding is deterministic, so `round(f(x))` recomputed equals `round(f(x))` stored. Recompute is not
an approximation here; it can be exact (pair with `tech-transcendental-replica`-style care about WHERE
the rounding points sit if the chain is nontrivial).

## Pattern

```python
# signature: bwd(grad_out, x, gate_saved, ...)   <- gate_saved is a whole input stream
# forward computed: gate = silu(x @ Wg)          <- x and Wg are already loaded here
g = silu_tile(ct.mma(x_tile, wg_tile, zero))     # recompute: FMAs replace a load stream
# ... use g exactly where gate_saved would have been loaded
```

Certification step, once per op at wrapper level, on REAL workload data:

```python
assert (recompute_expr(inputs) == stored_tensor).all()   # bit compare, not tolerance
```

## When to use

- Any backward kernel whose input list includes a tensor derivable from other inputs it already
  loads — gates, activations, norms of loaded values, generated bases (trig tables, decay vectors).
- The body is memory-bound (most backwards are): deleting a stream is worth dozens of FMAs per
  element. On compute-bound bodies, the reload keeps the FMA pipe free — race both.
- Generated tensors (positional bases, decay tables): recomputing in-kernel also removes the
  generator kernel from the critical path — the fastest load is the one that never happens.

## Caveats

- **The certification is mandatory, not paranoia: "derivable-looking" is not derivable.** On two
  boards the saved tensor and the recomputable expression were RELATED but independently drawn
  (a benchmark generator producing both from separate random draws; a stored tensor carrying an extra
  epsilon the formula lacked) — recompute was silently wrong. The bit-compare on real data is the
  entire safety of this technique; run it before any perf work.
- Re-run the certification when the upstream model or data pipeline changes — the equality is a
  property of the data contract, not of your kernel.
- The recompute must replicate the forward's ROUNDING points to be bit-exact: same dtype casts in the
  same places. A recompute in f32 of a chain the forward rounded to bf16 mid-way matches only to
  tolerance, which may still pass your gate — decide which you are claiming.
- Register pressure: the recompute's operands must already be live (or cheap to load); if it drags in
  a new operand stream, it is a reload wearing a costume.

## Evidence
- Negative instances (why the certification is the technique): two boards' "recomputable" tensors
  failed the bit-compare — one pair was independently generated data, one stored tensor embedded an
  epsilon absent from the naive formula. Recompute was rejected there at census time, before it could
  ship a wrong kernel. Previous measurements. [2026-07, B200, cuda-tile 1.2.0,
  N=2 kernels]
