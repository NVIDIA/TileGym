---
id: tech-latency-hint
kind: technique
basis: ungraded-batch-1
title: Latency hints on loads, stores, and gathers
summary: latency=N (1-10) on ct.load/store/gather/scatter tells the compiler expected DRAM traffic so it can schedule prefetch; the integer is itself a tuned knob and deeper pipelining is not monotonically better.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Latency hints on loads, stores, and gathers

## What it is

`latency=N` on `ct.load` / `ct.store` / `ct.gather` / `ct.scatter` hints the compiler about expected memory
traffic intensity (int 1 low to 10 high, default None = compiler-inferred), enabling it to schedule the
memory op earlier and software-pipeline the consuming loop.
Deeper hints keep more state in flight: the price is register pressure.

## Pattern

```python
# Main input tensor loads: high traffic
ct.load(X, index=(bid, 0), shape=(M, N), latency=10)

# Stores: moderate
ct.store(Y, index=(bid, 0), tile=y, latency=3)

# Gather/scatter with few elements: low
ct.gather(x, (row, offs), latency=1)

# Sweep strategy: try {1, 2, 3, 6, 10} on the hottest loads; benchmark each
```

## When to use

- Memory-latency-bound kernels whose hot path issues one load/gather per work item: single-chunk row kernels
  (norms, softmax), decode-attention KV loads, persistent row loops where the hint pipelines the *next*
  iteration's load.
- Attention-style loops: common starting points are latency=6 for K/V loads and latency=10 for main
  input tensors; in-repo attention kernels use K=2/V=4/Q=2 (see Evidence) — the value is per-kernel.
- Prefer `latency=` hints over hand-written double-buffer prefetch; manual (n+1)-buffering has produced
  register-spill cliffs where the hint achieved the same pipelining within budget.
- As an autotune axis when one static value does not dominate (see the matmul LOAD_LATENCY evidence).

## Caveats

- **Straight-line paths only**: inside a multi-chunk gather loop the compiler schedules gathers from several
  iterations early to honor the budget, inflating register pressure into spill cliffs. If an op has
  single-chunk and multi-chunk dispatch paths, apply the hint only in the single-chunk variant.
- **Deeper is not better**: measured sweeps have picked latency=1 over 2 and 3 on register-heavy backward
  kernels (see Evidence) — treat the integer as a tuned knob, not a dial to max out.
- **But shallow is dangerous on mma-fed mainloop loads — hint depth must match the load's role.** On a
  plain bf16 GEMM where deep mainloop hints (latency=8) gained only ~1% over no hints, the SAME loads at
  latency=2 lost 15%: a shallow hint overrides the compiler's own deeper schedule and strangles
  pipelining. The role-matched scheme (deep 8-10 on mainloop/mma-fed loads, shallow 1-4 on epilogue
  loads, latency=1 on stores) is safe on both sides; the risk is mismatch, not use. When unsure on a
  mainloop load, omit the hint rather than guessing low. `latency=1` on stores measured free-to-slightly-
  positive on the same body.
- The kernel's register footprint must leave room for pipelining to engage at all; splitting a monolithic
  kernel into two sweeps has been the enabling move (see Evidence).
- The hint needs enough work to hide the extra in-flight state — small rows (n_cols<4096) with low
  compute-to-memory ratio have regressed under the same hint that wins on larger rows.
- A winning hinted gather can *be* the pipeline: replacing it with a TMA `ct.load` has removed the pipeline
  and regressed — do not "upgrade" a hinted gather to TMA without an A/B.

## Evidence
- cuTile matmul: `ct.load` cost exposed as autotune knob LOAD_LATENCY (1..10, -1 = compiler-inferred); only sm90 tunes it, all other arches keep -1. [2026-07]
- Other in-repo instances (B200/sm_100): liger FNA K=2/V=4/Q=2 with loads kept inside the `ct.mma` loop; flashinfer paged loads latency=2; liger persistent row loops latency=3; grpo_loss gather latency=10. [2026-07]
- Prior measured magnitudes: fused_add_rms_norm bwd single-chunk kernel +29% (n_cols=4096) and +45% (n_cols=8192) from `latency=3` on the row-loop gathers; rms_norm bwd +9% regression at n_cols=2048 from the same hint; MLA decode found `latency=2` optimal with `latency=3` neutral-to-regressing. [2026-07, B200]
- Role-mismatch asymmetry, standalone A/B on a bias-GEMM (M=N=K=4096 bf16, all three hint roles present, all variants value-correct): no hints 123.5us; role-matched scheme (mainloop 8 / epilogue 2 / store 1) 122.3us (+1.0%); INVERTED scheme (mainloop 2 / epilogue 8 / store 1) 145.3us (-15%); store latency=1 alone 122.9us. Previous measurement. [2026-08, B200, cuda-tile 1.2.0, N=1 repro]
