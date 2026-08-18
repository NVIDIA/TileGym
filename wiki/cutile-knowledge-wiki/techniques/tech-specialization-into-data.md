---
id: tech-specialization-into-data
kind: technique
basis: measured, N=3 kernels
title: Specialization into data
summary: Move per-case behavior (boundaries, masks, variant math) out of kernel branches and into precomputed data variants selected by index — the kernel body stays branch-free, and cases that wanted an if or a second kernel become rows of a small table.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Specialization into data

## What it is

When a kernel needs different behavior per region or per case — boundary tiles vs interior tiles,
masked vs unmasked rows, variant coefficients — the instinct is a runtime `if`, a mask multiply, or a
separate specialized kernel launch. All three have real costs in cuTile (branch tracing costs and
wrong-value hazards, extra launches pay launch + cold-L2). The alternative: precompute one small data
table whose rows ARE the specializations, index it by the case id, and run one branch-free kernel body.

The case analysis moves to host-side (or a tiny prep kernel) where it is cheap and testable; the device
kernel sees only "load my variant's row and compute".

## Pattern

For a stencil/conv kernel with `same` padding, the boundary tiles differ from interior only in which
taps are dropped. Dropping a tap is a LINEAR operation on the weights: for each of the 9 edge cases
(corner/edge/interior, `v = vy*3 + vx`), fold the dropped taps into a per-variant weight-and-bias set:

```python
# host or prep kernel: build 9 composed (W_v, b_v) variants, one per boundary class
# device kernel: every tile does the identical thing
v = boundary_class(tile_row, tile_col)          # 0..8, from compile-time grid math
w = ct.load(Wvar, index=(v, ...))               # this tile's variant weights
acc = stencil(x_tile, w) + ct.load(Bvar, index=(v, ...))
```

Same shape for masks (bake the mask into additive coefficients: -inf rows instead of a `ct.where` per
step) and for variant math (per-band scale tables instead of per-band code paths).

## When to use

- Border/boundary handling that would otherwise be a second "ring" kernel — a ring launch is nearly
  pure launch + cold-L2 overhead when the ring is thin.
- Any runtime `if` whose branches differ only in COEFFICIENTS, not structure. cuTile traces both
  sides, and branch-heavy bodies compile superlinearly slower.
- Case analysis derivable from compile-time or grid quantities (tile position, band index), so the
  variant index needs no data-dependent scalar extraction.

## Caveats

- Precompute variants on DEVICE in a one-time prep kernel or on host at wrapper level — but key
  nothing on tensor identity or values (the per-shape-bucket specialization contract,
  `cutile-language`); the variant table is an input like any other, rebuilt whenever its source
  weights change.
- The table must stay small (cases x weight size); this technique is for FEW discrete cases, not
  per-element variation (that is what masks are for).
- Derive the folded variants algebraically and validate against an fp64 direct implementation of each
  case; a wrong fold is a silent boundary bug, exactly the class of error correctness suites
  under-sample.
- After a constraint changes (e.g. weights become runtime instead of compile-time), RE-DERIVE the
  architecture choice — a variant scheme priced out under one constraint can become the winner under
  another. This exact re-derivation was missed once at real cost.

## Evidence
- Attention board: causal masking hardened into unconditional additive form (mask folded into data, no
  per-step branch) as the shipped form. Previous measurement. [2026-07, B200,
  cuda-tile 1.2.0, N=1 kernel]
