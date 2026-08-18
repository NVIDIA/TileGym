---
id: tech-num-ctas
kind: technique
basis: ungraded-batch-1
title: num_ctas (CGA width and 2-CTA MMA)
summary: num_ctas sets CTAs per cooperative grid array; num_ctas=2 enables 2-CTA MMA on Blackwell dense-dot kernels with wide accumulators and large tiles — unsupported pre-sm90; measured 1.76x slower on one thin-tile (N=16) MMA decode kernel.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# num_ctas (CGA width and 2-CTA MMA)

## What it is

`num_ctas` sets how many CTAs are clustered into one cooperative grid array (CGA) per launch unit (power of 2,
1-16; API default None = auto). On Blackwell (sm100+), `num_ctas=2` enables 2-CTA MMA on dense dot workloads:
paired CTAs cooperate on the MMA, amortizing operand traffic across the cluster. The pairing has
fixed overhead, so it pays only when each MMA instruction carries enough work.

## Pattern

```python
# Static (simple cases)
@ct.kernel(num_ctas=2)
def kernel(...): ...

# Per-architecture
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, default=1))
def kernel(...): ...

# As an autotune axis (production): pair multi-CTA only with large tiles
hints_fn=lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}
```

## When to use

- sm100+ matmul-family kernels (matmul, bmm, dense attention GEMMs) with wide MMA accumulators — N extent 64
  and up, typically 256x256+ output tiles; the in-repo sm100+ matmul space pairs `num_ctas` 2/4 only with
  256x256-and-larger tiles at occupancy 1 — see `reference/sm100_matmul_num_ctas_gating.md` for the config generator.

- As one axis of a per-arch autotune space, not a standalone default: keep `num_ctas=1` pre-sm90 (CGA
  unsupported; the pre-sm90 branch of the same generator pins it) and on sm120/121 (the in-repo branches
  sweep occupancy instead).
- Check the MMA accumulator shape, not the operator name: the knob is about per-instruction work, and the same
  op's cuTile and Triton kernels can have different tile shapes.

## Caveats

- **Thin-tile MMA**: on a kernel whose `ct.mma` accumulator is thin in N (at or below 16), num_ctas=2 pays
  its pairing overhead without amortization — measured 1.76x slower than num_ctas=1 on B200 MLA decode.
- **Does not transfer across backends**: the Triton-TileIR implementation of the same MLA decode op successfully
  uses num_ctas=2; the cuTile kernel's thinner tile made the same knob catastrophic — compare actual MMA tile
  shapes before copying launch knobs.
- Not universal even on wide-tile Blackwell kernels — at least one B200 matmul-family kernel measured best at
  num_ctas=1 (see Evidence); keep it an autotune axis.
- Hardware constraints prune the space silently: configs with num_ctas=2 on sm80 are invalid, so per-arch
  config generators must branch on compute capability rather than filtering at runtime.

## Evidence

- In-repo per-arch pairings: sm80 forbids num_ctas=2 outright (in-file comment: CGA unsupported pre-SM90); pre-sm90 defaults pin num_ctas=1; sm90 matmul/mla wins pair num_ctas=2 with 256-wide tiles while sm90 attention stayed single-CTA; sm100+ pairs 2/4 with 256x256+ tiles. [2026-07]
- B200, unsloth cross_entropy: search space extended to occupancy [1,2,4,8,16,32] x `num_ctas` [1,2] for a memory-bound large-vocab kernel. [2026-07]
- B200, MLA decode (thin-tile counterexample): `num_ctas=2` measured 1.76x slower than `num_ctas=1` on the `TILE_H=16` fp16 MMA path (previous measurement). [2026-07]
