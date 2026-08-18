---
id: tech-occupancy
kind: technique
basis: ungraded-batch-1
title: Occupancy hint
summary: The occupancy hint (1-32 expected active CTAs per SM) trades per-CTA register budget against latency hiding; the optimum is tile-config- and shape-dependent, so autotune it via hints_fn instead of hardcoding.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Occupancy hint

## What it is

The `occupancy` hint tells the cuTile compiler how many thread blocks the programmer expects to run
concurrently per SM (integer 1-32). It is a compiler *budgeting* hint: higher values
shrink the per-CTA register allowance (forcing tighter codegen, eventually spills) in exchange for more
resident CTAs to hide memory latency. It also feeds grid sizing for persistent kernels
(`num_programs = min(NUM_SM * occupancy, n_items)`).

## Pattern

```python
# Static hint (simple cases / per-arch defaults)
@ct.kernel(occupancy=4)
def kernel(X, Y, BLOCK: ct.constexpr): ...

# Autotuned (production): do NOT hardcode occupancy in @ct.kernel — pass via hints_fn
result = ct.tune.exhaustive_search(
    configs, stream=stream,
    grid_fn=lambda cfg: (min(NUM_SM * cfg.occupancy, n_items), 1, 1),
    kernel=kernel, args_fn=...,
    hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
)
# Cache the tuned kernel (kernel.replace_hints(...)) — rebuilding hints every call recompiles.
```

## When to use

- Worth tuning on most kernels; starting ranges (seeds for the sweep, not answers): 1-4
  compute-bound, 4-8 balanced (GEMM/TMA), 8-16 memory-bound reductions (softmax, layernorm), 16-32
  very light copies/casts.
- Whenever a kernel spills registers at a large tile size: sweeping occupancy *down* (or `num_worker_warps`
  up) is often the fix — the two knobs partition the same register file.
- Persistent kernels: the hint and the grid cap must move together, since the launch multiplies `NUM_SM` by
  the same value — see `tech-persistent-grid`.

## Caveats

- **Register-pressure coupling**: raising occupancy shrinks the per-CTA register budget; on wide-tile kernels
  the spill boundary arrives quickly, so document and sweep only the spill-safe range (see the softmax
  evidence below).
- **Tile-config-dependent optimum**: the same kernel's different tile paths can want different occupancy —
  one path improves while another regresses at the same value (see the linear-attention evidence below). Tune
  occupancy jointly with tile size, or per tile config at launch via `replace_hints`.
- `num_worker_warps` is a sibling knob (warps per CTA) that interacts with occupancy on the same register
  file; large-BLOCK kernels have fixed spills with `nww=8` where occupancy alone could not.
- Calling `replace_hints` on every invocation recompiles; cache the tuned kernel object keyed by everything
  that changes the optimum (shape, dtype, arch).
- The hint is an expectation, not a command — the compiler may not achieve it; verify with ncu
  (`sm__warps_active`) when the win matters. One verified silent-no-op mode: when a CTA's SMEM
  footprint exceeds roughly half an SM (~114 KB on B200), the hint produces a byte-identical binary —
  the cap is tile-height-scoped (hints take normally on TM=1 tiles). Check that the hint took before
  crediting or blaming it.
- **Occupancy optima do not survive body edits.** Measured optima moved after a loop restructure
  (6 -> 4), a chassis change (0 -> 8), and a store join (planned 4 lost to 1) — re-sweep after any
  structural change instead of inheriting the previous winner; sweep odd values too (7 has beaten 8
  on non-pow2 grids).
- **occupancy=1 can be a silent rate collapse, not a safe floor**: a (512,256,64) f32-accumulator
  GEMM at occupancy=1 measured ~50x slower (969 -> 19 TF/s) with correct values and no diagnostic
  (standalone repro) — never leave occ=1 unraced on mma bodies.

## Evidence

- Silent occ=1 collapse: standalone repro, (512,256,64) f32-acc GEMM — 19 TF/s at occupancy=1 vs 969 TF/s control, values correct, no diagnostic. [2026-08, B200, cuda-tile 1.2.0, N=1 repro]
- SMEM-cap silent no-op (byte-identical binary above ~half-SM footprint; TM=1 tiles unaffected) and post-edit optimum moves (three independent kernels) — previous measurements. [2026-07, B200, cuda-tile 1.2.0, N=2+3 kernels]
- B200, liger softmax/rms_norm/fused_add_rms_norm: occupancy x num_worker_warps autotuned on first call, cached by (path, n_cols); documented spill-safe space for the exp2+APPROX softmax fwd: occ in [2,3,4], `nww=4` only — nww=8 or occ>=5 spills at large N. [2026-07]
- liger rms_norm bwd: `num_worker_warps=8` selected when BLOCK_SIZE>=8192, fixing a register spill; previously measured -49% at c=16384. [2026-07, B200]
- Other in-repo instances (B200/sm_100): RoPE occupancy autotuned over (1,7,9,12); grpo_loss per-batch static selection; liger dyt bwd nww=8; flashinfer decode/prefill spaces; rope_quantize_fp8 TOKENS_PER_BLOCK x occupancy. [2026-07]
