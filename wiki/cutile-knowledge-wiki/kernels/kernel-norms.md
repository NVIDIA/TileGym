---
id: kernel-norms
kind: kernel
title: Normalization family (layer_norm, rms_norm, strided variants)
summary: Per-slice statistics plus broadcast normalize-and-affine; single-pass fused stats, thin fp32 stat stores, and gather/scatter-vs-tiled-load for strided (NCHW-style) reduction dims dominate.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Normalization family (layer_norm, rms_norm, strided variants)

## What it computes

All members compute statistics over a normalized slice of the input, then apply a broadcast
normalize-and-affine map:

```
rms_norm:     y = x * rsqrt(mean(x^2) + eps) * (offset + w)        # offset 0.0 Llama, 1.0 Gemma3
layer_norm:   y = (x - mean(x)) * rsqrt(var(x) + eps) * (w + shift) + b
```

Layout variants:

- **Contiguous last-dim** — normalize over the innermost dim of a `(M, N)` view: `rms_norm`
  (`src/tilegym/ops/cutile/rms_norm.py`), `layer_norm_legacy` and `persistent_layer_norm`
  (`src/tilegym/ops/cutile/layer_norm_legacy.py`).
- **Backward** — forward passes save thin fp32 `mean`/`rstd` tensors; rms_norm has a persistent
  cuTile backward with fused weight-grad accumulation (`cutile/rms_norm.py`); the cuTile
  layer_norm variants raise `NotImplementedError` on backward.

## Computational shape

Reduce-then-broadcast over independent slices: each slice needs its statistics before any of its
outputs can be produced, so the minimal schedule is (load slice) → (reduce) → (rescale the same
elements) → (store). The op is memory-bound; arithmetic per element is a handful of FMAs plus one
`rsqrt` per slice. Everything interesting is about how many times slice bytes cross the memory
system and how the strided variant assembles its tiles:

- Contiguous case: a row is one tile (or a short chain of tiles); the reduction is a single
  `ct.sum` over the tile axis.
- Strided case: a `(BLOCK_SIZE_C, BLOCK_SIZE_W)` 2D tile covers a chunk of the reduction dim × a
  chunk of the trailing dim; reducing axis 0 yields per-`w` statistics vectorized across `W` lanes.
  The `W` axis supplies the memory coalescing that the strided `C` axis cannot.
- Weight gradient (backward) is a *column* reduction across all rows — orthogonal to the row-parallel
  `dx` — so it decomposes into per-block partials plus a second-stage reduction.

## What dominates performance

- **Single-pass fused stats vs reload.** The naive schedule reads the slice once per statistic
  (mean pass, variance pass, normalize pass — `_layer_norm_fwd_kernel` in
  `cutile/layer_norm_legacy.py` is exactly this three-pass shape). The fused pattern loads
  the tile once, computes all statistics from the live registers, and normalizes the already-loaded
  values: rms_norm needs only the sum of squares, and layer_norm gets mean and variance in the same
  pass via `E[x²] − E[x]²`, as in `_persistent_layer_norm_fwd_kernel` — see `reference/layer-norm-legacy-tiled-load.md`.

  Cutting passes is the dominant lever for the contiguous variants. Fusing MULTIPLE statistics into
  one reduction is also safe to do aggressively: a tuple-valued `ct.reduce` (sum and sum-of-squares in
  one pass, `func=add_pair`) measured at parity to 24% FASTER than two separate `ct.sum` calls in a
  standalone A/B at two tile shapes — earlier folklore that tuple reduce lowers slowly did not
  reproduce. [2026-08, B200, cuda-tile 1.2.0, N=1 repro]
- **Thin stat stores.** `mean`/`rstd` are `(M,)` fp32 vectors — a few bytes per row-block next to a
  row-tile store. These stores are issued with `allow_tma=False` (the two `ct.store(Mean, ...)` /
  `ct.store(Rstd, ...)` lines in the snippet `reference/layer-norm-legacy-tiled-load.md`; same in `cutile/rms_norm.py`): a TMA descriptor
  transaction per tiny store is pure
  overhead. The main output store on the persistent rms_norm kernel also opts out of TMA and carries
  a latency hint, with the measured gains recorded as comments at the store site — see `reference/rms-norm-store.md`.

  Treat those numbers as priors to re-measure, scoped to that kernel.
- **Tile shape by row width.** Rows-per-tile shrinks as the row widens: the static-persistent
  rms_norm picks `TILE_SIZE_M` from `TILE_SIZE_N = next_power_of_2(N)` (16 rows when
  `TILE_SIZE_N` ≤ 1024, 4 default, 2 when `TILE_SIZE_N` ≥ 16384), and the mode heuristic switches
  between one-CTA-per-row and persistent scheduling at `M > NUM_SMS * 2` (both in
  `cutile/rms_norm.py`). The legacy persistent layer_norm exposes the same knob as an autotuned
  `BLOCK_N` (`cutile/layer_norm_legacy.py`).
- **Gather/scatter vs tiled load on the strided variant.** With the reduction dim at stride `W`, the
  gather-route implementation flattens the tensor and computes explicit flat offsets
  (`col * STRIDE_C + w * STRIDE_W + row * STRIDE_N`) for `ct.gather`/`ct.scatter`. Two hazards come
  with that route. First, out-of-range `C` offsets still land *inside* the buffer (in the next
  row's data), so gather's bounds padding never fires and an explicit `ct.where` mask is mandatory.

  Second, masked-off scatter lanes must be redirected to an out-of-bounds sentinel index rather than
  left pointing at live data.

  The structural alternative is a tiled 3D load
  treating the tensor as `(N, C, W)` and letting the tile machinery handle the stride — denser
  transfers and no index algebra, available only when the layout is expressible as a real tensor
  view. The gather route is the general fallback; expect it to trail a contiguous-equivalent kernel
  because the strided axis defeats coalescing.
- **Split reduction for weight grads.** rms_norm backward accumulates each block's `dw` contribution
  into a `(grid, TILE_N)` fp32 partial buffer inside the persistent loop, and the host finishes with
  `dwp.sum(0)` (`cutile/rms_norm.py`) — avoiding both atomics and an `M×N`
  temporary.
- **Zero-padding as a correctness lever for reductions.** With `TILE_SIZE_N = next_power_of_2(N)`,
  out-of-range lanes must contribute zero to sums of squares; `padding_mode=ZERO` on loads is what
  guarantees that. The rms_norm forward load carries the rationale in-line (plus a scoped latency
  hint) — see `reference/rms-norm-load.md`.

  Uninitialized padding lanes inflate the variance silently — wrong results, not a crash.

## Applicable techniques

- **Fused one-pass statistics** — sum-of-squares (rms) or `E[x²]−E[x]²` (layer_norm) computed from a
  single tile load; the first thing to check on any norm-shaped kernel.
- **Persistent grid-stride scheduling with rows-per-tile tiering** (`tech-persistent-grid`) — pair
  `TILE_SIZE_M` with row width; switch to one-CTA-per-row below the persistent break-even.
- **TMA opt-out for thin stores** (`tech-tma-store-disable`) — `allow_tma=False` on per-row statistic
  stores and other small-stride writes.
- **Load/store latency hints** (`tech-latency-hint`) — per-instruction scheduling hints on the dominant
  tile load and store (the `latency=10` load and `latency=3` store shown above, `cutile/rms_norm.py`).
- **Explicit masking + sentinel-redirect scatter** — required whenever offsets are hand-computed
  over a strided layout; gather bounds checks do not protect against in-buffer wrong-row hits.
- **Split two-stage reduction** — per-block partials plus host (or second-kernel) combine for
  cross-row gradients.
- **Zero padding-mode on reduction inputs** — makes non-power-of-two widths safe without per-lane
  predication.

## Where it lives

- `rms_norm`: dispatch `src/tilegym/ops/ops.py`; cuTile `src/tilegym/ops/cutile/rms_norm.py`
  (three forward modes + persistent backward); Triton `src/tilegym/ops/triton/rms_norm.py`;
  tests `tests/ops/test_rms_norm.py`.
- `layer_norm_legacy` / `persistent_layer_norm` (contiguous last-dim): dispatch
  `src/tilegym/ops/ops.py`; cuTile `src/tilegym/ops/cutile/layer_norm_legacy.py`; tests
  `tests/ops/test_layer_norm_legacy.py`.
