---
id: cutile-language
kind: language
title: cuTile (Python)
summary: The cuTile working model and the rules that bite, organized by what you touch — execution, tiles, memory, numerics, traced Python, hints and the wrapper; the official docs are the language reference.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# cuTile (Python)

## Overview

The language reference is the official documentation: <https://docs.nvidia.com/cuda/cutile-python>
— execution/data/memory model, every op's semantics and valid arguments, performance tuning, known
issues. This page holds only what the docs cannot: the working model and the failure knowledge for
someone writing or porting kernels. It is organized by what you touch, in the order you touch it;
each section is a short model plus the rules that bite there, each fact stated once.

cuTile (`import cuda.tile as ct` — never `import cutile`) is NVIDIA's tile-based Python DSL, at the
same abstraction level as Triton with two load-bearing differences: addressing is **tile-space**
(block indices against N-D arrays that carry their own shape and strides — no pointer arithmetic,
no stride kernel args), and bulk memory movement is **descriptor-driven** (the compiler builds TMA
transfers from `ct.load`/`ct.store`). TileGym's cuTile kernels live in `src/tilegym/ops/cutile/`
and `src/tilegym/suites/*/cutile/`.

## Execution and launch

A launch is a grid of kernel instances; `ct.bid(axis)` tells an instance which tile it owns. Inside
a kernel there are no thread ids, shared-memory declarations, or barriers — the compiler owns all
of that. A kernel is one logical program over tile values; the only global-memory boundary is
`ct.load`/`ct.store`/`ct.gather`/`ct.scatter`, and everything between is dataflow the compiler may
reschedule.

- The grid need not equal the work-item count: the persistent idiom is
  `for i in range(ct.bid(0), n_items, ct.num_blocks(0))` with the grid sized from
  `NUM_SM * occupancy` (`tech-persistent-grid`).
- **The kernel launch is the only inter-CTA synchronization point.** There is no fence primitive,
  so within one launch a CTA cannot correctly wait on another — chained blocks and spin locks have
  no sound spelling and fail nondeterministically (hangs, stale reads). Split sequential
  dependencies into phases at a launch boundary; use `ct.atomic_add` for order-independent
  cross-CTA accumulation.
- Fold big dims into grid axis 0 or go persistent (axes 1–2 have low hardware caps — see the docs);
  `ct.launch` takes no `None` arguments (pass a dummy tensor plus a flag); don't pass
  `requires_grad=True` tensors. Launches are stream-ordered: later work on the same stream needs no
  `torch.cuda.synchronize()`, but host-side reads and timing still require one, and handing results
  to a different stream needs explicit synchronization (an event or stream wait).

## Tiles and indexing

`index` in `ct.load`/`ct.store` names **which tile**, not an element offset:
`ct.load(X, index=(bid,), shape=(BLOCK,))` reads elements `[bid*BLOCK, (bid+1)*BLOCK)`.

- Writing `index=(bid * BLOCK,)` is the single most common conversion bug — silently wrong tiles
  when it happens to compile.
- `index`/`shape` rank must equal the array rank: load the full-rank tile and `reshape` down;
  `reshape` back up before `ct.store`.
- Every tile dim is a compile-time power of two. Pass both `N` and `N_padded` to the kernel, mask
  reductions over the pad, and divide by `N`, not `N_padded`.
- **There is no element-granular in-tile shift, rotate, or shuffle** — warp-shuffle and
  smem-offset idioms from CUDA/Triton have no direct spelling; ports restructure. Shifts come from
  ADDRESSING — load k offset views of the source (out-of-range regions zero-pad, which is also the
  free `same`-padding halo idiom; keep shifted views 16-byte-aligned) — or from precomputed layout
  (`tech-specialization-into-data`). Whole-row shifts are just
  index changes.
- `order='F'` is not a transpose — it permutes access order, not the tile; use
  `ct.transpose`/`ct.permute`.

## Memory

Two paths, chosen per call site. The primitive you write fixes the path — the compiler never
converts one into the other:

- **`ct.load`/`ct.store`** — block-aligned bulk copies; the compiler decides TMA use (`allow_tma=`)
  and builds descriptors from array metadata. Loads stage through shared memory: amortized when the
  tile is re-read, pure overhead when it is not.
- **`ct.gather`/`ct.scatter`** — software-addressed element access straight to registers, with
  `mask=`/`padding_value=`/`check_bounds=` semantics.

The decision rule is **reuse decides**, and the naive heuristics are wrong in both directions:

- Tile re-read (GEMM K-loop, attention K/V, conv weights, multi-pass row statistics) → `ct.load`.
- Single-pass streaming (elementwise ops, fused one-pass row reductions) → `ct.gather` often wins;
  `ct.load` pays smem staging for a tile consumed once and materializes the full padded tile.
- Truly random access (hash tables, genuine sparsity) → gather/scatter is the only option.
- **Runtime indices do not force gather**: paged/indirect access is `ct.gather(table, ...).item()`
  then `ct.load(cache, index=(page_id, ...), allow_tma=True)` — TMA descriptors accept runtime tile
  indices; gathering everything instead measured 78x slower on a paged decode KV path.
  Ragged-but-contiguous segments use `Array.slice(axis, start, stop)` + `ct.load`.
- When you do gather, keep the array N-D and pass a tuple of index tiles; flattening to
  hand-computed offsets hides the strided structure and degrades to per-lane addressing.

Boundary semantics differ between reads and writes:

- OOB `ct.load` is defined only under a `padding_mode` (the default is UNDETERMINED); `ct.gather`
  is bounds-checked with `padding_value=0` unless `check_bounds=False`.
- OOB `ct.store` clips silently — a kernel can write a fraction of its output with no error.
- Hand-computed offsets that stay in-bounds but land in the wrong row alias silently: mask
  explicitly, and redirect masked scatter lanes to an out-of-bounds sentinel.
- Zero padded lanes with `ct.where(mask, x, zeros)`, never `x * mask` (`NaN * 0 == NaN`).

## Numerics

- `ct.mma(a, b, acc)` preserves the accumulator dtype — accumulate in fp32; operands must share a
  dtype.
- **No fp32→tf32 promotion**: fp32 mma operands run on CUDA cores unless you `ct.astype` both to
  `ct.tfloat32` yourself (with the precision consequences that implies).
- Reductions accumulate in the tile dtype: `ct.sum` on a bf16 tile accumulates IN bf16 — dead for
  statistics; `astype(ct.float32)` before reducing, cast the result back.
- Python float kernel arguments truncate to f32 before the kernel sees them; bake precise constants
  as `ct.Constant[float]` or pre-round host-side and account for it in tolerances.
- The compiler fuses `a * b + c` into FMA; there is no `ct.fma`.

## Kernel Python is a traced subset

Kernel code is staged, not interpreted; the tracer accepts a subset of Python:

- No `break`/`continue`; no Python slice syntax on tiles (`tile[:, 0]` fails — use `ct.extract` or
  a scalar load); define every variable on all branches.
- Helpers are plain single-exit `def`: no lambdas, no multi-return helpers inside loops.
- No tuple/list iteration or subscripts (unroll by hand; hoist lookups to scalars); no `**kwargs`;
  `None` checks belong in the wrapper.
- No growing a tile across loop paths — pre-size the tile and write by offset.
- Bool tiles are masks, not numbers: combine with `&`/`|`, select with `ct.where`, cast before
  arithmetic.
- Use Python operators for index math — `ct.add`/`ct.mul` can promote int32 to float.
- Build masks and constant tiles once, outside loops — there is no reliable loop-invariant hoisting
  of tile creation.

## Hints, specialization, and the wrapper

Hints (`occupancy`, `num_ctas`, `num_worker_warps`, per-op `latency`, `opt_level`) are compile-time
contracts on the kernel object (`@ct.kernel(...)`, `kernel.replace_hints(...)`, per-architecture
`ct.ByTarget`). Valid ranges are in the docs; what each hint trades is on its technique page
(`tech-occupancy`, `tech-num-ctas`, `tech-latency-hint`). There is deliberately no
`num_warps`/`num_stages` — warp layout and pipeline depth are the compiler's job, with a
`num_worker_warps` heuristic when unhinted.

- **Each distinct hint set is a distinct compiled kernel**: build hint variants once at module
  scope or behind a tune-once cache; `replace_hints` on the hot path recompiles every launch.
- Kernels specialize per SHAPE-CONFIG bucket, plus per `ct.Constant` value
  (`ct.static_eval`/`ct.static_iter`/`ct.static_assert` for compile-time metaprogramming) — never
  per tensor value or identity. A wrapper cache keyed finer than shape (`id(tensor)`,
  `data_ptr()`, value hashes) recompiles on every call under any caller that re-allocates tensors;
  move per-value work into data passed as a normal input.
- Autotune trial runs re-execute the kernel: in-place kernels must search on a split buffer.
- Every kernel parameter is compiled in — unused parameters cost registers.
- `tensor.view(-1)` on a non-contiguous (e.g. NHWC) input silently reorders data before the kernel
  ever runs.
