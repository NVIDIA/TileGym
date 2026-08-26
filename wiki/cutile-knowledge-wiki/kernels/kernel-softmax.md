---
id: kernel-softmax
kind: kernel
title: Row softmax
summary: Batch of independent row reductions plus elementwise normalization; passes-over-the-row, occupancy-vs-row-width, and host routing between kernel variants dominate.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Row softmax

## What it computes

Row-wise softmax over a 2D tensor `(n_rows, n_cols)`:

```
forward:   y[i, :] = exp(x[i, :] - max(x[i, :])) / sum(exp(x[i, :] - max(x[i, :])))
```

The max subtraction is mandatory for numerical stability on unbounded inputs (attention-style
kernels with a range proof on the scores can elide it — `tech-softmax-max-elision`). Accumulation is fp32 regardless of I/O
dtype — every kernel converts on load, and padded lanes arrive as `-inf` so they drop out of both
statistics. The gather-persistent kernel shows both conventions plus the grid-stride row loop — see `reference/softmax-persistent-row-loop.md`.

Forward needs two
row statistics with a dependency between them (max, then sum of shifted exponentials).

## Computational shape

A batch of independent per-row pipelines: reduce → broadcast → elementwise map → store. Rows never
interact, so the grid parallelizes over rows and all interesting structure is *within* a row. The
two-statistic dependency chain forces one of these shapes:

1. **Full-row residency** — the whole row in one tile; both reductions are register-local, one read
   and one write per element. Only viable while the tile fits the register/SMEM budget.
2. **Multi-pass** — re-read the row once per statistic: max pass, sum pass, normalize pass (three
   reads, one write). The chunked kernel (`cutile/softmax.py`) is this shape.

The op is memory-bound at the wide end (traffic = passes × row bytes) and launch/occupancy-bound at
the narrow end; in between, fp32 `exp` throughput and the reduction dependency can become visible.

## What dominates performance

- **Number of full passes over the row.** The minimum is read+write for forward; every extra
  statistic pass adds a full row read. Choosing the pass structure by row width is the
  single largest lever, worth more than any tuning knob once rows exceed register residency.
- **Occupancy tiering by row width.** Wider rows mean bigger tiles, more registers/SMEM per CTA,
  fewer resident CTAs. The cuTile kernels pin occupancy per variant. The narrow gather-persistent
  kernel uses occupancy 4; the full-row TMA path searches a small set on first use and caches the
  winner per device, dtype, and shape. The host uses the selected value for both the compiler hint
  and persistent-grid multiplier (`min(NUM_SM * occupancy, n_rows)`; all in `cutile/softmax.py`).
  Register and shared-memory pressure generally favor lower occupancy as row width grows, but the
  winner can be non-monotonic because grid-wave and compiler effects also matter. Autotuning adapts
  those effects to the installed compiler instead of encoding version-specific shape tiers.
- **Routing between kernel variants.** The repo keeps several forward kernels alive simultaneously —
  gather-persistent, one-CTA-per-row register-cached, full-row TMA persistent, chunked
  three-pass — because each wins in a row-width band, and the host routes among them
  (`_Softmax.forward`, `cutile/softmax.py`). Width thresholds in the host code are
  shape/arch-sensitive tuning state — re-measure them, treat only the
  *existence* of the bands as durable. Routing also handles capability: `use_tma=True` on a
  pre-sm90 device warns and falls back to the non-TMA path.
- **Tile rounding and padding.** `TILE_SIZE = next_power_of_2(n_cols)` means a non-power-of-two row
  wastes up to half its lanes; padded lanes load as `-inf` (`padding_mode NEG_INF` /
  `padding_value=-math.inf`, as in the gather snippet, `reference/softmax-persistent-row-loop.md`) so `max` ignores them and
  `exp(-inf) = 0` drops out of the sum. Bounds checks compile out entirely when the row is an exact
  power of two — the check flag is a compile-time expression, removing per-lane predication from the
  hot path — see `reference/softmax-bounds-check.md`.
- **Grid regime: persistent vs one-CTA-per-row.** Grid-stride persistent scheduling amortizes launch
  and setup when `n_rows >> num_SMs`; a one-block-per-row grid ("multi-wave") is simpler and can win
  when rows are few or the loop overhead matters (`_softmax_kernel_multi_wave_full_row_reg_cached_ldg`
  in `cutile/softmax.py`).

## Applicable techniques
- **Persistent grid-stride scheduling** (`tech-persistent-grid`) — decouples grid size from `n_rows`;
  pair the grid multiple with the kernel's selected occupancy.
- **Occupancy autotuning** (`tech-occupancy`) — search a bounded set and cache the winner instead of
  accepting the compiler default or encoding compiler-version-specific shape tiers.
- **TMA vs gather loads** (`tech-tma-load`) — full-row `ct.load` with padding mode vs `ct.gather` with
  index tiles; both paths exist per variant and the winner is width-dependent.
- **Compile-time bounds-check elision** — route exact power-of-two widths to a check-free
  instantiation (`TILE_SIZE != N` as a constexpr condition).
- **Host-side variant routing** — a dispatch table over (row width, pow2-ness, TMA capability) is
  part of the kernel's design surface, not an afterthought.

## Where it lives

- Dispatch: `softmax` in `src/tilegym/ops/ops.py`.
- cuTile implementation (kernel variants + routing):
  `src/tilegym/ops/cutile/softmax.py`.
- Tests: `tests/ops/test_softmax.py` (`Test_Softmax`; TMA / chunked / multi-wave parametrized).
