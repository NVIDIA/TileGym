---
id: tech-tma-store-disable
kind: technique
basis: ungraded-batch-1
title: Disable TMA on stores (allow_tma=False)
summary: Try allow_tma=False on output stores — the TMA store path has overhead for some access patterns, and a plain store can win; the sign flips per kernel, so always A/B.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Disable TMA on stores (allow_tma=False)

## What it is

`ct.store(..., allow_tma=False)` disables the TMA path for that store, falling back to a direct store. The TMA
store path carries fixed overhead (descriptor + transaction setup) that a small or latency-sensitive
store may not amortize. It is a one-parameter,
zero-risk-to-correctness change whose only cost is the benchmark run needed to keep or drop it.

## Pattern

```python
# Before (default): TMA store
ct.store(Y, index=(bid, 0), tile=result)

# Candidate: direct store — keep whichever measures faster
ct.store(Y, index=(bid, 0), tile=result, allow_tma=False)
```

## When to use

- The store tile is small relative to the surrounding work: scalar stats (mean/rstd), thin row stores, or a
  store that follows a reduction which already serialized the epilogue.
- Profiling shows store latency on the critical path of a memory-bound kernel.
- As a cheap sweep item after the load path is settled: benchmark both settings and keep the faster
  one.

## Caveats

- **Does not always help** — the often-quoted +10-30% range is anchored on one op (rms_norm); the
  same knob has been applied and then reversed on the same kernel after re-measurement (see
  Evidence).
- Effects are shape-heterogeneous: when a full shape matrix shows the knob winning on some shapes and losing
  on others, gate it on the measured shapes rather than setting a global default.
- `allow_tma=False` changes only the store path. Pre-sm90 there is no TMA hardware: a store with
  `allow_tma=True` runs a `cp.async` emulation with ~8-15% overhead, so on Ampere this knob is
  about avoiding the emulation, not tuning real TMA.
- The load-side twin is a different decision with different evidence — see `tech-tma-load`.

## Evidence

- rms_norm fwd: measured +30% from `allow_tma=False` on the output store. [2026-07, B200]
- B200, unsloth layernorm: the store knob flipped three times on the same kernels — `allow_tma=False` stores removed (re-enabling TMA), re-added on scalar mean/rstd and row stores alongside a compile-time masking guard, then removed again (current state: zero `allow_tma=False` stores in the file). The sequence documents that the knob's sign is per-kernel and per-measurement, not copyable. [2026-07]
