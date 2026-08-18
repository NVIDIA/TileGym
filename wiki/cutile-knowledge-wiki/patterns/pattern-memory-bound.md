---
id: pattern-memory-bound
kind: pattern
title: Memory-bound kernel (bandwidth-limited rows/reductions/elementwise)
summary: Kernel time tracks bytes moved, not FLOPs — compute utilization is low while memory throughput is high; fix the memory path first (TMA, tile size, prefetch), then change the decomposition if bytes themselves must shrink.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Memory-bound kernel (bandwidth-limited rows/reductions/elementwise)

## Symptom

Kernel time scales with bytes transferred, not with arithmetic. Nsight Compute classifies the kernel
"Memory Bound": memory throughput is high (>~80% of peak) while compute (SM) utilization is low (<~50%).
Typical of elementwise ops, row-wise norms/softmax, and reductions — arithmetic intensity below ~10
FLOPs/byte.

The score that matters is **achieved bandwidth**: `bytes_moved / kernel_time` versus the board's
peak. A kernel near peak is finished; a kernel far
below peak on regular data has a memory-path problem.

## Likely causes

1. **Software gather/scatter on regular access.** The kernel uses `ct.gather`/`ct.scatter` (per-element
   software addressing) where access is actually contiguous or block-aligned. Confirm: read the kernel — if
   indices are `bid * BLOCK + arange`, a slice of a dense tensor, or a paged block with a scalar page id, the
   access is TMA-expressible.
2. **Tile size mismatched to transaction granularity.** Tiles too small waste transaction width and multiply
   launch/loop overhead; tiles too large push registers and reduce concurrent CTAs. Confirm: sweep 2-4
   plausible tile sizes and watch achieved bandwidth move; check ncu registers/thread at the wide end.
3. **No load/compute overlap.** Loads are not prefetched ahead of use, so the kernel alternates
   load-wait-compute. Confirm: `tileiras --remarks` Gantt output shows sequential load → compute instead of
   overlapped bars.
4. **The decomposition itself moves too many bytes or strides badly.** Re-reads across blocks, a reduction
   walked along the strided axis, or padding traffic. Confirm: count the minimum bytes the op *must* move
   (inputs once + outputs once) and compare to what achieved-bandwidth arithmetic implies the kernel actually
   moved; a large ratio means the fix is the decomposition, not a knob.
5. **Contended atomic accumulation.** Duplicate-index scatter (`ct.atomic_add`) serializes on hot
   rows, so throughput sits far below peak with atomics in the store path. Confirm: the index
   distribution is skewed and the atomic is per input element.

## Candidate techniques

Ordered by expected value — a reference list, not the full candidate space; explorations beyond it
are encouraged:

1. `tech-tma-load` — replace gather/scatter with hardware TMA `ct.load`/`ct.store`; the
   single most impactful memory choice in the playbook (Optimization A), and the first thing to check on any
   memory-bound kernel.
2. `tech-tile-size` — the most versatile knob: sets transaction granularity and the
   elements-per-thread → registers chain; sweep per architecture rather than reasoning it out.
3. `tech-latency-hint` — `latency=N` on the hottest loads/stores buys prefetch overlap
   for a few percent once the access path is right.
4. Change the decomposition when knobs are neutral and achieved bandwidth is still far from peak:
   recompute a stored input stream instead of loading it (`tech-recompute-vs-reload`); compose
   chained linear ops into one denser op; for skinny-M GEMM
   shapes, stream the big operand once (weight-stationary form); or pick a layout that makes the
   dominant access contiguous.
5. `tech-scatter-to-gather-inversion` — duplicate-index atomic accumulation (cause 5) inverts into
   a rank pass plus a contention-free binned gather-reduce.

## Caveats

- **Near-peak means done.** If achieved bandwidth is close to the board's peak, further knob-tuning is noise;
  the only remaining lever is moving fewer bytes (fusion, smaller dtypes, alternate decomposition).
- Stores are a separate decision from loads: the TMA store path sometimes loses to a plain store — see
  `tech-tma-store-disable`.
- Approximate-math knobs rarely move a genuinely bandwidth-limited kernel. Occupancy can — more resident
  CTAs mean more outstanding loads, and the performance model seeds 8-16 for memory-bound reductions
  (`tech-occupancy`) — but sweep it after the memory path is right, not instead of fixing it.
- Measure with an L2-flushed timer; a tight CUDA-event loop times L2-resident performance and can misclassify
  the kernel entirely.
- At small shapes the same op is often launch-bound, not bandwidth-bound — check the wall-time vs device-time
  gap first (`pattern-host-overhead-bound`).
