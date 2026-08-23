---
id: pattern-shape-heterogeneity
kind: pattern
title: Shape heterogeneity (best config varies by input shape)
summary: A candidate wins on some shapes and regresses others because different shapes sit in different regimes — gate the win to the shapes where it holds, autotune per shape, or route between variants; never promote a subset win to a global default.
---

<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Shape heterogeneity (best config varies by input shape)

## Symptom

The per-config benchmark table shows sign flips: the candidate is clearly faster on some shapes and clearly
slower on others, and no single config/kernel wins everywhere. Aggregates lie in both directions here — a
geomean can dilute a real win on the shapes that matter or hide a real regression on a subset. The tell is
disagreement *across* shapes that exceeds the noise band *within* each shape.

## Likely causes

1. **Shapes sit in different bottleneck regimes.** Small shapes are launch-bound, mid shapes bandwidth-bound,
   wide shapes register-bound — one knob setting cannot serve all three. Confirm: classify each shape
   independently (device-vs-wall time gap, achieved bandwidth, registers/occupancy) and see whether the win/
   loss boundary coincides with a regime boundary.
2. **A resource wall is crossed at a threshold.** Registers past a width, activation memory past a size, a
   wave boundary at a block count. Confirm: the flip point is sharp and tracks one input variable (N, batch,
   vocab); identify that variable by sweeping it in isolation.
3. **The config was tuned on a subset.** The candidate was selected on a smoke set, a best node, or the
   motivating shape only, and its "regression" is just the first honest full-matrix measurement. Confirm:
   check what was actually measured before the change was adopted.

## Candidate techniques

Ordered by expected value — a reference list, not the full candidate space; explorations beyond it
are encouraged:

1. Retain the win behind the narrowest measured shape gate (a dtype/shape/arch predicate on public
   input properties, validated on both sides of the boundary); this converts a mixed result into a
   strict improvement without inventing new kernels.
2. When the best *config* (not the best kernel) varies by shape, let per-shape autotune choose it
   at runtime, cached by shape key; configs are seeds to re-measure, not answers.
3. When no single kernel spans the regimes, dispatch between variants at a measured threshold on
   stable public properties (shape, dtype, arch, memory bound); routing and gating are the same
   decision viewed from opposite ends.

## Caveats

- The paired failure mode is a subset-only win promoted to a global default. Every gate or route
  boundary needs measurements on *both* sides.
- Gate keep/revert decisions per config, never on a geomean alone.
- Thresholds are board- and compiler-scoped priors; re-measure the boundary when the board or toolchain
  changes rather than porting the constant.
- Before adding a gate, make sure the cross-shape disagreement is real: near-ties and drifting cells mimic
  heterogeneity; confirm the disagreement reproduces before gating on it.
- Each gate adds a dispatch surface and doubles the kernels to maintain; prefer one kernel when a measured
  single choice is within noise of the gated pair.
