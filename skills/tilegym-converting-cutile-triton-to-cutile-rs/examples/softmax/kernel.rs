// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax — canonical tilegym-converting-cutile-triton-to-cutile-rs example kernel.
//
// This file is the FROM-SCRATCH template you use for the first kernel you
// convert. The cutile-rs Rust lives inside the tilegym tree
// (src/tilegym/ops/cutile_rs/) — there is no separate cutile-rs checkout. It is
// intentionally small so every load-bearing line is visible.
//
// What this kernel demonstrates (the full conversion stack in 1 file):
//
//   1. dtype generic           : `<E: ElementType>`              (Rule 16)
//   2. tile-size const generics: `<const BM, const BN>`          (Rule 23)
//   3. latency const generic   : `<const LATENCY: i32>`          (perf)
//   4. partition view I/O      : `tview.partition(shape)`        (Rule 28 v2)
//   5. assume hints            : pointer div_by + scalar bounds  (Rule 21 v2)
//   6. tile-form math          : reduce_max / exp2 / reduce_sum  (op-mapping.md)
//   7. boundary safety         : padding_value=zero auto-emitted
//   8. TMA on by default       : `tma::Enabled` everywhere       (perf)
//
// What you MUST swap when reusing this template:
//
//   * `softmax_module`         → your own `{kernel_name}_module`
//   * `softmax_kernel`         → your own entry name
//   * the entry signature      → the args your op needs
//   * the body                 → your op's math
//   * everything else (assume hints, latency wiring, partition pattern,
//     `<E: ElementType>`, optimization_hints) is boilerplate — keep as-is.
//
// Cross-references:
//   * `references/coding-rules.md`   — all rules cited above
//   * `references/op-mapping.md`     — cuTile-py / Triton-TileIR op  →  cutile-rs op
//   * `concepts/strided-view-to-partition-view.md` — when the reference IR
//     comes from Triton-TileIR (uses element-level offsets), how to translate to
//     cutile-rs's tile-level partition view indexing.
//   * `concepts/tensor-vs-pointer-pattern.md` — when to pick partition-view
//     (this template) vs raw-pointer scatter/gather.
//
// Naming convention: `kernel.rs` lives at
//     ${CUTILE_KERNEL_OUT_ROOT}/softmax/kernel.rs
// (per-kernel working dir) and is mirrored into the aggregated crate's sibling
//     src/tilegym/ops/cutile_rs/softmax_kernel/kernel.rs
// where it is `include!()`-d by both:
//     src/tilegym/ops/cutile_rs/cutile_kernels/src/lib.rs   (FFI export — one
//         `mod softmax { include!("../../softmax_kernel/kernel.rs"); ... }`)
//     ${CUTILE_KERNEL_OUT_ROOT}/softmax/softmax_pipeline.rs (compile test)

#[cutile::module]
pub mod softmax_module {
    use cutile::core::*;

    /// Online softmax along the last (N) dimension.
    ///
    /// Layout: `y[m, n] = exp(x[m, n] - max(x[m, :])) / sum(exp(x[m, :] - max))`.
    /// Tiled M×N at `(BM, BN)` granularity.  N must satisfy `N <= BN` for the
    /// vanilla version; for `N > BN` you would add an inner reduce loop (see
    /// `cutile-examples/examples/softmax.rs` for the multi-tile variant).
    ///
    /// Const generics:
    ///   * `E`        : data element type (f16 / bf16 / f32) — Rule 16
    ///   * `BM`, `BN` : tile sizes — Rule 23
    ///   * `LATENCY`  : pipeline depth hint passed to every load/store — perf
    ///
    /// Runtime args (all `unsafe` — caller must ensure validity):
    ///   * `y_ptr`, `x_ptr` : raw pointers into device memory.
    ///   * `m`, `n`         : logical rows and columns.
    ///   * `s_m`, `s_n`     : strides (innermost typically 1).
    ///
    /// Grid: 1D over `M / BM` rows. Host-side launcher computes
    /// `grid = ceildiv(m, BM)` and passes via `CompileOptions`.
    #[cutile::entry(
        unchecked_accesses = true,
        // sm_100: occupancy + num_cta_in_cga. Adjust per Agent A's analysis.json.
        // For sm_80 / sm_90 / sm_120 add additional `sm_XX = (...)` blocks here.
        optimization_hints = (
            sm_100 = (occupancy = 2, num_cta_in_cga = 1,),
        ),
    )]
    unsafe fn softmax_kernel<E: ElementType, const BM: i32, const BN: i32, const LATENCY: i32>(
        y_ptr: *mut E,
        x_ptr: *mut E,
        m: i32,
        n: i32,
        s_m: i32,
        s_n: i32,
    ) {
        // ─── Rule 21 v2: pointer div_by + scalar bounds_lower ──────────────
        //
        // POINTER assumes (always 16-byte aligned for fp16/bf16/f32 in PyTorch
        // tensor allocations) → enables vectorized loads/stores in tileiras.
        let y_ptr = unsafe { assume_div_by::<_, 16>(y_ptr) };
        let x_ptr = unsafe { assume_div_by::<_, 16>(x_ptr) };
        // SCALAR assumes — bounds_lower<0> only. NEVER `assume_div_by` on
        // scalars: when N doesn't divide the runtime value the compiler
        // silently produces wrong output (see memory `feedback_cutile_rs_
        // scalar_assume_div_by` and Rule 21 v2 history of K=511 corruption).
        let m = unsafe { assume_bounds_lower::<_, 0>(m) };
        let n = unsafe { assume_bounds_lower::<_, 0>(n) };
        let s_m = unsafe { assume_bounds_lower::<_, 0>(s_m) };
        let s_n = unsafe { assume_bounds_lower::<_, 0>(s_n) };

        // ─── 1D grid: each block handles one BM-row strip ──────────────────
        // `get_tile_block_id()` is the canonical 3D-tuple block-id intrinsic
        // (cutile/src/_core.rs:1029). Use the .0 field for 1-D grids.
        // `block_id::<N>()` does NOT exist — older drafts of this file referenced
        // it; that was a typo. Always use `get_tile_block_id()`.
        let bid_m = get_tile_block_id().0;

        // ─── Tensor views over the flat pointers ───────────────────────────
        let x_tview: TensorView<E, 2> = unsafe { make_tensor_view(x_ptr, [m, n], [s_m, s_n]) };
        let mut y_tview: TensorView<E, 2> = unsafe { make_tensor_view(y_ptr, [m, n], [s_m, s_n]) };

        // ─── Boundary-safe partition views (Rule 28 v2) ────────────────────
        //
        // `tview.partition(const_shape![BM, BN])` is the cutile-rs shorthand
        // for `make_partition_view(self, shape, padding::Zero, dim_map::
        // Identity, token)`. The `padding_value = zero` it emits is what makes
        // the kernel boundary-safe on M%BM != 0 / N%BN != 0 shapes WITHOUT
        // needing to disable TMA — keep `tma::Enabled` (memory
        // `feedback_keep_tma_enabled_for_perf`).
        let x_part: Partition<E, { [BM, BN] }> = x_tview.partition(const_shape![BM, BN]);
        let mut y_part: PartitionMut<E, { [BM, BN] }> =
            unsafe { y_tview.partition_mut(const_shape![BM, BN]) };

        // ─── Load: TMA on, latency hint per op ────────────────────────────
        //
        // Reference IR (Triton-TileIR or cuTile-py) typically has `latency = N` on
        // every load/store; matching it enables the same software pipelining
        // depth in tileiras. Pass `Some(LATENCY)` (NOT `None`) — `None`
        // inherits entry default which is often suboptimal.
        let x_tile: Tile<E, { [BM, BN] }> = load_view_tko(
            &x_part,
            [bid_m, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );

        // ─── Online softmax body — tile-form math ──────────────────────────
        //
        // 1. Cast input to f32 for stable reduction (fp16/bf16 → f32) — Rule 8.
        //    `convert_tile` is a no-op when E == f32.
        let x_f32: Tile<f32, { [BM, BN] }> = convert_tile(x_tile);

        // 2. row-wise max  : Tile<f32, [BM, BN]>  →  Tile<f32, [BM]>
        let row_max: Tile<f32, { [BM] }> = reduce_max(x_f32, 1i32);

        // 3. broadcast row_max back to (BM, BN) and subtract.
        let row_max_2d: Tile<f32, { [BM, BN] }> = row_max
            .reshape(const_shape![BM, 1])
            .broadcast(x_f32.shape());
        let shifted: Tile<f32, { [BM, BN] }> = x_f32 - row_max_2d;

        // 4. exp via exp2(x * INV_LOG_2). flush_to_zero matches Triton-TileIR/cuTile-py
        //    reference IR (Rule 26 — Agent C will fail you if it's missing).
        let inv_log2: Tile<f32, { [BM, BN] }> =
            broadcast_scalar(constant(std::f32::consts::LOG2_E), shifted.shape());
        let exped: Tile<f32, { [BM, BN] }> = exp2(mulf(shifted, inv_log2, ftz::Enabled));

        // 5. row-wise sum + divide.
        let row_sum: Tile<f32, { [BM] }> = reduce_sum(exped, 1i32);
        let row_sum_2d: Tile<f32, { [BM, BN] }> = row_sum
            .reshape(const_shape![BM, 1])
            .broadcast(exped.shape());
        // `true_div` (rounding<approx> + flush_to_zero) matches the Triton-TileIR
        // reference. `divf` with `rounding::Approx` hits a known cutile-rs
        // encoding bug (Approx→5 vs printer<approx>→4) — use `true_div`.
        let normalized: Tile<f32, { [BM, BN] }> = true_div(exped, row_sum_2d);

        // 6. Cast back to E and store.
        let result: Tile<E, { [BM, BN] }> = convert_tile(normalized);

        unsafe {
            store_view_tko_mut(
                &mut y_part,
                result,
                [bid_m, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
        }
    }
}
