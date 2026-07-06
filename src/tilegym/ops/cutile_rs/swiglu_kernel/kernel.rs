// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// swiglu forward — cutile-rs device kernel (pointer scatter/gather).
//   c = SiLU(a) * b,   SiLU(x) = x * sigmoid(x)   (computed in f32).
// One block per row; TILE_SIZE = next_power_of_2(n_cols), column-masked.

#[cutile::module]
pub mod swiglu_module {
    use cutile::core::*;

    /// SwiGLU forward, one block per flattened row.
    #[cutile::entry()]
    pub unsafe fn swiglu_forward_kernel<E: ElementType, const TILE_SIZE: i32>(
        a_ptr: *mut E,
        a_shape0: i32,
        a_shape1: i32,
        a_stride0: i32,
        a_stride1: i32,
        b_ptr: *mut E,
        b_shape0: i32,
        b_shape1: i32,
        b_stride0: i32,
        b_stride1: i32,
        c_ptr: *mut E,
        c_shape0: i32,
        c_shape1: i32,
        c_stride0: i32,
        c_stride1: i32,
    ) {
        // ─── pointer div_by<16> only; scalar bounds_lower<0> ────────
        let a_ptr = unsafe { assume_div_by::<_, 16>(a_ptr) };
        let b_ptr = unsafe { assume_div_by::<_, 16>(b_ptr) };
        let c_ptr = unsafe { assume_div_by::<_, 16>(c_ptr) };

        let a_shape0 = unsafe { assume_bounds_lower::<_, 0>(a_shape0) };
        let a_shape1 = unsafe { assume_bounds_lower::<_, 0>(a_shape1) };
        let a_stride0 = unsafe { assume_bounds_lower::<_, 0>(a_stride0) };
        let b_shape0 = unsafe { assume_bounds_lower::<_, 0>(b_shape0) };
        let b_shape1 = unsafe { assume_bounds_lower::<_, 0>(b_shape1) };
        let b_stride0 = unsafe { assume_bounds_lower::<_, 0>(b_stride0) };
        let c_shape0 = unsafe { assume_bounds_lower::<_, 0>(c_shape0) };
        let c_shape1 = unsafe { assume_bounds_lower::<_, 0>(c_shape1) };
        let c_stride0 = unsafe { assume_bounds_lower::<_, 0>(c_stride0) };
        let _ = a_stride1;
        let _ = b_stride1;
        let _ = c_stride1;

        // ─── 1D grid: each block handles one flattened row ──────────────────
        let row: i32 = get_tile_block_id().0;

        // Column index vector [0, 1, ..., TILE_SIZE-1] and its i64 widening.
        let cols_i32: Tile<i32, { [TILE_SIZE] }> = iota(const_shape![TILE_SIZE]);
        let cols_i64: Tile<i64, { [TILE_SIZE] }> = exti(cols_i32);

        // row widened + broadcast to the tile width (i64).
        let row_i32_s: Tile<i32, { [] }> = scalar_to_tile(row);
        let row_i64_s: Tile<i64, { [] }> = exti(row_i32_s);
        let row_i64: Tile<i64, { [TILE_SIZE] }> = row_i64_s
            .reshape(const_shape![1])
            .broadcast(const_shape![TILE_SIZE]);

        // ─── a gather ───────────────────────────────────────────────────────
        let a_row_ok: Tile<bool, { [TILE_SIZE] }> = {
            let lim_i32: Tile<i32, { [] }> = scalar_to_tile(a_shape0);
            let lim_s: Tile<i64, { [] }> = exti(lim_i32);
            let lim: Tile<i64, { [TILE_SIZE] }> = lim_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            cmpi(row_i64, lim, predicate::LessThan)
        };
        let a_col_ok: Tile<bool, { [TILE_SIZE] }> = {
            let lim_i32: Tile<i32, { [] }> = scalar_to_tile(a_shape1);
            let lim_s: Tile<i64, { [] }> = exti(lim_i32);
            let lim: Tile<i64, { [TILE_SIZE] }> = lim_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            cmpi(cols_i64, lim, predicate::LessThan)
        };
        let a_mask: Tile<bool, { [TILE_SIZE] }> = andi(a_row_ok, a_col_ok);
        let a_off: Tile<i64, { [TILE_SIZE] }> = {
            let str0_i32: Tile<i32, { [] }> = scalar_to_tile(a_stride0);
            let str0_s: Tile<i64, { [] }> = exti(str0_i32);
            let str0: Tile<i64, { [TILE_SIZE] }> = str0_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            let base: Tile<i64, { [TILE_SIZE] }> = row_i64 * str0;
            base + cols_i64
        };
        let a_p0: PointerTile<*mut E, { [] }> = pointer_to_tile(a_ptr);
        let a_p1: PointerTile<*mut E, { [1] }> = a_p0.reshape(const_shape![1]);
        let a_pb: PointerTile<*mut E, { [TILE_SIZE] }> = a_p1.broadcast(const_shape![TILE_SIZE]);
        let a_ptrs: PointerTile<*mut E, { [TILE_SIZE] }> = a_pb.offset_tile(a_off);
        // padding_value: None. The cutile-rs JIT cannot pass a generic
        // `Some(E::ZERO)` (associated-const
        // in expression position). Masked-off lanes are never stored (the same
        // mask gates the scatter) and all math is per-lane, so an undefined pad
        // is functionally identical to a 0 pad here.
        let (a_tile, a_tok): (Tile<E, { [TILE_SIZE] }>, Token) = load_ptr_tko(
            a_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(a_mask),
            None,
            None,
            Latency::<0>,
        );
        let _ = a_tok;

        // ─── b gather ───────────────────────────────────────────────────────
        let b_row_ok: Tile<bool, { [TILE_SIZE] }> = {
            let lim_i32: Tile<i32, { [] }> = scalar_to_tile(b_shape0);
            let lim_s: Tile<i64, { [] }> = exti(lim_i32);
            let lim: Tile<i64, { [TILE_SIZE] }> = lim_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            cmpi(row_i64, lim, predicate::LessThan)
        };
        let b_col_ok: Tile<bool, { [TILE_SIZE] }> = {
            let lim_i32: Tile<i32, { [] }> = scalar_to_tile(b_shape1);
            let lim_s: Tile<i64, { [] }> = exti(lim_i32);
            let lim: Tile<i64, { [TILE_SIZE] }> = lim_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            cmpi(cols_i64, lim, predicate::LessThan)
        };
        let b_mask: Tile<bool, { [TILE_SIZE] }> = andi(b_row_ok, b_col_ok);
        let b_off: Tile<i64, { [TILE_SIZE] }> = {
            let str0_i32: Tile<i32, { [] }> = scalar_to_tile(b_stride0);
            let str0_s: Tile<i64, { [] }> = exti(str0_i32);
            let str0: Tile<i64, { [TILE_SIZE] }> = str0_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            let base: Tile<i64, { [TILE_SIZE] }> = row_i64 * str0;
            base + cols_i64
        };
        let b_p0: PointerTile<*mut E, { [] }> = pointer_to_tile(b_ptr);
        let b_p1: PointerTile<*mut E, { [1] }> = b_p0.reshape(const_shape![1]);
        let b_pb: PointerTile<*mut E, { [TILE_SIZE] }> = b_p1.broadcast(const_shape![TILE_SIZE]);
        let b_ptrs: PointerTile<*mut E, { [TILE_SIZE] }> = b_pb.offset_tile(b_off);
        let (b_tile, b_tok): (Tile<E, { [TILE_SIZE] }>, Token) = load_ptr_tko(
            b_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(b_mask),
            None,
            None,
            Latency::<0>,
        );
        let _ = b_tok;

        // ─── SwiGLU body — sigmoid in f32, silu cast to E, * b ──────────────
        // a in f32 (convert_tile is a no-op when E == f32).
        let a_f32: Tile<f32, { [TILE_SIZE] }> = convert_tile(a_tile);
        // sigmoid(x) = 1 / (1 + exp(-x)) with rounding<approx> + flush_to_zero.
        let neg_a: Tile<f32, { [TILE_SIZE] }> = negf(a_f32);
        let exp_neg: Tile<f32, { [TILE_SIZE] }> = exp(neg_a);
        let one: Tile<f32, { [TILE_SIZE] }> = broadcast_scalar(1.0f32, const_shape![TILE_SIZE]);
        let denom: Tile<f32, { [TILE_SIZE] }> = one + exp_neg;
        let sigmoid: Tile<f32, { [TILE_SIZE] }> = true_div(one, denom);
        // silu(x) = x * sigmoid(x), still f32.
        let silu_f32: Tile<f32, { [TILE_SIZE] }> = a_f32 * sigmoid;
        // cast silu to E, then multiply by b (in E) — matches .astype(a.dtype) * b.
        let silu_e: Tile<E, { [TILE_SIZE] }> = convert_tile(silu_f32);
        let c_tile: Tile<E, { [TILE_SIZE] }> = silu_e * b_tile;

        // ─── c scatter ──────────────────────────────────────────────────────
        let c_row_ok: Tile<bool, { [TILE_SIZE] }> = {
            let lim_i32: Tile<i32, { [] }> = scalar_to_tile(c_shape0);
            let lim_s: Tile<i64, { [] }> = exti(lim_i32);
            let lim: Tile<i64, { [TILE_SIZE] }> = lim_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            cmpi(row_i64, lim, predicate::LessThan)
        };
        let c_col_ok: Tile<bool, { [TILE_SIZE] }> = {
            let lim_i32: Tile<i32, { [] }> = scalar_to_tile(c_shape1);
            let lim_s: Tile<i64, { [] }> = exti(lim_i32);
            let lim: Tile<i64, { [TILE_SIZE] }> = lim_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            cmpi(cols_i64, lim, predicate::LessThan)
        };
        let c_mask: Tile<bool, { [TILE_SIZE] }> = andi(c_row_ok, c_col_ok);
        let c_off: Tile<i64, { [TILE_SIZE] }> = {
            let str0_i32: Tile<i32, { [] }> = scalar_to_tile(c_stride0);
            let str0_s: Tile<i64, { [] }> = exti(str0_i32);
            let str0: Tile<i64, { [TILE_SIZE] }> = str0_s
                .reshape(const_shape![1])
                .broadcast(const_shape![TILE_SIZE]);
            let base: Tile<i64, { [TILE_SIZE] }> = row_i64 * str0;
            base + cols_i64
        };
        let c_p0: PointerTile<*mut E, { [] }> = pointer_to_tile(c_ptr);
        let c_p1: PointerTile<*mut E, { [1] }> = c_p0.reshape(const_shape![1]);
        let c_pb: PointerTile<*mut E, { [TILE_SIZE] }> = c_p1.broadcast(const_shape![TILE_SIZE]);
        let c_ptrs: PointerTile<*mut E, { [TILE_SIZE] }> = c_pb.offset_tile(c_off);
        let c_tok = store_ptr_tko(
            c_ptrs,
            c_tile,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(c_mask),
            None,
            Latency::<0>,
        );
        let _ = c_tok;
    }
}
