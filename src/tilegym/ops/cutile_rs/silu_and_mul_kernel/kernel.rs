// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// silu_and_mul — row-wise elementwise (pointer scatter/gather).
//   out[row, c] = silu(a) * b,   a = input[row, c], b = input[row, hidden_size + c]
// One block per row; BLOCK_SIZE = next_power_of_2(hidden_size), column-masked.

#[cutile::module]
pub mod silu_and_mul_module {
    use cutile::core::*;

    /// SiLU-and-multiply, row-wise pointer scatter/gather.
    ///
    /// Const generics:
    ///   * `E`          : data element type (f16 / bf16 / f32).
    ///   * `BLOCK_SIZE` : tile width = next_power_of_2(hidden_size).
    ///
    /// Runtime args (all `unsafe` — caller ensures validity):
    ///   * `out_ptr`           : output base pointer (device).
    ///   * `in_ptr`            : input base pointer (device).
    ///   * `hidden_size`       : logical output columns; mask bound.
    ///   * `in_row_stride`     : input  row stride in elements (= 2 * hidden_size).
    ///   * `out_row_stride`    : output row stride in elements (= hidden_size).
    ///
    /// Grid: 1D over n_rows = product of leading dims. One block per row.
    #[cutile::entry()]
    pub unsafe fn silu_and_mul_kernel<E: ElementType, const BLOCK_SIZE: i32>(
        out_ptr: *mut E,
        in_ptr: *mut E,
        hidden_size: i32,
        in_row_stride: i32,
        out_row_stride: i32,
    ) {
        // Pointer alignment assumes: only on the two base pointers.
        let out_ptr = unsafe { assume_div_by::<_, 16>(out_ptr) };
        let in_ptr = unsafe { assume_div_by::<_, 16>(in_ptr) };

        // Flat 1D grid: one block per row.
        let pid: i32 = get_tile_block_id().0;

        // pid -> i64 row offsets (mirror reference: exti signed, muli by strides).
        let pid32: Tile<i32, { [] }> = scalar_to_tile(pid);
        let pid64: Tile<i64, { [] }> = exti(pid32);

        let in_stride32: Tile<i32, { [] }> = scalar_to_tile(in_row_stride);
        let in_stride64: Tile<i64, { [] }> = exti(in_stride32);
        let out_stride32: Tile<i32, { [] }> = scalar_to_tile(out_row_stride);
        let out_stride64: Tile<i64, { [] }> = exti(out_stride32);

        let in_row_off: Tile<i64, { [] }> = pid64 * in_stride64;
        let out_row_off: Tile<i64, { [] }> = pid64 * out_stride64;

        // Base pointers for this row.
        let in_base0: PointerTile<*mut E, { [] }> = pointer_to_tile(in_ptr);
        let in_row_base: PointerTile<*mut E, { [] }> = in_base0.offset_tile(in_row_off);
        let out_base0: PointerTile<*mut E, { [] }> = pointer_to_tile(out_ptr);
        let out_row_base: PointerTile<*mut E, { [] }> = out_base0.offset_tile(out_row_off);

        // Column indices + mask (iota < hidden_size), reused on loads + store.
        let cols: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let hidden_tile: Tile<i32, { [BLOCK_SIZE] }> =
            broadcast_scalar(hidden_size, const_shape![BLOCK_SIZE]);
        let mask: Tile<bool, { [BLOCK_SIZE] }> = cmpi(cols, hidden_tile, predicate::LessThan);

        // a = input[row, col] : reshape(ptr->1) -> broadcast(BLOCK) -> offset(cols).
        let a_p0: PointerTile<*mut E, { [1] }> = in_row_base.reshape(const_shape![1]);
        let a_ps: PointerTile<*mut E, { [BLOCK_SIZE] }> = a_p0.broadcast(const_shape![BLOCK_SIZE]);
        let a_ptrs: PointerTile<*mut E, { [BLOCK_SIZE] }> = a_ps.offset_tile(cols);
        let (a_tile, a_tok): (Tile<E, { [BLOCK_SIZE] }>, Token) = load_ptr_tko(
            a_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None::<E>,
            None::<Token>,
            Latency::<3>,
        );
        let _ = a_tok;

        // b = input[row, hidden_size + col].
        let b_row_base: PointerTile<*mut E, { [] }> = in_row_base.offset(hidden_size);
        let b_p0: PointerTile<*mut E, { [1] }> = b_row_base.reshape(const_shape![1]);
        let b_ps: PointerTile<*mut E, { [BLOCK_SIZE] }> = b_p0.broadcast(const_shape![BLOCK_SIZE]);
        let b_ptrs: PointerTile<*mut E, { [BLOCK_SIZE] }> = b_ps.offset_tile(cols);
        let (b_tile, b_tok): (Tile<E, { [BLOCK_SIZE] }>, Token) = load_ptr_tko(
            b_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None::<E>,
            None::<Token>,
            Latency::<3>,
        );
        let _ = b_tok;

        // Cast to f32 for compute (no-op when E == f32 -> DCE-elided).
        let a_f32: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(a_tile);
        let b_f32: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(b_tile);

        // silu(a) = a * (e / (1 + e)), e = exp2(a * log2e). All FTZ enabled.
        let log2e: Tile<f32, { [BLOCK_SIZE] }> = constant(1.44269502f32, const_shape![BLOCK_SIZE]);
        let scaled: Tile<f32, { [BLOCK_SIZE] }> =
            mulf(a_f32, log2e, rounding::NearestEven, ftz::Enabled);
        let e: Tile<f32, { [BLOCK_SIZE] }> = exp2(scaled, ftz::Enabled);
        let one: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0f32, const_shape![BLOCK_SIZE]);
        let one_plus_e: Tile<f32, { [BLOCK_SIZE] }> =
            addf(e, one, rounding::NearestEven, ftz::Enabled);
        // Reference uses `divf rounding<approx> flush_to_zero` for e/(1+e). cutile-rs
        // `divf(.., rounding::Approx, ..)` mis-encodes to `rounding<full>`;
        // `true_div` lowers to the matching `rounding<approx> flush_to_zero`.
        let sig: Tile<f32, { [BLOCK_SIZE] }> = true_div(e, one_plus_e);
        let silu: Tile<f32, { [BLOCK_SIZE] }> =
            mulf(a_f32, sig, rounding::NearestEven, ftz::Enabled);
        let result_f32: Tile<f32, { [BLOCK_SIZE] }> =
            mulf(silu, b_f32, rounding::NearestEven, ftz::Enabled);

        // Cast back to output dtype.
        let result: Tile<E, { [BLOCK_SIZE] }> = convert_tile(result_f32);

        // out[row, col] = result.
        let o_p0: PointerTile<*mut E, { [1] }> = out_row_base.reshape(const_shape![1]);
        let o_ps: PointerTile<*mut E, { [BLOCK_SIZE] }> = o_p0.broadcast(const_shape![BLOCK_SIZE]);
        let o_ptrs: PointerTile<*mut E, { [BLOCK_SIZE] }> = o_ps.offset_tile(cols);
        let _store_tok: Token = store_ptr_tko(
            o_ptrs,
            result,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None::<Token>,
            Latency::<3>,
        );
    }
}
