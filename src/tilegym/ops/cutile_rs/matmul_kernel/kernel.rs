// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// matmul — cutile-rs device kernel. Variants: non_persistent, static_persistent.
// f32 inputs cast to tf32 before mmaf; f32 accumulator.
//
// const-generic order (MUST match the FFI .generics()):
//   non_persistent:    <E, BM, BN, BK, CAST_TF32>
//   static_persistent: <E, BM, BN, BK, GROUP_SIZE_M, CAST_TF32>

#[cutile::module]
pub mod matmul_module {
    use cutile::core::*;

    // ── non-persistent matmul: C = A @ B, 1D grid = cdiv(M,BM)*cdiv(N,BN) ──
    //
    // GROUP_SIZE_M = 8 swizzle is hardcoded inside the body. num K-tiles comes
    // from `get_index_space_shape` over an unpadded metadata partition.
    #[cutile::entry()]
    pub unsafe fn non_persistent_matmul_kernel<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const CAST_TF32: i32,
    >(
        a: &Tensor<E, { [-1, -1] }>,
        b: &Tensor<E, { [-1, -1] }>,
        c: &Tensor<E, { [-1, -1] }>, // OUTPUT — read-only param type; written via partition_full_mut
    ) {
        // Scalar grid math from metadata (no real get_tensor_shape op).
        let a_shape: Shape<{ [-1, -1] }> = a.shape();
        let b_shape: Shape<{ [-1, -1] }> = b.shape();
        let m: i32 = a_shape[0];
        let n: i32 = b_shape[1];

        let bid: i32 = get_tile_block_id().0;

        // GROUP_SIZE_M = 8 swizzle.
        let group_size_m: i32 = 8i32;
        let num_bid_m: i32 = (m + BM - 1i32) / BM;
        let num_bid_n: i32 = (n + BN - 1i32) / BN;
        let num_bid_in_group: i32 = group_size_m * num_bid_n;
        let group_id: i32 = bid / num_bid_in_group;
        let first_bid_m: i32 = group_id * group_size_m;
        let gsm: i32 = {
            let rem: i32 = num_bid_m - first_bid_m;
            if rem < group_size_m {
                rem
            } else {
                group_size_m
            }
        };
        let bid_m: i32 = first_bid_m + (bid % gsm);
        let bid_n: i32 = (bid % num_bid_in_group) / gsm;

        // Unpadded metadata partition for the K-tile count (mirrors reference).
        let token: Token = get_tensor_token(a);
        let a_iter: Partition<E, { [BM, BK] }> = make_partition_view(
            a,
            const_shape![BM, BK],
            padding::None,
            dim_map::Identity,
            token,
        );
        let a_space: [i32; 2] = get_index_space_shape(&a_iter);
        let num_tiles_k: i32 = a_space[1];

        // Padded partition views for the actual loads (auto padding_value = zero).
        let a_part: Partition<E, { [BM, BK] }> = a.partition(const_shape![BM, BK]);
        let b_part: Partition<E, { [BK, BN] }> = b.partition(const_shape![BK, BN]);

        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

        for k in 0i32..num_tiles_k {
            let a_tile: Tile<E, { [BM, BK] }> = load_view_tko(
                &a_part,
                [bid_m, k],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            let b_tile: Tile<E, { [BK, BN] }> = load_view_tko(
                &b_part,
                [k, bid_n],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            if CAST_TF32 == 1i32 {
                let a_tf32: Tile<tf32, { [BM, BK] }> = convert_tile(a_tile);
                let b_tf32: Tile<tf32, { [BK, BN] }> = convert_tile(b_tile);
                acc = mmaf(a_tf32, b_tf32, acc);
            } else {
                acc = mmaf(a_tile, b_tile, acc);
            }
        }

        let result: Tile<E, { [BM, BN] }> = convert_tile(acc);

        let mut c_part: PartitionMut<E, { [BM, BN] }> =
            unsafe { c.partition_full_mut(const_shape![BM, BN]) };
        unsafe {
            store_view_tko_mut(
                &mut c_part,
                result,
                [bid_m, bid_n],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }
    }

    // ── static-persistent matmul: grid-stride loop over output tiles ──
    //
    // M, N, K are runtime scalars. The outer persistent loop steps by the grid
    // size; the inner swizzle + K-loop mirror the non-persistent body.
    // GROUP_SIZE_M is a const generic here.
    #[cutile::entry()]
    pub unsafe fn static_persistent_matmul_kernel<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_SIZE_M: i32,
        const CAST_TF32: i32,
    >(
        a: &Tensor<E, { [-1, -1] }>,
        b: &Tensor<E, { [-1, -1] }>,
        c: &Tensor<E, { [-1, -1] }>, // OUTPUT — read-only param type
        m: i32,
        n: i32,
        k: i32,
    ) {
        let num_bid_m: i32 = (m + BM - 1i32) / BM;
        let num_bid_n: i32 = (n + BN - 1i32) / BN;
        let k_tiles: i32 = (k + BK - 1i32) / BK;
        let num_tiles: i32 = num_bid_m * num_bid_n;

        let start_bid: i32 = get_tile_block_id().0;
        let grid_x: i32 = get_num_tile_blocks().0;

        let num_bid_in_group: i32 = GROUP_SIZE_M * num_bid_n;

        let a_part: Partition<E, { [BM, BK] }> = a.partition(const_shape![BM, BK]);
        let b_part: Partition<E, { [BK, BN] }> = b.partition(const_shape![BK, BN]);
        let mut c_part: PartitionMut<E, { [BM, BN] }> =
            unsafe { c.partition_full_mut(const_shape![BM, BN]) };

        for tile_id in (start_bid..num_tiles).step_by(grid_x as usize) {
            // GROUP_SIZE_M swizzle on tile_id.
            let group_id: i32 = tile_id / num_bid_in_group;
            let first_bid_m: i32 = group_id * GROUP_SIZE_M;
            let gsm: i32 = {
                let rem: i32 = num_bid_m - first_bid_m;
                if rem < GROUP_SIZE_M {
                    rem
                } else {
                    GROUP_SIZE_M
                }
            };
            let bid_m: i32 = first_bid_m + (tile_id % gsm);
            let bid_n: i32 = (tile_id % num_bid_in_group) / gsm;

            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

            for kt in 0i32..k_tiles {
                let a_tile: Tile<E, { [BM, BK] }> = load_view_tko(
                    &a_part,
                    [bid_m, kt],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let b_tile: Tile<E, { [BK, BN] }> = load_view_tko(
                    &b_part,
                    [kt, bid_n],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                if CAST_TF32 == 1i32 {
                    let a_tf32: Tile<tf32, { [BM, BK] }> = convert_tile(a_tile);
                    let b_tf32: Tile<tf32, { [BK, BN] }> = convert_tile(b_tile);
                    acc = mmaf(a_tf32, b_tf32, acc);
                } else {
                    acc = mmaf(a_tile, b_tile, acc);
                }
            }

            let result: Tile<E, { [BM, BN] }> = convert_tile(acc);
            unsafe {
                store_view_tko_mut(
                    &mut c_part,
                    result,
                    [bid_m, bid_n],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
            }
        }
    }
}
