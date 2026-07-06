// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
//   C[Q, M, N] = A[Q, M, K] @ B[Q, K, N]   (batched GEMM, device kernel only)
//
// Variants: static_persistent (grid-stride, GROUP_SIZE_M ordering, optional
// transpose) and non_persistent (direct 3D grid, no transpose). f32 accumulator.

#[cutile::module]
pub mod bmm_module {
    use cutile::core::*;

    /// Static-persistent batched matmul (variant 1).
    ///
    /// Const generics (order matters for FFI .generics()):
    ///   E, BM, BN, BK, GROUP_SIZE_M, TRANSPOSE_A, TRANSPOSE_B
    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_100 = (occupancy = 1, num_cta_in_cga = 2,),
        ),
    )]
    pub unsafe fn static_persistent_bmm_kernel<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_SIZE_M: i32,
        const TRANSPOSE_A: i32,
        const TRANSPOSE_B: i32,
    >(
        a: &Tensor<E, { [-1, -1, -1] }>,
        b: &Tensor<E, { [-1, -1, -1] }>,
        c: &Tensor<E, { [-1, -1, -1] }>,
        rt_q: i32,
        rt_m: i32,
        rt_n: i32,
        rt_k: i32,
    ) {
        // scalar lower-bound assumes
        let rt_q = unsafe { assume_bounds_lower::<_, 0>(rt_q) };
        let rt_m = unsafe { assume_bounds_lower::<_, 0>(rt_m) };
        let rt_n = unsafe { assume_bounds_lower::<_, 0>(rt_n) };
        let rt_k = unsafe { assume_bounds_lower::<_, 0>(rt_k) };

        // grouped-schedule tile geometry
        let num_pid_m: i32 = (rt_m + BM - 1) / BM;
        let num_pid_n: i32 = (rt_n + BN - 1) / BN;
        let num_pid_in_batch: i32 = num_pid_m * num_pid_n;
        let total_tiles: i32 = num_pid_in_batch * rt_q;
        let num_pid_in_group: i32 = GROUP_SIZE_M * num_pid_n;
        let num_k: i32 = (rt_k + BK - 1) / BK;

        let bid_x: i32 = get_tile_block_id().0;
        let grid_x: i32 = get_num_tile_blocks().0;

        // output partition (full, schedule-indexed) over read-only &Tensor output
        let mut c_part: PartitionMut<E, { [1, BM, BN] }> =
            unsafe { c.partition_full_mut(const_shape![1, BM, BN]) };

        // persistent grid-stride loop
        for tile_id in (bid_x..total_tiles).step_by(grid_x as usize) {
            let batch: i32 = tile_id / num_pid_in_batch;
            let pid_in_batch: i32 = tile_id % num_pid_in_batch;

            // grouped (M-major) tile ordering
            let group_id: i32 = pid_in_batch / num_pid_in_group;
            let first_pid_m: i32 = group_id * GROUP_SIZE_M;
            let group_size_m_eff: i32 = {
                let rem = num_pid_m - first_pid_m;
                if rem < GROUP_SIZE_M {
                    rem
                } else {
                    GROUP_SIZE_M
                }
            };
            let pid_m: i32 = first_pid_m + (pid_in_batch % group_size_m_eff);
            let pid_n: i32 = (pid_in_batch % num_pid_in_group) / group_size_m_eff;

            // inner K loop with f32 accumulator
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

            for ki in 0i32..num_k {
                // A tile -> [BM, BK]
                let a_part_n: Partition<E, { [1, BM, BK] }> = a.partition(const_shape![1, BM, BK]);
                let a_ld_n: Tile<E, { [1, BM, BK] }> = load_view_tko(
                    &a_part_n,
                    [batch, pid_m, ki],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(3i32),
                    tma::Enabled,
                );
                let mut a_tile: Tile<E, { [BM, BK] }> = a_ld_n.reshape(const_shape![BM, BK]);
                if TRANSPOSE_A == 1i32 {
                    // A physical [Q, K, M]; load [1, BK, BM], permute to [1, BM, BK].
                    let a_part_t: Partition<E, { [1, BK, BM] }> =
                        a.partition(const_shape![1, BK, BM]);
                    let a_ld_t: Tile<E, { [1, BK, BM] }> = load_view_tko(
                        &a_part_t,
                        [batch, ki, pid_m],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(3i32),
                        tma::Enabled,
                    );
                    let a_perm: Tile<E, { [1, BM, BK] }> = permute(a_ld_t, const_array![0, 2, 1]);
                    a_tile = a_perm.reshape(const_shape![BM, BK]);
                } else {
                    a_tile = a_tile;
                }

                // B tile -> [BK, BN]
                let b_part_n: Partition<E, { [1, BK, BN] }> = b.partition(const_shape![1, BK, BN]);
                let b_ld_n: Tile<E, { [1, BK, BN] }> = load_view_tko(
                    &b_part_n,
                    [batch, ki, pid_n],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(3i32),
                    tma::Enabled,
                );
                let mut b_tile: Tile<E, { [BK, BN] }> = b_ld_n.reshape(const_shape![BK, BN]);
                if TRANSPOSE_B == 1i32 {
                    // B physical [Q, N, K]; load [1, BN, BK], permute to [1, BK, BN].
                    let b_part_t: Partition<E, { [1, BN, BK] }> =
                        b.partition(const_shape![1, BN, BK]);
                    let b_ld_t: Tile<E, { [1, BN, BK] }> = load_view_tko(
                        &b_part_t,
                        [batch, pid_n, ki],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(3i32),
                        tma::Enabled,
                    );
                    let b_perm: Tile<E, { [1, BK, BN] }> = permute(b_ld_t, const_array![0, 2, 1]);
                    b_tile = b_perm.reshape(const_shape![BK, BN]);
                } else {
                    b_tile = b_tile;
                }

                acc = mmaf(a_tile, b_tile, acc);
            }

            // cast back to E and store the output tile
            let out2d: Tile<E, { [BM, BN] }> = convert_tile(acc);
            let out3d: Tile<E, { [1, BM, BN] }> = out2d.reshape(const_shape![1, BM, BN]);

            unsafe {
                store_view_tko_mut(
                    &mut c_part,
                    out3d,
                    [batch, pid_m, pid_n],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(3i32),
                    tma::Enabled,
                );
            }
        }
    }

    /// Non-persistent batched matmul (variant 2): direct 3D launch.
    ///
    /// Grid = (cdiv(M,BM), cdiv(N,BN), Q) with x=M, y=N, z=Q.
    /// K-tile count comes from get_index_space_shape on a metadata partition
    /// view. No transpose support.
    ///
    /// Const generics (order matters for FFI .generics()): E, BM, BN, BK
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn non_persistent_bmm_kernel<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
    >(
        a: &Tensor<E, { [-1, -1, -1] }>,
        b: &Tensor<E, { [-1, -1, -1] }>,
        c: &Tensor<E, { [-1, -1, -1] }>,
    ) {
        // 3D block id: x=M-tile, y=N-tile, z=batch
        let (bid_m, bid_n, batch) = get_tile_block_id();

        // metadata partition view (no padding) to query K-tile count, mirroring
        // the reference's get_index_space_shape on a plain partition_view.
        let a_iter: Partition<E, { [1, BM, BK] }> = make_partition_view(
            a,
            const_shape![1, BM, BK],
            padding::None,
            dim_map::Identity,
            get_tensor_token(a),
        );
        let k_space: [i32; 3] = get_index_space_shape(&a_iter);
        let num_k: i32 = k_space[2];

        // padded A/B load partitions
        let a_part: Partition<E, { [1, BM, BK] }> = a.partition(const_shape![1, BM, BK]);
        let b_part: Partition<E, { [1, BK, BN] }> = b.partition(const_shape![1, BK, BN]);

        // output partition (full, schedule-indexed) over read-only &Tensor output
        let mut c_part: PartitionMut<E, { [1, BM, BN] }> =
            unsafe { c.partition_full_mut(const_shape![1, BM, BN]) };

        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

        for ki in 0i32..num_k {
            let a_ld: Tile<E, { [1, BM, BK] }> = load_view_tko(
                &a_part,
                [batch, bid_m, ki],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            let a_tile: Tile<E, { [BM, BK] }> = a_ld.reshape(const_shape![BM, BK]);

            let b_ld: Tile<E, { [1, BK, BN] }> = load_view_tko(
                &b_part,
                [batch, ki, bid_n],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            let b_tile: Tile<E, { [BK, BN] }> = b_ld.reshape(const_shape![BK, BN]);

            acc = mmaf(a_tile, b_tile, acc);
        }

        let out2d: Tile<E, { [BM, BN] }> = convert_tile(acc);
        let out3d: Tile<E, { [1, BM, BN] }> = out2d.reshape(const_shape![1, BM, BN]);

        unsafe {
            store_view_tko_mut(
                &mut c_part,
                out3d,
                [batch, bid_m, bid_n],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }
    }
}
