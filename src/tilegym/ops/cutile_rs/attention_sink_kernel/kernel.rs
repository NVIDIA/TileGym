// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// attention_sink — flash-attention forward with attention-sink tokens.
// cutile-rs port of cuTile-Python `_attention_sink_kernel`. Online softmax with
// causal mask; bf16 QK/PV mma, f32 accumulate.

#[cutile::module]
pub mod attention_sink_module {
    use cutile::core::*;

    /// Const generics (SAME ORDER as the cuTile-Python kernel param list after the
    /// tensor/scalar runtime args):
    ///   * `E`                 : data element type (bf16 / f16)
    ///   * `TILE_D`            : head dim tile (= head_dim)
    ///   * `H`                 : number of heads (n_kv_heads * query_group_size)
    ///   * `N_KV_CTX`          : key/value context length
    ///   * `TILE_M`, `TILE_N`  : query / key tile sizes
    ///   * `QUERY_GROUP_SIZE`  : repeat_kv (group size)
    ///   * `BANDWIDTH`         : sliding window (0 = no sliding window) — ct.Constant[int]
    ///
    /// Runtime args:
    ///   * `q`,`k`,`v`         : rank-4 tensor views [bs, heads, seq, head_dim] (physical)
    ///   * `sinks`             : rank-1 tensor view [heads]
    ///   * `out`               : rank-4 tensor view [bs, heads, n_ctx, head_dim] (OUTPUT, read-only param)
    ///   * `start_q`           : rank-1 i32 tensor view [1]
    ///   * `qk_scale`          : f32 runtime scalar (raw sm_scale)
    #[cutile::entry(
        optimization_hints = (
            sm_100 = (occupancy = 2,),
        ),
    )]
    pub unsafe fn attention_sink_kernel<
        E: ElementType,
        const TILE_D: i32,
        const H: i32,
        const N_KV_CTX: i32,
        const TILE_M: i32,
        const TILE_N: i32,
        const QUERY_GROUP_SIZE: i32,
        const BANDWIDTH: i32,
    >(
        q: &Tensor<E, { [-1, -1, -1, -1] }>,
        k: &Tensor<E, { [-1, -1, -1, -1] }>,
        v: &Tensor<E, { [-1, -1, -1, -1] }>,
        sinks: &Tensor<E, { [-1] }>,
        out: &Tensor<E, { [-1, -1, -1, -1] }>,
        start_q: &Tensor<i32, { [-1] }>,
        qk_scale: f32,
    ) {
        // ─── block ids → batch / head / kv-head ────────────────────────────
        let (bid_x, bid_y, _bid_z) = get_tile_block_id();
        let batch_idx: i32 = bid_y / H;
        let head_idx: i32 = bid_y % H;
        let off_kv_h: i32 = head_idx / QUERY_GROUP_SIZE;

        // ─── load start_q from tensor [1] → scalar ─────────────────────────
        let sq_part: Partition<i32, { [1] }> = start_q.partition(const_shape![1]);
        let sq_tile: Tile<i32, { [1] }> = load_view_tko(
            &sq_part,
            [0i32],
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        let start_q_t: Tile<i32, { [] }> = sq_tile.reshape(const_shape![]);
        let start_q_scalar: i32 = tile_to_scalar(start_q_t);

        // ─── qk_scale adjusted for exp2 (base-2): * 1/ln2 ──────────────────
        let qk_scale_log2: f32 = qk_scale * 1.4426950408889634f32;

        // ─── load sink for this head, sink_scaled = sink * 1/ln2 ───────────
        let sink_part: Partition<E, { [1] }> = sinks.partition(const_shape![1]);
        let sink_tile: Tile<E, { [1] }> = load_view_tko(
            &sink_part,
            [head_idx],
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        let sink_f32_1: Tile<f32, { [1] }> = convert_tile(sink_tile);
        let sink_f32: Tile<f32, { [] }> = sink_f32_1.reshape(const_shape![]);
        let sink_scalar: f32 = tile_to_scalar(sink_f32);
        let sink_scaled: f32 = sink_scalar * 1.4426950408889634f32;

        // ─── query-tile row offsets offs_m = bid_x*TILE_M + arange(TILE_M) ─
        let row_base: i32 = bid_x * TILE_M;
        let iota_m: Tile<i32, { [TILE_M] }> = iota(const_shape![TILE_M]);
        let row_base_b: Tile<i32, { [TILE_M] }> = broadcast_scalar(row_base, const_shape![TILE_M]);
        let offs_m_1d: Tile<i32, { [TILE_M] }> = row_base_b + iota_m;
        let offs_m: Tile<i32, { [TILE_M, 1] }> = offs_m_1d.reshape(const_shape![TILE_M, 1]);

        // local key offsets offs_n_tile = arange(TILE_N)  → [1, TILE_N]
        let iota_n: Tile<i32, { [TILE_N] }> = iota(const_shape![TILE_N]);
        let offs_n_tile: Tile<i32, { [1, TILE_N] }> = iota_n.reshape(const_shape![1, TILE_N]);

        // ─── online-softmax accumulators ───────────────────────────────────
        // m_i starts at sink_scaled (sink contributes to running max).
        let m_i_init: Tile<f32, { [TILE_M, 1] }> =
            broadcast_scalar(sink_scaled, const_shape![TILE_M, 1]);
        let l_i_init: Tile<f32, { [TILE_M, 1] }> = constant(0.0f32, const_shape![TILE_M, 1]);
        let acc_init: Tile<f32, { [TILE_M, TILE_D] }> =
            constant(0.0f32, const_shape![TILE_M, TILE_D]);

        // ─── load query tile q[batch, head, bid_x, 0] → [TILE_M, TILE_D] ───
        let q_part: Partition<E, { [1, 1, TILE_M, TILE_D] }> =
            q.partition(const_shape![1, 1, TILE_M, TILE_D]);
        let q_ld: Tile<E, { [1, 1, TILE_M, TILE_D] }> = load_view_tko(
            &q_part,
            [batch_idx, head_idx, bid_x, 0i32],
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        let q_tile: Tile<E, { [TILE_M, TILE_D] }> = q_ld.reshape(const_shape![TILE_M, TILE_D]);

        // ─── loop bounds ───────────────────────────────────────────────────
        let hi_raw: i32 = start_q_scalar + (bid_x + 1i32) * TILE_M;
        let hi: i32 = if hi_raw < N_KV_CTX { hi_raw } else { N_KV_CTX };
        let tc: i32 = (hi + TILE_N - 1i32) / TILE_N;
        let start_block: i32 = if BANDWIDTH > 0i32 {
            let lo_raw: i32 = start_q_scalar + bid_x * TILE_M - BANDWIDTH;
            let lo: i32 = if lo_raw > 0i32 { lo_raw } else { 0i32 };
            lo / TILE_N
        } else {
            0i32
        };

        // K transposed partition (dim_map=[0,1,3,2]) — load shape [1,1,TILE_D,TILE_N]
        let k_part: Partition<E, { [1, 1, TILE_D, TILE_N] }> =
            k.partition_permuted(const_shape![1, 1, TILE_D, TILE_N], const_array![0, 1, 3, 2]);
        // V partition — load shape [1,1,TILE_N,TILE_D]
        let v_part: Partition<E, { [1, 1, TILE_N, TILE_D] }> =
            v.partition(const_shape![1, 1, TILE_N, TILE_D]);

        // ─── online-softmax loop over K/V blocks ───────────────────────────
        let mut acc: Tile<f32, { [TILE_M, TILE_D] }> = acc_init;
        let mut l_i: Tile<f32, { [TILE_M, 1] }> = l_i_init;
        let mut m_i: Tile<f32, { [TILE_M, 1] }> = m_i_init;

        for j in start_block..tc {
            let start_n: i32 = j * TILE_N;
            let start_n_b: Tile<i32, { [1, TILE_N] }> =
                broadcast_scalar(start_n, const_shape![1, TILE_N]);
            let offs_n: Tile<i32, { [1, TILE_N] }> = start_n_b + offs_n_tile;

            // ─── load K transposed → [TILE_D, TILE_N] ──────────────────────
            let k_ld: Tile<E, { [1, 1, TILE_D, TILE_N] }> = load_view_tko(
                &k_part,
                [batch_idx, off_kv_h, 0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(6i32),
                tma::Enabled,
            );
            let k_tile: Tile<E, { [TILE_D, TILE_N] }> = k_ld.reshape(const_shape![TILE_D, TILE_N]);

            // ─── QK = q @ k  (bf16 x bf16 -> f32) ──────────────────────────
            let qk_zero: Tile<f32, { [TILE_M, TILE_N] }> =
                constant(0.0f32, const_shape![TILE_M, TILE_N]);
            let qk_mma: Tile<f32, { [TILE_M, TILE_N] }> = mmaf(q_tile, k_tile, qk_zero);

            // ─── masking ───────────────────────────────────────────────────
            // query_pos = start_q + offs_m  → [TILE_M, 1]
            let start_q_b: Tile<i32, { [TILE_M, 1] }> =
                broadcast_scalar(start_q_scalar, const_shape![TILE_M, 1]);
            let query_pos: Tile<i32, { [TILE_M, 1] }> = start_q_b + offs_m;

            // causal_mask = offs_n > query_pos  (broadcast to [TILE_M, TILE_N])
            let offs_n_2d: Tile<i32, { [TILE_M, TILE_N] }> =
                offs_n.broadcast(const_shape![TILE_M, TILE_N]);
            let query_pos_2d: Tile<i32, { [TILE_M, TILE_N] }> =
                query_pos.broadcast(const_shape![TILE_M, TILE_N]);
            let causal_mask: Tile<bool, { [TILE_M, TILE_N] }> =
                cmpi(offs_n_2d, query_pos_2d, predicate::GreaterThan);

            // oob_mask = offs_n >= N_KV_CTX  (on the [1, TILE_N] row, then broadcast)
            let nkv_b: Tile<i32, { [1, TILE_N] }> =
                broadcast_scalar(N_KV_CTX, const_shape![1, TILE_N]);
            let oob_row: Tile<bool, { [1, TILE_N] }> =
                cmpi(offs_n, nkv_b, predicate::GreaterThanOrEqual);
            let oob_mask: Tile<bool, { [TILE_M, TILE_N] }> =
                oob_row.broadcast(const_shape![TILE_M, TILE_N]);

            let mut mask: Tile<bool, { [TILE_M, TILE_N] }> = ori(causal_mask, oob_mask);
            if BANDWIDTH > 0i32 {
                // too_old = offs_n < (query_pos - BANDWIDTH + 1)
                let bw_b: Tile<i32, { [TILE_M, 1] }> =
                    broadcast_scalar(BANDWIDTH - 1i32, const_shape![TILE_M, 1]);
                let too_old_lim: Tile<i32, { [TILE_M, 1] }> = query_pos - bw_b;
                let too_old_lim_2d: Tile<i32, { [TILE_M, TILE_N] }> =
                    too_old_lim.broadcast(const_shape![TILE_M, TILE_N]);
                let too_old: Tile<bool, { [TILE_M, TILE_N] }> =
                    cmpi(offs_n_2d, too_old_lim_2d, predicate::LessThan);
                mask = ori(mask, too_old);
            } else {
                mask = mask;
            }

            // qk = qk + where(mask, -1e6, 0.0)
            let neg_big: Tile<f32, { [TILE_M, TILE_N] }> =
                constant(-1000000.0f32, const_shape![TILE_M, TILE_N]);
            let zero_t: Tile<f32, { [TILE_M, TILE_N] }> =
                constant(0.0f32, const_shape![TILE_M, TILE_N]);
            let mask_add: Tile<f32, { [TILE_M, TILE_N] }> = select(mask, neg_big, zero_t);
            let qk: Tile<f32, { [TILE_M, TILE_N] }> = qk_mma + mask_add;

            // ─── online softmax update ─────────────────────────────────────
            // m_ij = max(m_i, max(qk, dim=1) * qk_scale_log2)
            let row_max: Tile<f32, { [TILE_M] }> = reduce_max(qk, 1i32);
            let row_max_col: Tile<f32, { [TILE_M, 1] }> = row_max.reshape(const_shape![TILE_M, 1]);
            let scale_col: Tile<f32, { [TILE_M, 1] }> =
                broadcast_scalar(qk_scale_log2, const_shape![TILE_M, 1]);
            let row_max_scaled: Tile<f32, { [TILE_M, 1] }> = row_max_col * scale_col;
            let m_ij: Tile<f32, { [TILE_M, 1] }> =
                maxf(m_i, row_max_scaled, nan::Disabled, ftz::Disabled);

            // qk_scaled = qk * qk_scale_log2 - m_ij  (fma form: qk * scale + (-m_ij))
            let scale_2d: Tile<f32, { [TILE_M, TILE_N] }> =
                broadcast_scalar(qk_scale_log2, const_shape![TILE_M, TILE_N]);
            let neg_m_ij: Tile<f32, { [TILE_M, TILE_N] }> =
                negf(m_ij.broadcast(const_shape![TILE_M, TILE_N]));
            let qk_shifted: Tile<f32, { [TILE_M, TILE_N] }> =
                fma(qk, scale_2d, neg_m_ij, rounding::NearestEven, ftz::Disabled);

            // p = exp2(qk_shifted, flush_to_zero)
            let p: Tile<f32, { [TILE_M, TILE_N] }> = exp2(qk_shifted, ftz::Enabled);

            // l_ij = sum(p, dim=1)
            let row_sum: Tile<f32, { [TILE_M] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [TILE_M, 1] }> = row_sum.reshape(const_shape![TILE_M, 1]);

            // alpha = exp2(m_i - m_ij, flush_to_zero)
            let m_diff: Tile<f32, { [TILE_M, 1] }> = m_i - m_ij;
            let alpha: Tile<f32, { [TILE_M, 1] }> = exp2(m_diff, ftz::Enabled);

            // l_i = l_i * alpha + l_ij
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            // acc = acc * alpha
            let alpha_d: Tile<f32, { [TILE_M, TILE_D] }> =
                alpha.broadcast(const_shape![TILE_M, TILE_D]);
            acc = acc * alpha_d;

            // ─── load V → [TILE_N, TILE_D] ─────────────────────────────────
            let v_ld: Tile<E, { [1, 1, TILE_N, TILE_D] }> = load_view_tko(
                &v_part,
                [batch_idx, off_kv_h, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(6i32),
                tma::Enabled,
            );
            let v_tile: Tile<E, { [TILE_N, TILE_D] }> = v_ld.reshape(const_shape![TILE_N, TILE_D]);

            // p cast to E, acc = mma(p, v, acc)
            let p_e: Tile<E, { [TILE_M, TILE_N] }> = convert_tile(p);
            acc = mmaf(p_e, v_tile, acc);

            // m_i = m_ij
            m_i = m_ij;
        }

        // ─── sink contribution to denominator ──────────────────────────────
        // sink_exp = exp2(sink_scaled - m_i, flush_to_zero)
        let sink_scaled_b: Tile<f32, { [TILE_M, 1] }> =
            broadcast_scalar(sink_scaled, const_shape![TILE_M, 1]);
        let sink_diff: Tile<f32, { [TILE_M, 1] }> = sink_scaled_b - m_i;
        let sink_exp: Tile<f32, { [TILE_M, 1] }> = exp2(sink_diff, ftz::Enabled);
        let z: Tile<f32, { [TILE_M, 1] }> = l_i + sink_exp;

        // ─── final normalize: acc / z  (divf rounding<approx> flush_to_zero) ─
        // `true_div` lowers to the approx-rounding + flush_to_zero divf the
        // reference uses; plain `divf(..., rounding::Approx, ...)` hits a known
        // cutile-rs encoding bug (Approx→full) — see softmax example.
        let z_d: Tile<f32, { [TILE_M, TILE_D] }> = z.broadcast(const_shape![TILE_M, TILE_D]);
        let acc_norm: Tile<f32, { [TILE_M, TILE_D] }> = true_div(acc, z_d);

        // cast to E, reshape to [1,1,TILE_M,TILE_D], store
        let out_e: Tile<E, { [TILE_M, TILE_D] }> = convert_tile(acc_norm);
        let out_4d: Tile<E, { [1, 1, TILE_M, TILE_D] }> =
            out_e.reshape(const_shape![1, 1, TILE_M, TILE_D]);

        let mut out_part: PartitionMut<E, { [1, 1, TILE_M, TILE_D] }> =
            unsafe { out.partition_full_mut(const_shape![1, 1, TILE_M, TILE_D]) };
        unsafe {
            store_view_tko_mut(
                &mut out_part,
                out_4d,
                [batch_idx, head_idx, bid_x, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }
    }
}
