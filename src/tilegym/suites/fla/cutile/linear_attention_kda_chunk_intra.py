# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Standalone KDA intra-chunk forward and backward helpers.

Call ``chunk_kda_fwd_intra`` and ``chunk_kda_bwd_intra`` directly from
``tilegym.suites.fla.cutile.linear_attention_kda_chunk_intra``. These helpers
have their own correctness and performance tests and do not register a
whole-attention dispatch operation.
"""

from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilegym.suites.fla.cutile._host_utils import _segmented_arange
from tilegym.suites.fla.cutile._host_utils import input_guard
from tilegym.suites.fla.cutile._tuning import tune_launch

__all__ = ["chunk_kda_fwd_intra", "chunk_kda_bwd_intra"]

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]


# Default (occ=1/nww=4) is listed first so exhaustive_search always has the
# safe baseline as a candidate; occ up to 4 and nww in {4,8} sweep the space.
_INTER_SOLVE_HINTS = ((1, 4), (2, 8), (2, 4), (3, 8), (4, 8), (4, 4), (1, 8))

# Loop-scatter kernels cap occupancy at two to avoid compilation stalls.
_RECOMPUTE_WU_HINTS = ((1, 4), (2, 8), (2, 4), (1, 8))


def _tuned_launch_hints(kernel, stream, grid, args, cache_key, hint_configs):
    choices = tuple(hint_configs)
    tune_launch(
        kernel,
        stream,
        lambda c: grid,
        lambda c: args,
        lambda: [SimpleNamespace(occupancy=o, num_worker_warps=w) for o, w in choices],
        cache_key,
        config_key=("hints", choices),
    )


# ---------------------------------------------------------------------------
# fla.utils detection (TF32 support) — host side
# ---------------------------------------------------------------------------


def _detect_tf32_supported() -> bool:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)[0] >= 8
    except BaseException:
        pass
    return False


IS_TF32_SUPPORTED = _detect_tf32_supported()


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_token_indices(cu_seqlens: torch.Tensor) -> torch.Tensor:
    lens = prepare_lens(cu_seqlens)
    seg_id, intra_idx = _segmented_arange(lens)
    return torch.stack([seg_id, intra_idx], 1).to(cu_seqlens)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _cdiv_kernel(a, b):
    # In-kernel ceil-div over compile-time constants.
    return (a + b - 1) // b


# cuTile launch args cannot be None, so flag-off optional args pass tiny dummy
# tensors. Allocating them fresh every call shows up as a per-launch
# FillFunctor; cache per (device, dtype, shape) so the steady-state launch path
# allocates nothing. Kernels never write to dummies (all stores are flag-gated).
_DUMMY_CACHE: dict = {}


def _cached_dummy(device, dtype, *shape):
    key = (device, dtype, shape)
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.zeros(*shape, device=device, dtype=dtype)
        _DUMMY_CACHE[key] = t
    return t


def _check_flat_addressing(*tensors):
    # Flat element-offset kernels compute int32 offsets over the whole
    # tensor; beyond 2**31 - 1 elements the arithmetic would wrap and,
    # with check_bounds=False, silently read/write wrong addresses.
    for t in tensors:
        if t.numel() > torch.iinfo(torch.int32).max:
            raise NotImplementedError(
                f"tensor too large for int32 flat-offset addressing ({t.numel()} elements >= 2**31); split the batch"
            )


# ======================== cuTile in-kernel helpers ========================


@ct.function
def _mm_tf32(a, b, USE_TF32: ConstBool):
    acc = ct.zeros((a.shape[0], b.shape[1]), dtype=ct.float32)
    if USE_TF32:
        a = ct.astype(a, ct.tfloat32)
        b = ct.astype(b, ct.tfloat32)
    return ct.mma(a, b, acc)


@ct.function
def _mm_offdiag(b_qg, b_kg, b_kgt, BC: ConstInt):
    lhs = ct.astype(ct.cat((b_qg, b_kg), 0), ct.tfloat32)
    rhs = ct.astype(b_kgt, ct.tfloat32)
    acc = ct.zeros((2 * BC, BC), dtype=ct.float32)
    r = ct.mma(lhs, rhs, acc)
    return ct.extract(r, index=(0, 0), shape=(BC, BC)), ct.extract(r, index=(1, 0), shape=(BC, BC))


@ct.function
def _mm_pair_rhs_tf32(lhs_a, lhs_b, rhs, BC: ConstInt, BK: ConstInt):
    """Two fp32 [BC, BC] @ [BC, BK] dots sharing one RHS as a single stacked
    [2*BC, BC] @ [BC, BK] tf32 mma (same trick as `_mm_offdiag`: lifts M from
    BC=16, which cuTile lowers to a SIMT fallback, to 32 -> real HMMA).
    Returns the two fp32 [BC, BK] products."""
    lhs = ct.astype(ct.cat((lhs_a, lhs_b), 0), ct.tfloat32)
    acc = ct.zeros((2 * BC, BK), dtype=ct.float32)
    r = ct.mma(lhs, ct.astype(rhs, ct.tfloat32), acc)
    return ct.extract(r, index=(0, 0), shape=(BC, BK)), ct.extract(r, index=(1, 0), shape=(BC, BK))


# ======================== cuTile Kernels ========================


# ---------------------------------------------------------------------------
# fla.ops.kda.chunk_intra_token_parallel
# ---------------------------------------------------------------------------


@ct.kernel
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    g,  # [BT_total, HV, K]
    beta,  # [BT_total, HV]
    Aqk,  # [BT_total, HV, BT]  (output)
    Akk,  # [BT_total, HV, BC]  (output, fp32 diagonal)
    token_indices,  # [BT_total, 2] int (dummy [1, 2] when not IS_VARLEN)
    cu_seqlens,  # [num_seq + 1] int (dummy [1] when not IS_VARLEN)
    scale: ConstFloat,
    T,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BH: ConstInt,
    BK: ConstInt,
    G: ConstInt,  # HV // H  (GVA group size)
    IS_VARLEN: ConstBool,
):
    i_tg = ct.bid(0)
    i_hg = ct.bid(1)

    if IS_VARLEN:
        i_n = ct.load(token_indices, index=(i_tg, 0), shape=()).astype(ct.int32)
        i_t = ct.load(token_indices, index=(i_tg, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
        seqlen = eos - bos
    else:
        bos = (i_tg // T) * T
        eos = bos + T
        i_t = i_tg % T
        seqlen = T

    if i_t < seqlen:
        # sub-chunk geometry
        i_c = i_t // BT
        i_s = (i_t % BT) // BC
        i_tc = i_c * BT
        i_ts = i_tc + i_s * BC

        off_hv = i_hg * BH

        q_seg = q.slice(axis=0, start=bos, stop=eos)
        k_seg = k.slice(axis=0, start=bos, stop=eos)
        g_seg = g.slice(axis=0, start=bos, stop=eos)
        beta_seg = beta.slice(axis=0, start=bos, stop=eos)
        Aqk_seg = Aqk.slice(axis=0, start=bos, stop=eos)
        Akk_seg = Akk.slice(axis=0, start=bos, stop=eos)

        o_hv = off_hv + ct.arange(BH, dtype=ct.int32)
        o_h = o_hv // G
        o_k = ct.arange(BK, dtype=ct.int32)
        m_hv = o_hv < HV
        m_k = o_k < K
        m_hk = m_hv[:, None] & m_k[None, :]

        # q/k : [BH, BK] gathered via per-(head, dim) flat index from the
        # mapped qk head index o_h. (head dim is irregularly mapped -> gather)
        hk_row = o_h[:, None]  # [BH, 1]
        k_col = o_k[None, :]  # [1, BK]
        b_q = ct.astype(
            ct.gather(q_seg, (ct.full((BH, BK), i_t, dtype=ct.int32), hk_row, k_col), mask=m_hk, padding_value=0.0),
            ct.float32,
        )
        b_k = ct.astype(
            ct.gather(k_seg, (ct.full((BH, BK), i_t, dtype=ct.int32), hk_row, k_col), mask=m_hk, padding_value=0.0),
            ct.float32,
        )

        # g : [BH, BK]  (per-HV head, regular block) and beta : [BH].
        # Gather with the HV mask (not a plain load) so padded heads [HV, BH)
        # read 0 instead of out-of-bounds when BH = next_pow2(HV) > HV.
        hv_row = o_hv[:, None]  # [BH, 1]
        b_g = ct.astype(
            ct.gather(g_seg, (ct.full((BH, BK), i_t, dtype=ct.int32), hv_row, k_col), mask=m_hk, padding_value=0.0),
            ct.float32,
        )
        b_beta = ct.astype(
            ct.gather(beta_seg, (ct.full((BH,), i_t, dtype=ct.int32), o_hv), mask=m_hv, padding_value=0.0),
            ct.float32,
        )
        b_k = b_k * b_beta[:, None]

        # inner loop over j in [i_ts, min(i_t+1, min(seqlen, i_ts+BC)))
        # static upper bound BC; runtime guard skips inactive iterations.
        for jj in range(0, BC):
            j = i_ts + jj
            if j < i_t + 1 and j < seqlen and j < i_ts + BC:
                b_kj = ct.astype(
                    ct.gather(
                        k_seg,
                        (ct.full((BH, BK), j, dtype=ct.int32), hk_row, k_col),
                        mask=m_hk,
                        padding_value=0.0,
                    ),
                    ct.float32,
                )
                b_gj = ct.astype(
                    ct.gather(
                        g_seg,
                        (ct.full((BH, BK), j, dtype=ct.int32), hv_row, k_col),
                        mask=m_hk,
                        padding_value=0.0,
                    ),
                    ct.float32,
                )
                b_kgj = ct.where(m_k[None, :], b_kj * ct.exp2(b_g - b_gj), 0.0)
                b_Aqk = ct.sum(b_q * b_kgj, axis=1) * scale
                b_Akk = ct.sum(b_k * b_kgj, axis=1) * ct.where(j < i_t, 1.0, 0.0)

                # Aqk[i_t, o_hv, j % BT] ; Akk[i_t, o_hv, j - i_ts]
                zeros_bh = ct.zeros((BH,), dtype=ct.int32)
                col_aqk = (j % BT) + zeros_bh
                col_akk = (j - i_ts) + zeros_bh
                row_idx = i_t + zeros_bh
                ct.scatter(
                    Aqk_seg,
                    (row_idx, o_hv, col_aqk),
                    ct.astype(b_Aqk, Aqk.dtype),
                    mask=m_hv,
                )
                ct.scatter(
                    Akk_seg,
                    (row_idx, o_hv, col_akk),
                    ct.astype(b_Akk, Akk.dtype),
                    mask=m_hv,
                )


# ---------------------------------------------------------------------------
# fla.ops.kda.wy_fast (recompute_w_u_fwd)
# ---------------------------------------------------------------------------


@ct.kernel
def recompute_w_u_fwd_kda_kernel(
    q,  # flat [BT_total * H * K]  (dummy when not STORE_QG)
    k,  # flat [BT_total * H * K]
    qg,  # flat [BT_total * HV * K]  (dummy when not STORE_QG)
    kg,  # flat [BT_total * HV * K]
    v,  # flat [BT_total * HV * V]
    beta,  # flat [BT_total * HV]
    w,  # flat [BT_total * HV * K]  (output)
    u,  # flat [BT_total * HV * V]  (output)
    A,  # flat [BT_total * HV * BT]
    gk,  # flat [BT_total * HV * K]
    chunk_indices,  # flat [NT * 2] int (dummy when not IS_VARLEN)
    cu_seqlens,  # flat [num_seq + 1] int (dummy when not IS_VARLEN)
    T,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    G: ConstInt,  # HV // H
    STORE_QG: ConstBool,
    STORE_KG: ConstBool,
    IS_VARLEN: ConstBool,
):
    H = HV // G
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // G

    bos = i_b * T
    T_eff = T
    if IS_VARLEN:
        i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
        i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
        bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
        eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
        T_eff = eos - bos

    k_base = (bos * H + i_h) * K
    v_base = (bos * HV + i_hv) * V
    u_base = (bos * HV + i_hv) * V
    w_base = (bos * HV + i_hv) * K
    gk_base = (bos * HV + i_hv) * K
    beta_base = bos * HV + i_hv
    A_base = (bos * HV + i_hv) * BT
    q_base = (bos * H + i_h) * K
    qg_base = (bos * HV + i_hv) * K
    kg_base = (bos * HV + i_hv) * K

    o_bt = ct.arange(BT, dtype=ct.int32)
    rows = i_t * BT + o_bt
    m_rows = rows < T_eff

    # b_b : [BT] beta strided HV
    b_b = ct.astype(
        ct.gather(beta, beta_base + rows * HV, mask=m_rows, check_bounds=False, padding_value=0.0), ct.float32
    )

    # b_A : [BT, BT]  block-ptr (T, BT) stride (HV*BT, 1)
    o_col = ct.arange(BT, dtype=ct.int32)
    A_off = A_base + ct.expand_dims(rows, 1) * (HV * BT) + ct.expand_dims(o_col, 0)
    A_mask = ct.broadcast_to(ct.expand_dims(m_rows, 1), (BT, BT))
    b_A = ct.gather(A, A_off, mask=A_mask, check_bounds=False, padding_value=0.0)

    # u = A @ (v * beta)
    for i_v in range(_cdiv_kernel(V, BV)):
        v_col = i_v * BV + ct.arange(BV, dtype=ct.int32)
        m_vcol = v_col < V
        v_off = v_base + ct.expand_dims(rows, 1) * (HV * V) + ct.expand_dims(v_col, 0)
        v_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_vcol, 0)
        b_v = ct.gather(v, v_off, mask=v_mask, check_bounds=False, padding_value=0.0)
        b_vb = ct.astype(ct.astype(b_v, ct.float32) * b_b[:, None], b_v.dtype)
        acc = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = ct.mma(b_A, b_vb, acc)
        u_off = u_base + ct.expand_dims(rows, 1) * (HV * V) + ct.expand_dims(v_col, 0)
        ct.scatter(u, u_off, ct.astype(b_u, u.dtype), mask=v_mask, check_bounds=False)

    last_idx = ct.minimum(i_t * BT + BT, T_eff) - 1

    for i_k in range(_cdiv_kernel(K, BK)):
        k_col = i_k * BK + ct.arange(BK, dtype=ct.int32)
        m_kcol = k_col < K
        kk_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_kcol, 0)

        k_off = k_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(k_col, 0)
        b_k = ct.gather(k, k_off, mask=kk_mask, check_bounds=False, padding_value=0.0)
        b_kb = ct.astype(b_k, ct.float32) * b_b[:, None]

        gk_off = gk_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
        b_gk = ct.astype(ct.gather(gk, gk_off, mask=kk_mask, check_bounds=False, padding_value=0.0), ct.float32)
        b_kb = b_kb * ct.exp2(b_gk)

        if STORE_QG:
            q_off = q_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(k_col, 0)
            b_q = ct.gather(q, q_off, mask=kk_mask, check_bounds=False, padding_value=0.0)
            b_qg = ct.astype(b_q, ct.float32) * ct.exp2(b_gk)
            qg_off = qg_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
            ct.scatter(qg, qg_off, ct.astype(b_qg, qg.dtype), mask=kk_mask, check_bounds=False)

        if STORE_KG:
            # b_gn : last-time-step gate row [BK] at element gk + last_idx*HV*K
            b_gn = ct.astype(
                ct.gather(
                    gk, gk_base + last_idx * (HV * K) + k_col, mask=m_kcol, check_bounds=False, padding_value=0.0
                ),
                ct.float32,
            )
            decay = ct.where(
                ct.expand_dims(m_rows, 1),
                ct.exp2(b_gn[None, :] - b_gk),
                ct.zeros((BT, BK), dtype=ct.float32),
            )
            b_kg = ct.astype(b_k, ct.float32) * decay
            kg_off = kg_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
            ct.scatter(kg, kg_off, ct.astype(b_kg, kg.dtype), mask=kk_mask, check_bounds=False)

        acc = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = ct.mma(b_A, ct.astype(b_kb, b_k.dtype), acc)
        w_off = w_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
        ct.scatter(w, w_off, ct.astype(b_w, w.dtype), mask=kk_mask, check_bounds=False)


# ---------------------------------------------------------------------------
# fla.ops.kda.chunk_intra  (fused inter + solve_tril)
# ---------------------------------------------------------------------------


@ct.function
def _solve_diag_block(b_Akkd, b_Ai, o_i, n_steps, BC: ConstInt):
    """In-register forward substitution on one diagonal [BC, BC] block.

    `b_Ai` starts as `-(strict-lower of A_diag)`. `b_Akkd` is the ORIGINAL
    (un-negated) diagonal [BC, BC] block already resident in registers (loaded
    once before this substitution destroys `b_Ai`). Each row `i` (>=2) is read
    with an in-register `ct.extract`. The unrolled loop uses a compile-time
    row index into the resident block.
    """
    for i in range(2, BC):
        if i < n_steps:
            # b_a = -b_Akkd[i, :]  (one row of the resident block, [BC])
            b_a = -ct.extract(b_Akkd, index=(i, 0), shape=(1, BC)).reshape((BC,))
            b_a = ct.where(o_i < i, b_a, ct.zeros((BC,), dtype=ct.float32))
            b_a = b_a + ct.sum(b_a[:, None] * b_Ai, axis=0)
            b_Ai = ct.where((o_i == i)[:, None], b_a[None, :], b_Ai)
    return b_Ai


@ct.kernel
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    g,  # [BT_total, HV, K]
    beta,  # [BT_total, HV]
    Aqk,  # [BT_total, HV, BT]  (output, off-diagonal)
    Akkd,  # [BT_total, HV, BC]  (input, fp32 diagonal Akk blocks)
    Akk,  # [BT_total, HV, BT]  (output, solved inverse)
    chunk_indices,  # [NT, 2] int (dummy when not IS_VARLEN)
    cu_seqlens,  # [num_seq + 1] int (dummy when not IS_VARLEN)
    scale: ConstFloat,
    T,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    NC: ConstInt,
    BK: ConstInt,
    G: ConstInt,  # HV // H
    IS_VARLEN: ConstBool,
    USE_SAFE_GATE: ConstBool,
    USE_TF32: ConstBool,
):
    """Fused: off-diagonal Aqk/Akk + solve_tril (I + A)^{-1}, one pass.

    Prerequisite: token_parallel (or the safe-gate sub_chunk kernel) has already
    written the diagonal Akk blocks into `Akkd` (fp32) and diagonal Aqk into Aqk.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // G

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t

    seqlen = eos - bos
    off_t = i_tt * BT

    if off_t < seqlen:
        q_seg = q.slice(axis=0, start=bos, stop=eos)
        k_seg = k.slice(axis=0, start=bos, stop=eos)
        g_seg = g.slice(axis=0, start=bos, stop=eos)
        beta_seg = beta.slice(axis=0, start=bos, stop=eos)
        Aqk_seg = Aqk.slice(axis=0, start=bos, stop=eos)
        Akk_seg = Akk.slice(axis=0, start=bos, stop=eos)
        Akkd_seg = Akkd.slice(axis=0, start=bos, stop=eos)

        # tile-space row index of sub-chunk c = i_tt * NC + c ; element start.
        SUB = BT // BC
        base_row = i_tt * SUB
        e_tc0 = off_t
        e_tc1 = e_tc0 + BC
        e_tc2 = e_tc0 + 2 * BC
        e_tc3 = e_tc0 + 3 * BC

        o_i = ct.arange(BC, dtype=ct.int32)
        m_tc1 = (e_tc1 + o_i) < seqlen
        m_tc2 = (e_tc2 + o_i) < seqlen
        m_tc3 = (e_tc3 + o_i) < seqlen

        b_Aqk10 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk10 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Aqk20 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk20 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Aqk21 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk21 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Aqk30 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk30 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Aqk31 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk31 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Aqk32 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk32 = ct.zeros((BC, BC), dtype=ct.float32)

        ##############################################################
        # off-diagonal blocks
        ##############################################################
        for i_k in range(_cdiv_kernel(K, BK)):
            o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
            m_k = o_k < K

            # row block 0 (k/g only)
            b_k0 = ct.astype(
                ct.load(k_seg, index=(base_row + 0, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
            )
            b_g0 = ct.astype(
                ct.load(g_seg, index=(base_row + 0, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
            )

            if e_tc1 < seqlen:
                b_q1 = ct.astype(
                    ct.load(q_seg, index=(base_row + 1, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                b_k1 = ct.astype(
                    ct.load(k_seg, index=(base_row + 1, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                b_g1 = ct.astype(
                    ct.load(g_seg, index=(base_row + 1, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                # b_gn1 : gate at row e_tc1 -> [BK]. Row e_tc1 is the first row of
                # block 1 (already resident as b_g1), so extract it in-register
                # instead of a DRAM gather. m_k re-masks the [K, BK) padding lanes
                # to 0 to match the gather's padding_value.
                b_gn1 = ct.where(m_k, ct.extract(b_g1, index=(0, 0), shape=(1, BK)).reshape((BK,)), 0.0)
                b_gqn = ct.where(m_tc1[:, None], ct.exp2(b_g1 - b_gn1[None, :]), 0.0)
                b_kgt = ct.transpose(b_k0 * ct.exp2(b_gn1[None, :] - b_g0))
                r_Aqk10, r_Akk10 = _mm_offdiag(b_q1 * b_gqn, b_k1 * b_gqn, b_kgt, BC)
                b_Aqk10 = b_Aqk10 + r_Aqk10
                b_Akk10 = b_Akk10 + r_Akk10

                if NC >= 3 and e_tc2 < seqlen:
                    b_q2 = ct.astype(
                        ct.load(q_seg, index=(base_row + 2, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                    )
                    b_k2 = ct.astype(
                        ct.load(k_seg, index=(base_row + 2, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                    )
                    b_g2 = ct.astype(
                        ct.load(g_seg, index=(base_row + 2, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                    )
                    # gate at row e_tc2 = first row of block 2 (resident b_g2).
                    b_gn2 = ct.where(m_k, ct.extract(b_g2, index=(0, 0), shape=(1, BK)).reshape((BK,)), 0.0)
                    b_gqn2 = ct.where(m_tc2[:, None], ct.exp2(b_g2 - b_gn2[None, :]), 0.0)
                    b_qg2 = b_q2 * b_gqn2
                    b_kg2 = b_k2 * b_gqn2
                    b_kgt = ct.transpose(b_k0 * ct.exp2(b_gn2[None, :] - b_g0))
                    r_Aqk20, r_Akk20 = _mm_offdiag(b_qg2, b_kg2, b_kgt, BC)
                    b_Aqk20 = b_Aqk20 + r_Aqk20
                    b_Akk20 = b_Akk20 + r_Akk20
                    b_kgt = ct.transpose(b_k1 * ct.exp2(b_gn2[None, :] - b_g1))
                    r_Aqk21, r_Akk21 = _mm_offdiag(b_qg2, b_kg2, b_kgt, BC)
                    b_Aqk21 = b_Aqk21 + r_Aqk21
                    b_Akk21 = b_Akk21 + r_Akk21

                    if NC >= 4 and e_tc3 < seqlen:
                        b_q3 = ct.astype(
                            ct.load(q_seg, index=(base_row + 3, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)),
                            ct.float32,
                        )
                        b_k3 = ct.astype(
                            ct.load(k_seg, index=(base_row + 3, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)),
                            ct.float32,
                        )
                        b_g3 = ct.astype(
                            ct.load(g_seg, index=(base_row + 3, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)),
                            ct.float32,
                        )
                        # gate at row e_tc3 = first row of block 3 (resident b_g3).
                        b_gn3 = ct.where(m_k, ct.extract(b_g3, index=(0, 0), shape=(1, BK)).reshape((BK,)), 0.0)
                        b_gqn3 = ct.where(m_tc3[:, None], ct.exp2(b_g3 - b_gn3[None, :]), 0.0)
                        b_qg3 = b_q3 * b_gqn3
                        b_kg3 = b_k3 * b_gqn3
                        b_kgt = ct.transpose(b_k0 * ct.exp2(b_gn3[None, :] - b_g0))
                        r_Aqk30, r_Akk30 = _mm_offdiag(b_qg3, b_kg3, b_kgt, BC)
                        b_Aqk30 = b_Aqk30 + r_Aqk30
                        b_Akk30 = b_Akk30 + r_Akk30
                        b_kgt = ct.transpose(b_k1 * ct.exp2(b_gn3[None, :] - b_g1))
                        r_Aqk31, r_Akk31 = _mm_offdiag(b_qg3, b_kg3, b_kgt, BC)
                        b_Aqk31 = b_Aqk31 + r_Aqk31
                        b_Akk31 = b_Akk31 + r_Akk31
                        b_kgt = ct.transpose(b_k2 * ct.exp2(b_gn3[None, :] - b_g2))
                        r_Aqk32, r_Akk32 = _mm_offdiag(b_qg3, b_kg3, b_kgt, BC)
                        b_Aqk32 = b_Aqk32 + r_Aqk32
                        b_Akk32 = b_Akk32 + r_Akk32

        ##############################################################
        # save off-diagonal Aqk blocks and scale Akk by beta
        ##############################################################
        if e_tc1 < seqlen:
            ct.store(
                Aqk_seg,
                index=(base_row + 1, i_hv, 0),
                tile=ct.astype(b_Aqk10 * scale, Aqk.dtype).reshape((BC, 1, BC)),
            )
            b_b1 = ct.astype(ct.load(beta_seg, index=(base_row + 1, i_hv), shape=(BC, 1)).reshape((BC,)), ct.float32)
            b_Akk10 = b_Akk10 * b_b1[:, None]
        if NC >= 3 and e_tc2 < seqlen:
            ct.store(
                Aqk_seg, index=(base_row + 2, i_hv, 0), tile=ct.astype(b_Aqk20 * scale, Aqk.dtype).reshape((BC, 1, BC))
            )
            ct.store(
                Aqk_seg, index=(base_row + 2, i_hv, 1), tile=ct.astype(b_Aqk21 * scale, Aqk.dtype).reshape((BC, 1, BC))
            )
            b_b2 = ct.astype(ct.load(beta_seg, index=(base_row + 2, i_hv), shape=(BC, 1)).reshape((BC,)), ct.float32)
            b_Akk20 = b_Akk20 * b_b2[:, None]
            b_Akk21 = b_Akk21 * b_b2[:, None]
        if NC >= 4 and e_tc3 < seqlen:
            ct.store(
                Aqk_seg, index=(base_row + 3, i_hv, 0), tile=ct.astype(b_Aqk30 * scale, Aqk.dtype).reshape((BC, 1, BC))
            )
            ct.store(
                Aqk_seg, index=(base_row + 3, i_hv, 1), tile=ct.astype(b_Aqk31 * scale, Aqk.dtype).reshape((BC, 1, BC))
            )
            ct.store(
                Aqk_seg, index=(base_row + 3, i_hv, 2), tile=ct.astype(b_Aqk32 * scale, Aqk.dtype).reshape((BC, 1, BC))
            )
            b_b3 = ct.astype(ct.load(beta_seg, index=(base_row + 3, i_hv), shape=(BC, 1)).reshape((BC,)), ct.float32)
            b_Akk30 = b_Akk30 * b_b3[:, None]
            b_Akk31 = b_Akk31 * b_b3[:, None]
            b_Akk32 = b_Akk32 * b_b3[:, None]

        ##############################################################
        # load diagonal Akk inverse blocks from Akkd (fp32)
        ##############################################################
        b_Ai00 = ct.load(Akkd_seg, index=(base_row + 0, i_hv, 0), shape=(BC, 1, BC)).reshape((BC, BC))
        b_Ai11 = ct.load(Akkd_seg, index=(base_row + 1, i_hv, 0), shape=(BC, 1, BC)).reshape((BC, BC))
        b_Ai22 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Ai33 = ct.zeros((BC, BC), dtype=ct.float32)
        if NC >= 3:
            b_Ai22 = ct.load(Akkd_seg, index=(base_row + 2, i_hv, 0), shape=(BC, 1, BC)).reshape((BC, BC))
        if NC >= 4:
            b_Ai33 = ct.load(Akkd_seg, index=(base_row + 3, i_hv, 0), shape=(BC, 1, BC)).reshape((BC, BC))

        # Keep un-negated copies of the diagonal blocks so the forward
        # substitution can read row i via in-register ct.extract instead of a
        # per-row DRAM gather (see _solve_diag_block).
        b_Akkd00 = b_Ai00
        b_Akkd11 = b_Ai11
        b_Akkd22 = b_Ai22
        b_Akkd33 = b_Ai33

        ##############################################################
        # forward substitution on diagonals (skipped when safe-gate already
        # produced the inverse diagonal blocks)
        ##############################################################
        if not USE_SAFE_GATE:
            m_A = o_i[:, None] > o_i[None, :]
            m_I = ct.astype(o_i[:, None] == o_i[None, :], ct.float32)

            b_Ai00 = -ct.where(m_A, b_Ai00, ct.zeros((BC, BC), dtype=ct.float32))
            b_Ai11 = -ct.where(m_A, b_Ai11, ct.zeros((BC, BC), dtype=ct.float32))
            if NC >= 3:
                b_Ai22 = -ct.where(m_A, b_Ai22, ct.zeros((BC, BC), dtype=ct.float32))
            if NC >= 4:
                b_Ai33 = -ct.where(m_A, b_Ai33, ct.zeros((BC, BC), dtype=ct.float32))

            n0 = seqlen - e_tc0
            b_Ai00 = _solve_diag_block(b_Akkd00, b_Ai00, o_i, n0, BC)
            n1 = seqlen - e_tc1
            b_Ai11 = _solve_diag_block(b_Akkd11, b_Ai11, o_i, n1, BC)
            if NC >= 3:
                n2 = seqlen - e_tc2
                b_Ai22 = _solve_diag_block(b_Akkd22, b_Ai22, o_i, n2, BC)
            if NC >= 4:
                n3 = seqlen - e_tc3
                b_Ai33 = _solve_diag_block(b_Akkd33, b_Ai33, o_i, n3, BC)

            b_Ai00 = b_Ai00 + m_I
            b_Ai11 = b_Ai11 + m_I
            if NC >= 3:
                b_Ai22 = b_Ai22 + m_I
            if NC >= 4:
                b_Ai33 = b_Ai33 + m_I

        ##############################################################
        # compute merged inverse using off-diagonals (tf32-precision dots)
        ##############################################################
        b_Ai10 = -_mm_tf32(_mm_tf32(b_Ai11, b_Akk10, USE_TF32), b_Ai00, USE_TF32)
        b_Ai21 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Ai20 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Ai32 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Ai31 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Ai30 = ct.zeros((BC, BC), dtype=ct.float32)
        if NC >= 3:
            b_Ai21 = -_mm_tf32(_mm_tf32(b_Ai22, b_Akk21, USE_TF32), b_Ai11, USE_TF32)
            b_Ai20 = -_mm_tf32(
                b_Ai22,
                _mm_tf32(b_Akk20, b_Ai00, USE_TF32) + _mm_tf32(b_Akk21, b_Ai10, USE_TF32),
                USE_TF32,
            )
        if NC >= 4:
            b_Ai32 = -_mm_tf32(_mm_tf32(b_Ai33, b_Akk32, USE_TF32), b_Ai22, USE_TF32)
            b_Ai31 = -_mm_tf32(
                b_Ai33,
                _mm_tf32(b_Akk31, b_Ai11, USE_TF32) + _mm_tf32(b_Akk32, b_Ai21, USE_TF32),
                USE_TF32,
            )
            b_Ai30 = -_mm_tf32(
                b_Ai33,
                _mm_tf32(b_Akk30, b_Ai00, USE_TF32)
                + _mm_tf32(b_Akk31, b_Ai10, USE_TF32)
                + _mm_tf32(b_Akk32, b_Ai20, USE_TF32),
                USE_TF32,
            )

        ##############################################################
        # store full Akk_inv to Akk
        ##############################################################
        ct.store(Akk_seg, index=(base_row + 0, i_hv, 0), tile=ct.astype(b_Ai00, Akk.dtype).reshape((BC, 1, BC)))
        ct.store(Akk_seg, index=(base_row + 1, i_hv, 0), tile=ct.astype(b_Ai10, Akk.dtype).reshape((BC, 1, BC)))
        ct.store(Akk_seg, index=(base_row + 1, i_hv, 1), tile=ct.astype(b_Ai11, Akk.dtype).reshape((BC, 1, BC)))
        if NC >= 3:
            ct.store(Akk_seg, index=(base_row + 2, i_hv, 0), tile=ct.astype(b_Ai20, Akk.dtype).reshape((BC, 1, BC)))
            ct.store(Akk_seg, index=(base_row + 2, i_hv, 1), tile=ct.astype(b_Ai21, Akk.dtype).reshape((BC, 1, BC)))
            ct.store(Akk_seg, index=(base_row + 2, i_hv, 2), tile=ct.astype(b_Ai22, Akk.dtype).reshape((BC, 1, BC)))
        if NC >= 4:
            ct.store(Akk_seg, index=(base_row + 3, i_hv, 0), tile=ct.astype(b_Ai30, Akk.dtype).reshape((BC, 1, BC)))
            ct.store(Akk_seg, index=(base_row + 3, i_hv, 1), tile=ct.astype(b_Ai31, Akk.dtype).reshape((BC, 1, BC)))
            ct.store(Akk_seg, index=(base_row + 3, i_hv, 2), tile=ct.astype(b_Ai32, Akk.dtype).reshape((BC, 1, BC)))
            ct.store(Akk_seg, index=(base_row + 3, i_hv, 3), tile=ct.astype(b_Ai33, Akk.dtype).reshape((BC, 1, BC)))


# ---------------------------------------------------------------------------
# fla.ops.kda.chunk_intra  (fused inter + solve_tril + recompute_w_u, NC == 2)
# ---------------------------------------------------------------------------


@ct.kernel
def chunk_kda_fwd_kernel_inter_solve_wu_fused(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    g,  # [BT_total, HV, K]
    beta,  # [BT_total, HV]
    Aqk,  # [BT_total, HV, BT]  (output, off-diagonal)
    Akkd,  # [BT_total, HV, BC]  (input, fp32 diagonal Akk blocks)
    # Flat element-offset form -- see the note on recompute_w_u_fwd_kda_kernel.
    qf,  # flat [BT_total * H * K]  (dummy [1] when not STORE_QG)
    kf,  # flat [BT_total * H * K]
    gf,  # flat [BT_total * HV * K]
    betaf,  # flat [BT_total * HV]
    vf,  # flat [BT_total * HV * V]
    Akkf,  # flat [BT_total * HV * BT]  (output, FULL solved tile incl. zeros)
    wf,  # flat [BT_total * HV * K]  (output)
    uf,  # flat [BT_total * HV * V]  (output)
    qgf,  # flat [BT_total * HV * K]  (output; dummy [1] when not STORE_QG)
    kgf,  # flat [BT_total * HV * K]  (output)
    chunk_indices,  # [NT, 2] int (dummy when not IS_VARLEN)
    cu_seqlens,  # [num_seq + 1] int (dummy when not IS_VARLEN)
    scale: ConstFloat,
    T,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
    BKW: ConstInt,
    BV: ConstInt,
    G: ConstInt,  # HV // H
    STORE_QG: ConstBool,
    IS_VARLEN: ConstBool,
    USE_SAFE_GATE: ConstBool,
    USE_TF32: ConstBool,
):
    """NC==2 specialization of `chunk_kda_fwd_kernel_inter_solve_fused` with
    `recompute_w_u_fwd_kda_kernel` fused onto the tail (same grid, so the
    solved A tile never round-trips DRAM between the two kernels; the second
    launch's head/tail latency and its A/gk/beta re-loads disappear).

    Numerics match the unfused pair exactly: the merged inverse is cast to
    `Akk.dtype` (the unfused path's bf16 store) BEFORE feeding the w/u mmas,
    which is precisely the tile the unfused recompute kernel would re-load.
    The FULL [BT, BT] merged tile (incl. its zero upper blocks) is stored to
    Akk, so the host can skip the `torch.zeros` fill on the Akk buffer.
    Host-gated to NC == 2 (the only tiling reachable for BT in {32, 64} with
    K >= 64, and BT == 32 in general); other configs use the unfused pair.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // G

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t

    seqlen = eos - bos
    off_t = i_tt * BT

    if off_t < seqlen:
        q_seg = q.slice(axis=0, start=bos, stop=eos)
        k_seg = k.slice(axis=0, start=bos, stop=eos)
        g_seg = g.slice(axis=0, start=bos, stop=eos)
        beta_seg = beta.slice(axis=0, start=bos, stop=eos)
        Aqk_seg = Aqk.slice(axis=0, start=bos, stop=eos)
        Akkd_seg = Akkd.slice(axis=0, start=bos, stop=eos)

        # tile-space row index of sub-chunk c = i_tt * 2 + c ; element start.
        base_row = i_tt * 2
        e_tc1 = off_t + BC

        o_i = ct.arange(BC, dtype=ct.int32)
        m_tc1 = (e_tc1 + o_i) < seqlen

        b_Aqk10 = ct.zeros((BC, BC), dtype=ct.float32)
        b_Akk10 = ct.zeros((BC, BC), dtype=ct.float32)

        ##############################################################
        # off-diagonal block (1, 0)
        ##############################################################
        for i_k in range(_cdiv_kernel(K, BK)):
            o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
            m_k = o_k < K

            b_k0 = ct.astype(
                ct.load(k_seg, index=(base_row + 0, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
            )
            b_g0 = ct.astype(
                ct.load(g_seg, index=(base_row + 0, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
            )

            if e_tc1 < seqlen:
                b_q1 = ct.astype(
                    ct.load(q_seg, index=(base_row + 1, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                b_k1 = ct.astype(
                    ct.load(k_seg, index=(base_row + 1, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                b_g1 = ct.astype(
                    ct.load(g_seg, index=(base_row + 1, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                # gate at row e_tc1 = first row of block 1 (resident b_g1).
                b_gn1 = ct.where(m_k, ct.extract(b_g1, index=(0, 0), shape=(1, BK)).reshape((BK,)), 0.0)
                b_gqn = ct.where(m_tc1[:, None], ct.exp2(b_g1 - b_gn1[None, :]), 0.0)
                b_kgt = ct.transpose(b_k0 * ct.exp2(b_gn1[None, :] - b_g0))
                r_Aqk10, r_Akk10 = _mm_offdiag(b_q1 * b_gqn, b_k1 * b_gqn, b_kgt, BC)
                b_Aqk10 = b_Aqk10 + r_Aqk10
                b_Akk10 = b_Akk10 + r_Akk10

        ##############################################################
        # save off-diagonal Aqk block and scale Akk by beta
        ##############################################################
        if e_tc1 < seqlen:
            ct.store(
                Aqk_seg,
                index=(base_row + 1, i_hv, 0),
                tile=ct.astype(b_Aqk10 * scale, Aqk.dtype).reshape((BC, 1, BC)),
            )
            b_b1 = ct.astype(ct.load(beta_seg, index=(base_row + 1, i_hv), shape=(BC, 1)).reshape((BC,)), ct.float32)
            b_Akk10 = b_Akk10 * b_b1[:, None]

        ##############################################################
        # load diagonal Akk blocks from Akkd (fp32) and solve
        ##############################################################
        b_Ai00 = ct.load(Akkd_seg, index=(base_row + 0, i_hv, 0), shape=(BC, 1, BC)).reshape((BC, BC))
        b_Ai11 = ct.load(Akkd_seg, index=(base_row + 1, i_hv, 0), shape=(BC, 1, BC)).reshape((BC, BC))

        if not USE_SAFE_GATE:
            b_Akkd00 = b_Ai00
            b_Akkd11 = b_Ai11
            m_A = o_i[:, None] > o_i[None, :]
            m_I = ct.astype(o_i[:, None] == o_i[None, :], ct.float32)
            b_Ai00 = -ct.where(m_A, b_Ai00, ct.zeros((BC, BC), dtype=ct.float32))
            b_Ai11 = -ct.where(m_A, b_Ai11, ct.zeros((BC, BC), dtype=ct.float32))
            b_Ai00 = _solve_diag_block(b_Akkd00, b_Ai00, o_i, seqlen - off_t, BC)
            b_Ai11 = _solve_diag_block(b_Akkd11, b_Ai11, o_i, seqlen - e_tc1, BC)
            b_Ai00 = b_Ai00 + m_I
            b_Ai11 = b_Ai11 + m_I

        # merged inverse block (1, 0)
        b_Ai10 = -_mm_tf32(_mm_tf32(b_Ai11, b_Akk10, USE_TF32), b_Ai00, USE_TF32)

        ##############################################################
        # assemble + store the FULL merged inverse tile (zeros upper block),
        # then keep the low-precision cast resident for the w/u mmas (this is
        # bit-identical to what the unfused recompute kernel re-loads).
        ##############################################################
        b_zero = ct.zeros((BC, BC), dtype=ct.float32)
        b_Afull = ct.cat((ct.cat((b_Ai00, b_zero), 1), ct.cat((b_Ai10, b_Ai11), 1)), 0)
        b_Alp = ct.astype(b_Afull, Akkf.dtype)

        ##############################################################
        # fused recompute_w_u tail: u = A @ (v * beta),
        # w = A @ (k * beta * 2^gk), kg = k * 2^(gn_last - gk), qg = q * 2^gk.
        # Flat element-offset gather/scatter form, ported verbatim from
        # recompute_w_u_fwd_kda_kernel (with the A round-trip removed).
        ##############################################################
        H = HV // G
        k_base = (bos * H + i_h) * K
        v_base = (bos * HV + i_hv) * V
        u_base = (bos * HV + i_hv) * V
        w_base = (bos * HV + i_hv) * K
        gk_base = (bos * HV + i_hv) * K
        beta_base = bos * HV + i_hv
        A_base = (bos * HV + i_hv) * BT
        q_base = (bos * H + i_h) * K
        qg_base = (bos * HV + i_hv) * K
        kg_base = (bos * HV + i_hv) * K

        o_bt = ct.arange(BT, dtype=ct.int32)
        rows = off_t + o_bt
        m_rows = rows < seqlen

        # store the full solved tile (T, BT) stride (HV*BT, 1)
        A_off = A_base + ct.expand_dims(rows, 1) * (HV * BT) + ct.expand_dims(o_bt, 0)
        A_mask = ct.broadcast_to(ct.expand_dims(m_rows, 1), (BT, BT))
        ct.scatter(Akkf, A_off, b_Alp, mask=A_mask, check_bounds=False)

        # b_b : [BT] beta strided HV
        b_bb = ct.astype(
            ct.gather(betaf, beta_base + rows * HV, mask=m_rows, check_bounds=False, padding_value=0.0), ct.float32
        )

        # u = A @ (v * beta)
        for i_v in range(_cdiv_kernel(V, BV)):
            v_col = i_v * BV + ct.arange(BV, dtype=ct.int32)
            m_vcol = v_col < V
            v_off = v_base + ct.expand_dims(rows, 1) * (HV * V) + ct.expand_dims(v_col, 0)
            v_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_vcol, 0)
            b_v = ct.gather(vf, v_off, mask=v_mask, check_bounds=False, padding_value=0.0)
            b_vb = ct.astype(ct.astype(b_v, ct.float32) * b_bb[:, None], b_v.dtype)
            b_u = ct.mma(b_Alp, b_vb, ct.zeros((BT, BV), dtype=ct.float32))
            u_off = u_base + ct.expand_dims(rows, 1) * (HV * V) + ct.expand_dims(v_col, 0)
            ct.scatter(uf, u_off, ct.astype(b_u, uf.dtype), mask=v_mask, check_bounds=False)

        last_idx = ct.minimum(off_t + BT, seqlen) - 1

        for i_kw in range(_cdiv_kernel(K, BKW)):
            k_col = i_kw * BKW + ct.arange(BKW, dtype=ct.int32)
            m_kcol = k_col < K
            kk_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_kcol, 0)

            k_off = k_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(k_col, 0)
            b_kk = ct.gather(kf, k_off, mask=kk_mask, check_bounds=False, padding_value=0.0)
            b_kb = ct.astype(b_kk, ct.float32) * b_bb[:, None]

            gk_off = gk_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
            b_gk = ct.astype(ct.gather(gf, gk_off, mask=kk_mask, check_bounds=False, padding_value=0.0), ct.float32)
            b_kb = b_kb * ct.exp2(b_gk)

            if STORE_QG:
                q_off = q_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(k_col, 0)
                b_qq = ct.gather(qf, q_off, mask=kk_mask, check_bounds=False, padding_value=0.0)
                b_qg = ct.astype(b_qq, ct.float32) * ct.exp2(b_gk)
                qg_off = qg_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
                ct.scatter(qgf, qg_off, ct.astype(b_qg, qgf.dtype), mask=kk_mask, check_bounds=False)

            # b_gn : last-time-step gate row [BKW] at element gk + last_idx*HV*K
            b_gn = ct.astype(
                ct.gather(
                    gf, gk_base + last_idx * (HV * K) + k_col, mask=m_kcol, check_bounds=False, padding_value=0.0
                ),
                ct.float32,
            )
            decay = ct.where(
                ct.expand_dims(m_rows, 1),
                ct.exp2(b_gn[None, :] - b_gk),
                ct.zeros((BT, BKW), dtype=ct.float32),
            )
            b_kg = ct.astype(b_kk, ct.float32) * decay
            kg_off = kg_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
            ct.scatter(kgf, kg_off, ct.astype(b_kg, kgf.dtype), mask=kk_mask, check_bounds=False)

            b_w = ct.mma(b_Alp, ct.astype(b_kb, b_kk.dtype), ct.zeros((BT, BKW), dtype=ct.float32))
            w_off = w_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(k_col, 0)
            ct.scatter(wf, w_off, ct.astype(b_w, wf.dtype), mask=kk_mask, check_bounds=False)


# ---------------------------------------------------------------------------
# fla.ops.kda.chunk_intra  (single fully-fused forward kernel:
#   diagonal sub-chunk + inter + solve_tril + recompute_w_u; BC == 32, K <= 64)
# ---------------------------------------------------------------------------


@ct.kernel
def chunk_kda_fwd_kernel_intra_wu_fused_full(
    qf,  # flat [BT_total * H * K]
    kf,  # flat [BT_total * H * K]
    gf,  # flat [BT_total * HV * K]
    betaf,  # flat [BT_total * HV]
    vf,  # flat [BT_total * HV * V]
    Aqkf,  # flat [BT_total * HV * BT]  (output: diag + off-diag blocks)
    Akkf,  # flat [BT_total * HV * BT]  (output, FULL solved tile incl. zeros)
    wf,  # flat [BT_total * HV * K]  (output)
    uf,  # flat [BT_total * HV * V]  (output)
    qgf,  # flat [BT_total * HV * K]  (output; dummy [1] when not STORE_QG)
    kgf,  # flat [BT_total * HV * K]  (output)
    chunk_indices,  # flat [NT * 2] int (dummy when not IS_VARLEN)
    cu_seqlens,  # flat [num_seq + 1] int (dummy when not IS_VARLEN)
    scale: ConstFloat,
    T,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    G: ConstInt,  # HV // H
    STORE_QG: ConstBool,
    IS_VARLEN: ConstBool,
    USE_TF32: ConstBool,
):
    """One kernel for the whole KDA intra-chunk forward at the NC==2 / BC==32
    / K<=64 tiling:

      1. diagonal Aqk/Akk sub-chunk blocks (mid-pivot factorized gates +
         Neumann-by-squaring solve -- verbatim `..._intra_sub_chunk` math),
      2. off-diagonal Aqk/Akk block (1, 0),
      3. merged (I + Akk)^{-1},
      4. recompute_w_u tail (w, u, kg, qg).

    vs the 3-kernel chain this removes two kernel launches, the Akkd
    (diagonal) and A (solved-tile) DRAM round trips, and the triple q/k/g
    re-loads: every operand is gathered ONCE up front (flat element-offset
    form; independent gathers overlap) and every block view is an in-register
    `ct.extract`. Grid (NT, B*HV); pure per-CTA latency at small grids.
    Outputs are bit-compatible with the unfused chain (the diagonal inverse
    round-tripped through Akkd in fp32, i.e. losslessly; the merged tile is
    cast to Akk's dtype before the w/u mmas exactly like the unfused
    recompute kernel's re-load).
    """
    H = HV // G
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // G

    bos = i_b * T
    T_eff = T
    if IS_VARLEN:
        i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
        i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
        bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
        eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
        T_eff = eos - bos

    off_t = i_t * BT
    if off_t < T_eff:
        q_base = (bos * H + i_h) * K
        k_base = (bos * H + i_h) * K
        g_base = (bos * HV + i_hv) * K
        beta_base = bos * HV + i_hv
        v_base = (bos * HV + i_hv) * V
        Aqk_base = (bos * HV + i_hv) * BT
        Akk_base = (bos * HV + i_hv) * BT
        w_base = (bos * HV + i_hv) * K
        u_base = (bos * HV + i_hv) * V
        qg_base = (bos * HV + i_hv) * K
        kg_base = (bos * HV + i_hv) * K

        o_bt = ct.arange(BT, dtype=ct.int32)
        rows = off_t + o_bt
        m_rows = rows < T_eff
        o_k = ct.arange(BK, dtype=ct.int32)
        m_k = o_k < K
        bt_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_k, 0)

        e_tc1 = off_t + BC
        o_i = ct.arange(BC, dtype=ct.int32)
        m_r0 = (off_t + o_i) < T_eff
        m_r1 = (e_tc1 + o_i) < T_eff

        rows0 = off_t + o_i
        rows1 = e_tc1 + o_i
        qk_mask0 = ct.expand_dims(m_r0, 1) & ct.expand_dims(m_k, 0)
        qk_mask1 = ct.expand_dims(m_r1, 1) & ct.expand_dims(m_k, 0)
        q_off0 = q_base + ct.expand_dims(rows0, 1) * (H * K) + ct.expand_dims(o_k, 0)
        q_off1 = q_base + ct.expand_dims(rows1, 1) * (H * K) + ct.expand_dims(o_k, 0)
        k_off0 = k_base + ct.expand_dims(rows0, 1) * (H * K) + ct.expand_dims(o_k, 0)
        k_off1 = k_base + ct.expand_dims(rows1, 1) * (H * K) + ct.expand_dims(o_k, 0)
        g_off0 = g_base + ct.expand_dims(rows0, 1) * (HV * K) + ct.expand_dims(o_k, 0)
        g_off1 = g_base + ct.expand_dims(rows1, 1) * (HV * K) + ct.expand_dims(o_k, 0)
        b_q0 = ct.astype(ct.gather(qf, q_off0, mask=qk_mask0, check_bounds=False, padding_value=0.0), ct.float32)
        b_q1 = ct.astype(ct.gather(qf, q_off1, mask=qk_mask1, check_bounds=False, padding_value=0.0), ct.float32)
        b_k0 = ct.astype(ct.gather(kf, k_off0, mask=qk_mask0, check_bounds=False, padding_value=0.0), ct.float32)
        b_k1 = ct.astype(ct.gather(kf, k_off1, mask=qk_mask1, check_bounds=False, padding_value=0.0), ct.float32)
        b_g0 = ct.astype(ct.gather(gf, g_off0, mask=qk_mask0, check_bounds=False, padding_value=0.0), ct.float32)
        b_g1 = ct.astype(ct.gather(gf, g_off1, mask=qk_mask1, check_bounds=False, padding_value=0.0), ct.float32)
        b_bb = ct.astype(
            ct.gather(betaf, beta_base + rows * HV, mask=m_rows, check_bounds=False, padding_value=0.0), ct.float32
        )
        # diagonal mid-pivot gate rows (numerically-stable midpoint; the row
        # index is runtime so these stay gathers)
        gn_row0 = off_t + ct.minimum(BC // 2, T_eff - off_t - 1)
        # e_tc1 >= T_eff makes this T_eff - 1 (block 1 fully masked below).
        gn_row1 = e_tc1 + ct.minimum(BC // 2, T_eff - e_tc1 - 1)
        b_gnd0 = ct.astype(
            ct.gather(gf, g_base + gn_row0 * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_gnd1 = ct.astype(
            ct.gather(gf, g_base + gn_row1 * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        # last-chunk-row gate (kg decay pivot)
        last_idx = ct.minimum(off_t + BT, T_eff) - 1
        b_gn_last = ct.astype(
            ct.gather(gf, g_base + last_idx * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
            ct.float32,
        )

        # in-register block views
        b_b0 = ct.extract(b_bb, index=(0,), shape=(BC,))
        b_b1 = ct.extract(b_bb, index=(1,), shape=(BC,))

        m_Aqk = o_i[:, None] >= o_i[None, :]
        m_Akk = o_i[:, None] > o_i[None, :]
        m_I = ct.astype(o_i[:, None] == o_i[None, :], ct.float32)

        ##############################################################
        # diagonal blocks (verbatim intra_sub_chunk math, both sub-chunks)
        ##############################################################
        b_gm0 = b_g0 - b_gnd0[None, :]
        b_gq0 = ct.where(m_r0[:, None], ct.exp2(b_gm0), 0.0)
        b_gk0 = ct.where(m_r0[:, None], ct.exp2(-b_gm0), 0.0)
        b_kgt0 = ct.transpose(b_k0 * b_gk0)
        r_Aqk0, r_Akk0 = _mm_offdiag(b_q0 * b_gq0, b_k0 * b_gq0, b_kgt0, BC)
        b_Aqkd0 = ct.where(m_Aqk, r_Aqk0 * scale, ct.zeros((BC, BC), dtype=ct.float32))
        b_Akkd0 = ct.where(m_Akk, r_Akk0 * b_b0[:, None], ct.zeros((BC, BC), dtype=ct.float32))

        b_gm1 = b_g1 - b_gnd1[None, :]
        b_gq1 = ct.where(m_r1[:, None], ct.exp2(b_gm1), 0.0)
        b_gk1 = ct.where(m_r1[:, None], ct.exp2(-b_gm1), 0.0)
        b_kgt1 = ct.transpose(b_k1 * b_gk1)
        r_Aqk1, r_Akk1 = _mm_offdiag(b_q1 * b_gq1, b_k1 * b_gq1, b_kgt1, BC)
        b_Aqkd1 = ct.where(m_Aqk, r_Aqk1 * scale, ct.zeros((BC, BC), dtype=ct.float32))
        b_Akkd1 = ct.where(m_Akk, r_Akk1 * b_b1[:, None], ct.zeros((BC, BC), dtype=ct.float32))

        # store diagonal Aqk blocks (rows masked; cols = own diagonal block)
        aqk_mask0 = ct.broadcast_to(ct.expand_dims(m_r0, 1), (BC, BC))
        aqk_mask1 = ct.broadcast_to(ct.expand_dims(m_r1, 1), (BC, BC))
        aq_off0 = Aqk_base + ct.expand_dims(rows0, 1) * (HV * BT) + ct.expand_dims(o_i, 0)
        aq_off1 = Aqk_base + ct.expand_dims(rows1, 1) * (HV * BT) + ct.expand_dims(BC + o_i, 0)
        ct.scatter(Aqkf, aq_off0, ct.astype(b_Aqkd0, Aqkf.dtype), mask=aqk_mask0, check_bounds=False)
        ct.scatter(Aqkf, aq_off1, ct.astype(b_Aqkd1, Aqkf.dtype), mask=aqk_mask1, check_bounds=False)

        # Neumann-series-by-squaring inverse of (I + strict-lower Akk) for both
        # diagonal blocks (independent chains; see intra_sub_chunk derivation).
        b_N0 = -b_Akkd0
        b_Ai00 = m_I + b_N0
        b_N1 = -b_Akkd1
        b_Ai11 = m_I + b_N1
        for ii in range(2, BC):
            if (ii & (ii - 1)) == 0:  # ii is a power of two (host-time test)
                b_N0 = _mm_tf32(b_N0, b_N0, True)
                b_Ai00 = b_Ai00 + _mm_tf32(b_Ai00, b_N0, True)
                b_N1 = _mm_tf32(b_N1, b_N1, True)
                b_Ai11 = b_Ai11 + _mm_tf32(b_Ai11, b_N1, True)

        ##############################################################
        # off-diagonal block (1, 0)  (verbatim inter_solve math)
        ##############################################################
        # gate at row e_tc1 = first row of block 1 (resident b_g1; masked
        # gather already zeroed the [K, BK) padding lanes).
        b_gn1 = ct.extract(b_g1, index=(0, 0), shape=(1, BK)).reshape((BK,))
        b_gqn = ct.where(m_r1[:, None], ct.exp2(b_g1 - b_gn1[None, :]), 0.0)
        b_kgt = ct.transpose(b_k0 * ct.exp2(b_gn1[None, :] - b_g0))
        b_Aqk10, b_Akk10 = _mm_offdiag(b_q1 * b_gqn, b_k1 * b_gqn, b_kgt, BC)

        if e_tc1 < T_eff:
            aq_off10 = Aqk_base + ct.expand_dims(rows1, 1) * (HV * BT) + ct.expand_dims(o_i, 0)
            ct.scatter(Aqkf, aq_off10, ct.astype(b_Aqk10 * scale, Aqkf.dtype), mask=aqk_mask1, check_bounds=False)
        b_Akk10 = b_Akk10 * b_b1[:, None]

        # merged inverse block (1, 0)
        b_Ai10 = -_mm_tf32(_mm_tf32(b_Ai11, b_Akk10, USE_TF32), b_Ai00, USE_TF32)

        ##############################################################
        # assemble + store the FULL merged inverse tile (zeros upper block)
        ##############################################################
        b_zero = ct.zeros((BC, BC), dtype=ct.float32)
        b_Afull = ct.cat((ct.cat((b_Ai00, b_zero), 1), ct.cat((b_Ai10, b_Ai11), 1)), 0)
        b_Alp = ct.astype(b_Afull, Akkf.dtype)
        A_off = Akk_base + ct.expand_dims(rows, 1) * (HV * BT) + ct.expand_dims(o_bt, 0)
        A_mask = ct.broadcast_to(ct.expand_dims(m_rows, 1), (BT, BT))
        ct.scatter(Akkf, A_off, b_Alp, mask=A_mask, check_bounds=False)

        ##############################################################
        # recompute_w_u tail (operands already resident)
        ##############################################################
        for i_v in range(_cdiv_kernel(V, BV)):
            v_col = i_v * BV + ct.arange(BV, dtype=ct.int32)
            m_vcol = v_col < V
            v_off = v_base + ct.expand_dims(rows, 1) * (HV * V) + ct.expand_dims(v_col, 0)
            v_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_vcol, 0)
            b_v = ct.gather(vf, v_off, mask=v_mask, check_bounds=False, padding_value=0.0)
            b_vb = ct.astype(ct.astype(b_v, ct.float32) * b_bb[:, None], b_v.dtype)
            b_u = ct.mma(b_Alp, b_vb, ct.zeros((BT, BV), dtype=ct.float32))
            u_off = u_base + ct.expand_dims(rows, 1) * (HV * V) + ct.expand_dims(v_col, 0)
            ct.scatter(uf, u_off, ct.astype(b_u, uf.dtype), mask=v_mask, check_bounds=False)

        # re-gather k/g for the w/kg/qg tail (L2-hot; cheaper than keeping the
        # [BT, BK] tiles live across the solve -- see the gather-section note)
        k_off = k_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(o_k, 0)
        g_off = g_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(o_k, 0)
        b_kt = ct.gather(kf, k_off, mask=bt_mask, check_bounds=False, padding_value=0.0)
        b_gt = ct.astype(ct.gather(gf, g_off, mask=bt_mask, check_bounds=False, padding_value=0.0), ct.float32)

        b_kb = ct.astype(b_kt, ct.float32) * b_bb[:, None]
        b_kb = b_kb * ct.exp2(b_gt)

        if STORE_QG:
            q_offt = q_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(o_k, 0)
            b_qt = ct.gather(qf, q_offt, mask=bt_mask, check_bounds=False, padding_value=0.0)
            b_qg = ct.astype(b_qt, ct.float32) * ct.exp2(b_gt)
            qg_off = qg_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(o_k, 0)
            ct.scatter(qgf, qg_off, ct.astype(b_qg, qgf.dtype), mask=bt_mask, check_bounds=False)

        decay = ct.where(
            ct.expand_dims(m_rows, 1),
            ct.exp2(b_gn_last[None, :] - b_gt),
            ct.zeros((BT, BK), dtype=ct.float32),
        )
        b_kg = ct.astype(b_kt, ct.float32) * decay
        kg_off = kg_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(o_k, 0)
        ct.scatter(kgf, kg_off, ct.astype(b_kg, kgf.dtype), mask=bt_mask, check_bounds=False)

        b_w = ct.mma(b_Alp, ct.astype(b_kb, b_kt.dtype), ct.zeros((BT, BK), dtype=ct.float32))
        w_off = w_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(o_k, 0)
        ct.scatter(wf, w_off, ct.astype(b_w, wf.dtype), mask=bt_mask, check_bounds=False)


# ---------------------------------------------------------------------------
# fla.ops.kda.chunk_intra  (safe-gate diagonal sub-chunk kernel)
# ---------------------------------------------------------------------------


@ct.kernel
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,  # flat [BT_total * H * K]
    k,  # flat [BT_total * H * K]
    g,  # flat [BT_total * HV * K]
    beta,  # flat [BT_total * HV]
    Aqk,  # flat [BT_total * HV * BT]  (output, diagonal)
    Akk,  # flat [BT_total * HV * BC]  (output, fp32 diagonal inverse)
    chunk_indices,  # flat [NT * 2] int (dummy when not IS_VARLEN)
    cu_seqlens,  # flat [num_seq + 1] int (dummy when not IS_VARLEN)
    scale: ConstFloat,
    T,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
    G: ConstInt,  # HV // H
    IS_VARLEN: ConstBool,
):
    # Flat element-offset form -- see the note on recompute_w_u_fwd_kda_kernel.
    H = HV // G
    i_t = ct.bid(0)
    i_i = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // G

    bos = i_b * T
    T_eff = T
    if IS_VARLEN:
        i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
        i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
        bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
        eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
        T_eff = eos - bos

    off_t = i_t * BT
    off_i = i_i * BC
    i_ti = off_t + off_i  # segment-space start row of this sub-chunk

    if i_ti < T_eff:
        q_base = (bos * H + i_h) * K
        k_base = (bos * H + i_h) * K
        g_base = (bos * HV + i_hv) * K
        beta_base = bos * HV + i_hv
        Aqk_base = (bos * HV + i_hv) * BT
        Akk_base = (bos * HV + i_hv) * BC

        o_k = ct.arange(BK, dtype=ct.int32)
        m_k = o_k < K
        rows = i_ti + ct.arange(BC, dtype=ct.int32)
        m_rows = rows < T_eff
        qk_mask = ct.expand_dims(m_rows, 1) & ct.expand_dims(m_k, 0)

        q_off = q_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(o_k, 0)
        k_off = k_base + ct.expand_dims(rows, 1) * (H * K) + ct.expand_dims(o_k, 0)
        g_off = g_base + ct.expand_dims(rows, 1) * (HV * K) + ct.expand_dims(o_k, 0)
        b_q = ct.gather(q, q_off, mask=qk_mask, check_bounds=False, padding_value=0.0)
        b_k = ct.gather(k, k_off, mask=qk_mask, check_bounds=False, padding_value=0.0)
        b_g = ct.astype(ct.gather(g, g_off, mask=qk_mask, check_bounds=False, padding_value=0.0), ct.float32)
        b_beta = ct.gather(beta, beta_base + rows * HV, mask=m_rows, check_bounds=False, padding_value=0.0)

        # b_gn : gate at the numerically-stable midpoint row -> [1, BK]
        gn_row = i_ti + ct.minimum(BC // 2, T_eff - i_ti - 1)
        b_gn = ct.astype(
            ct.gather(g, g_base + gn_row * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
            ct.float32,
        )[None, :]

        b_gm = b_g - b_gn
        b_gq = ct.where(m_rows[:, None], ct.exp2(b_gm), 0.0)
        b_gk = ct.where(m_rows[:, None], ct.exp2(-b_gm), 0.0)

        b_kgt = ct.transpose(ct.astype(b_k, ct.float32) * b_gk)
        r_Aqk, r_Akk = _mm_offdiag(ct.astype(b_q, ct.float32) * b_gq, ct.astype(b_k, ct.float32) * b_gq, b_kgt, BC)
        b_Aqk = r_Aqk * scale
        b_Akk = r_Akk * ct.astype(b_beta, ct.float32)[:, None]

        o_i = ct.arange(BC, dtype=ct.int32)
        m_Aqk = o_i[:, None] >= o_i[None, :]
        m_Akk = o_i[:, None] > o_i[None, :]
        m_I = ct.astype(o_i[:, None] == o_i[None, :], ct.float32)

        b_Aqk = ct.where(m_Aqk, b_Aqk, ct.zeros((BC, BC), dtype=ct.float32))
        b_Akk = ct.where(m_Akk, b_Akk, ct.zeros((BC, BC), dtype=ct.float32))

        # store Aqk block (T, BT) stride (HV*BT, 1) at (i_ti, off_i)
        aq_off = Aqk_base + ct.expand_dims(rows, 1) * (HV * BT) + ct.expand_dims(off_i + o_i, 0)
        aq_mask = ct.broadcast_to(ct.expand_dims(m_rows, 1), (BC, BC))
        ct.scatter(Aqk, aq_off, ct.astype(b_Aqk, Aqk.dtype), mask=aq_mask, check_bounds=False)

        # Akk block offsets (T, BC) stride (HV*BC, 1) at (i_ti, 0); only the
        # SOLVED block is scattered below (downstream reads Akkd inverses only).
        ak_off = Akk_base + ct.expand_dims(rows, 1) * (HV * BC) + ct.expand_dims(o_i, 0)
        ak_mask = ct.broadcast_to(ct.expand_dims(m_rows, 1), (BC, BC))

        b_N = -b_Akk
        b_Ai = m_I + b_N  # (I + N)
        for ii in range(2, BC):
            if (ii & (ii - 1)) == 0:  # ii is a power of two (host-time test)
                b_N = _mm_tf32(b_N, b_N, True)  # N -> N^2 -> N^4 ...
                b_Ai = b_Ai + _mm_tf32(b_Ai, b_N, True)  # (prod so far) @ (I + N^(2^j))
        ct.scatter(Akk, ak_off, ct.astype(b_Ai, Akk.dtype), mask=ak_mask, check_bounds=False)


# ---------------------------------------------------------------------------
# fla.ops.kda.chunk_intra  (intra-chunk backward)
# ---------------------------------------------------------------------------


@ct.kernel
def chunk_kda_bwd_kernel_intra(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    g,  # [BT_total, HV, K]
    beta,  # [BT_total, HV]
    dAqk,  # [BT_total, HV, BT]
    dAkk,  # [BT_total, HV, BT]
    dq,  # [BT_total, HV, K]
    dq2,  # [BT_total, HV, K]  (output)
    dk,  # [BT_total, HV, K]
    dk2,  # [BT_total, HV, K]  (output)
    dg,  # [BT_total, HV, K]
    dg2,  # [BT_total, HV, K]  (output, fp32)
    db,  # [NK * BT_total, HV]  (output, fp32)
    chunk_indices,  # [NT, 2] int (dummy when not IS_VARLEN)
    cu_seqlens,  # [num_seq + 1] int (dummy when not IS_VARLEN)
    B,
    T,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
    NC: ConstInt,
    G: ConstInt,  # HV // H
    IS_VARLEN: ConstBool,
    SAFE_GATE: ConstBool,
):
    i_kc = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // G
    i_k = i_kc // NC
    i_i = i_kc % NC

    all_tok = B * T
    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t

    seqlen = eos - bos
    off_t = i_tt * BT
    off_i = i_i * BC
    e_ti = off_t + off_i  # element-space start row of this sub-chunk
    SUB = BT // BC
    row_ti = i_tt * SUB + i_i

    if e_ti < seqlen:
        o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
        m_k = o_k < K

        q_seg = q.slice(axis=0, start=bos, stop=eos)
        k_seg = k.slice(axis=0, start=bos, stop=eos)
        g_seg = g.slice(axis=0, start=bos, stop=eos)
        beta_seg = beta.slice(axis=0, start=bos, stop=eos)
        dAqk_seg = dAqk.slice(axis=0, start=bos, stop=eos)
        dAkk_seg = dAkk.slice(axis=0, start=bos, stop=eos)
        dq_seg = dq.slice(axis=0, start=bos, stop=eos)
        dq2_seg = dq2.slice(axis=0, start=bos, stop=eos)
        dk_seg = dk.slice(axis=0, start=bos, stop=eos)
        dk2_seg = dk2.slice(axis=0, start=bos, stop=eos)
        dg_seg = dg.slice(axis=0, start=bos, stop=eos)
        dg2_seg = dg2.slice(axis=0, start=bos, stop=eos)

        b_g = ct.astype(ct.load(g_seg, index=(row_ti, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32)
        b_b = ct.load(beta_seg, index=(row_ti, i_hv), shape=(BC, 1)).reshape((BC,))

        o_i = ct.arange(BC, dtype=ct.int32)

        # ---------------- dq2 / dk2: contributions from earlier sub-chunks ----
        b_dq2 = ct.zeros((BC, BK), dtype=ct.float32)
        b_dk2 = ct.zeros((BC, BK), dtype=ct.float32)
        if i_i > 0:
            # b_gn : gate at this sub-chunk's first row -> [BK]
            b_gn = ct.astype(
                ct.gather(
                    g_seg,
                    (ct.full((BK,), e_ti, dtype=ct.int32), ct.full((BK,), i_hv, dtype=ct.int32), o_k),
                    mask=m_k,
                    padding_value=0.0,
                ),
                ct.float32,
            )[None, :]
            for i_j in range(0, NC):
                if i_j < i_i:
                    jrow = i_tt * SUB + i_j  # tile-space row of sub-chunk j
                    b_k = ct.astype(
                        ct.load(k_seg, index=(jrow, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                    )
                    b_gk = ct.astype(
                        ct.load(g_seg, index=(jrow, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                    )
                    # dAqk / dAkk : [BC, BC] at (row=row_ti, col block i_j)
                    b_dAqk = ct.astype(
                        ct.load(dAqk_seg, index=(row_ti, i_hv, i_j), shape=(BC, 1, BC)).reshape((BC, BC)), ct.float32
                    )
                    b_dAkk = ct.astype(
                        ct.load(dAkk_seg, index=(row_ti, i_hv, i_j), shape=(BC, 1, BC)).reshape((BC, BC)), ct.float32
                    )
                    b_kg = b_k * ct.exp2(b_gn - b_gk)
                    p_dq2, p_dk2 = _mm_pair_rhs_tf32(b_dAqk, b_dAkk, b_kg, BC, BK)
                    b_dq2 = b_dq2 + p_dq2
                    b_dk2 = b_dk2 + p_dk2
            b_gqn = ct.exp2(b_g - b_gn)
            b_dq2 = b_dq2 * b_gqn
            b_dk2 = b_dk2 * b_gqn

        # ---------------- diagonal contribution to dq2 / dk2 -----------------
        b_q = ct.load(q_seg, index=(row_ti, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK))
        b_k = ct.load(k_seg, index=(row_ti, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK))

        mid = ct.minimum(BC // 2, seqlen - e_ti - 1)
        b_gn_d = ct.astype(
            ct.gather(
                g_seg,
                (ct.full((BK,), e_ti, dtype=ct.int32) + mid, ct.full((BK,), i_hv, dtype=ct.int32), o_k),
                mask=m_k,
                padding_value=0.0,
            ),
            ct.float32,
        )[None, :]

        b_dAqk_diag = ct.astype(
            ct.load(
                dAqk_seg,
                index=(row_ti, i_hv, i_i),
                shape=(BC, 1, BC),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BC, BC)),
            ct.float32,
        )
        b_dAkk_diag = ct.astype(
            ct.load(
                dAkk_seg,
                index=(row_ti, i_hv, i_i),
                shape=(BC, 1, BC),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BC, BC)),
            ct.float32,
        )

        m_i_diag_qk = (
            (o_i[:, None] >= o_i[None, :]) & ((e_ti + o_i[:, None]) < seqlen) & ((e_ti + o_i[None, :]) < seqlen)
        )
        m_j_diag = (e_ti + o_i[:, None]) < seqlen

        b_dAqk_qk = ct.where(m_i_diag_qk, b_dAqk_diag, ct.zeros((BC, BC), dtype=ct.float32))
        b_dAkk_qk = ct.where(m_i_diag_qk, b_dAkk_diag, ct.zeros((BC, BC), dtype=ct.float32))
        b_g_diag = ct.where(m_j_diag, b_g - b_gn_d, 0.0)
        exp_b_g_diag = ct.where(m_j_diag, ct.exp2(b_g_diag), 0.0)
        exp_neg_b_g_diag = ct.where(m_j_diag, ct.exp2(-b_g_diag), 0.0)

        b_k_exp_diag = ct.astype(b_k, ct.float32) * exp_neg_b_g_diag
        # Stacked-pair tf32 mma (M=2*BC=32 -> HMMA), shared RHS b_k_exp_diag.
        p_dq2d, p_dk2d = _mm_pair_rhs_tf32(b_dAqk_qk, b_dAkk_qk, b_k_exp_diag, BC, BK)
        b_dq2 = b_dq2 + p_dq2d * exp_b_g_diag
        b_dk2 = b_dk2 + p_dk2d * exp_b_g_diag

        b_db = ct.sum(b_dk2 * ct.astype(b_k, ct.float32), axis=1)
        b_dk2 = b_dk2 * b_b[:, None]

        # db[(i_k*all_tok + e_ti) .. , i_hv] -> store the [BC] vector
        db_seg = db.slice(axis=0, start=i_k * all_tok + bos, stop=i_k * all_tok + eos)
        ct.store(db_seg, index=(row_ti, i_hv), tile=ct.astype(b_db, db.dtype).reshape((BC, 1)))

        b_dg2 = ct.astype(b_q, ct.float32) * b_dq2
        # dq2 = b_dq2 + dq
        b_dq_old = ct.astype(
            ct.load(dq_seg, index=(row_ti, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
        )
        b_dq2_out = b_dq2 + b_dq_old
        ct.store(dq2_seg, index=(row_ti, i_hv, i_k), tile=ct.astype(b_dq2_out, dq2.dtype).reshape((BC, 1, BK)))

        # ---------------- dk transpose contribution (later sub-chunks) --------
        # Accumulated in transposed space ([BK, BC], M=BK=32 -> HMMA): for
        # dkt = (dA_block)^T @ x it holds dkt^T = x^T @ dA_block, so the raw
        # (untransposed) dAqk/dAkk blocks feed the mma RHS directly and the
        # [BC, BC] transposes disappear.
        b_dkt_t = ct.zeros((BK, BC), dtype=ct.float32)
        # NC_eff = min(NC, cdiv(seqlen - off_t, BC)) handled by runtime guards.
        # gate at last row of this sub-chunk
        last_row = ct.minimum(e_ti + BC, seqlen) - 1
        b_gn2 = ct.astype(
            ct.gather(
                g_seg,
                (ct.full((BK,), last_row, dtype=ct.int32), ct.full((BK,), i_hv, dtype=ct.int32), o_k),
                mask=m_k,
                padding_value=0.0,
            ),
            ct.float32,
        )[None, :]
        for i_j in range(0, NC):
            e_j = off_t + i_j * BC
            if i_j > i_i and i_j < NC and e_j < seqlen:
                jrow = i_tt * SUB + i_j
                b_bj = ct.load(beta_seg, index=(jrow, i_hv), shape=(BC, 1)).reshape((BC,))
                b_qj = ct.astype(
                    ct.load(q_seg, index=(jrow, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                b_kbj = (
                    ct.astype(ct.load(k_seg, index=(jrow, i_h, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32)
                    * b_bj[:, None]
                )
                b_gkj = ct.astype(
                    ct.load(g_seg, index=(jrow, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
                )
                # dAqk / dAkk raw blocks (row block i_j, col block i_i); the
                # needed transpose is absorbed into the transposed-space mma.
                b_dAqk_j = ct.astype(
                    ct.load(dAqk_seg, index=(jrow, i_hv, i_i), shape=(BC, 1, BC)).reshape((BC, BC)), ct.float32
                )
                b_dAkk_j = ct.astype(
                    ct.load(dAkk_seg, index=(jrow, i_hv, i_i), shape=(BC, 1, BC)).reshape((BC, BC)), ct.float32
                )
                o_j = e_j + o_i
                m_j = (o_j < seqlen)[:, None]
                b_gkn = ct.exp2(b_gkj - b_gn2)
                b_qg = b_qj * ct.where(m_j, b_gkn, 0.0)
                b_kbg = b_kbj * ct.where(m_j, b_gkn, 0.0)
                b_dkt_t = ct.mma(ct.astype(ct.transpose(b_qg), ct.tfloat32), ct.astype(b_dAqk_j, ct.tfloat32), b_dkt_t)
                b_dkt_t = ct.mma(ct.astype(ct.transpose(b_kbg), ct.tfloat32), ct.astype(b_dAkk_j, ct.tfloat32), b_dkt_t)
        b_dkt = ct.transpose(b_dkt_t) * ct.exp2(b_gn2 - b_g)

        # diagonal contribution to dkt (mid-pivot factorized; see the
        # qk-diagonal comment above):
        #   dkt[i,d] += sum_j (i<=j<n_j) dA[j,i] * x[j,d] * 2^(g[j,d]-g[i,d])
        # In transposed space the upper-tri-masked transposed diag block
        # (m_kk o dA^T)^T equals the lower-tri-masked raw block, i.e. exactly
        # the resident b_dAqk_qk / b_dAkk_qk tiles from the qk section, so
        # dkt_diag^T = x_exp^T @ b_dA*_qk with M=BK=32 -> HMMA.
        b_b2 = ct.load(beta_seg, index=(row_ti, i_hv), shape=(BC, 1)).reshape((BC,))
        b_q_exp = ct.astype(b_q, ct.float32) * exp_b_g_diag
        b_kb_exp = ct.astype(b_k, ct.float32) * b_b2[:, None] * exp_b_g_diag
        b_dktd_t = ct.mma(
            ct.astype(ct.transpose(b_q_exp), ct.tfloat32),
            ct.astype(b_dAqk_qk, ct.tfloat32),
            ct.zeros((BK, BC), dtype=ct.float32),
        )
        b_dktd_t = ct.mma(ct.astype(ct.transpose(b_kb_exp), ct.tfloat32), ct.astype(b_dAkk_qk, ct.tfloat32), b_dktd_t)
        b_dkt = b_dkt + ct.transpose(b_dktd_t) * exp_neg_b_g_diag

        # ---------------- final dk2 / dg2 accumulation ------------------------
        b_dg_old = ct.astype(
            ct.load(dg_seg, index=(row_ti, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
        )
        b_dk_old = ct.astype(
            ct.load(dk_seg, index=(row_ti, i_hv, i_k), shape=(BC, 1, BK)).reshape((BC, BK)), ct.float32
        )
        b_dg2 = b_dg2 + (b_dk2 - b_dkt) * ct.astype(b_k, ct.float32) + b_dg_old
        b_dk2_out = b_dk2 + b_dk_old + b_dkt
        ct.store(dk2_seg, index=(row_ti, i_hv, i_k), tile=ct.astype(b_dk2_out, dk2.dtype).reshape((BC, 1, BK)))
        ct.store(dg2_seg, index=(row_ti, i_hv, i_k), tile=ct.astype(b_dg2, dg2.dtype).reshape((BC, 1, BK)))


# ======================== Wrapper Functions ========================


def _launch_fwd_full(stream, grid, args, cache_key):
    _tuned_launch_hints(chunk_kda_fwd_kernel_intra_wu_fused_full, stream, grid, args, cache_key, _RECOMPUTE_WU_HINTS)


def _launch_bwd_intra(stream, grid, args):
    ct.launch(stream, grid, chunk_kda_bwd_kernel_intra, args)


@input_guard
def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
):
    """Token-parallel diagonal Aqk / Akk. Writes in-place into Aqk, Akk."""
    B, T, H, K, HV = *q.shape, gk.shape[2]
    BT = chunk_size
    BC = sub_chunk_size
    BH = _next_power_of_2(HV)
    BK = _next_power_of_2(K)
    G = HV // H
    IS_VARLEN = cu_seqlens is not None

    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    g3 = gk.reshape(BT_total, HV, K)
    beta2 = beta.reshape(BT_total, HV)
    Aqk3 = Aqk.reshape(BT_total, HV, BT)
    Akk3 = Akk.reshape(BT_total, HV, BC)

    if IS_VARLEN:
        ti = prepare_token_indices(cu_seqlens).to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ti = _cached_dummy(k.device, torch.int32, 1, 2)
        cs = _cached_dummy(k.device, torch.int32, 1)

    grid = (BT_total, _cdiv(HV, BH), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_kda_fwd_kernel_intra_token_parallel,
        (
            q3,
            k3,
            g3,
            beta2,
            Aqk3,
            Akk3,
            ti,
            cs,
            scale,
            T,
            HV,
            K,
            BT,
            BC,
            BH,
            BK,
            G,
            IS_VARLEN,
        ),
    )
    return Aqk, Akk


@input_guard
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64
    G = HV // H

    STORE_QG = q is not None
    STORE_KG = True
    IS_VARLEN = cu_seqlens is not None

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    qg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype) if q is not None else None
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)

    # Flat element-offset form -- see the note on recompute_w_u_fwd_kda_kernel.
    _check_flat_addressing(k, v, beta, A, gk, w, u, kg)
    k3 = k.contiguous().reshape(-1)
    v3 = v.contiguous().reshape(-1)
    beta2 = beta.contiguous().reshape(-1)
    w3 = w.reshape(-1)
    u3 = u.reshape(-1)
    A3 = A.contiguous().reshape(-1)
    gk3 = gk.contiguous().reshape(-1)
    kg3 = kg.reshape(-1)

    if STORE_QG:
        q3 = q.contiguous().reshape(-1)
        qg3 = qg.reshape(-1)
    else:
        q3 = _cached_dummy(k.device, k.dtype, 1)
        qg3 = _cached_dummy(k.device, k.dtype, 1)

    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32).reshape(-1)
        cs = cu_seqlens.to(torch.int32).reshape(-1)
    else:
        ci = _cached_dummy(k.device, torch.int32, 2)
        cs = _cached_dummy(k.device, torch.int32, 1)

    grid = (NT, B * HV, 1)
    _tuned_launch_hints(
        recompute_w_u_fwd_kda_kernel,
        torch.cuda.current_stream(),
        grid,
        (
            q3,
            k3,
            qg3,
            kg3,
            v3,
            beta2,
            w3,
            u3,
            A3,
            gk3,
            ci,
            cs,
            T,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            G,
            STORE_QG,
            STORE_KG,
            IS_VARLEN,
        ),
        (T, HV, K, V, BT, BK, BV, G, STORE_QG, STORE_KG, IS_VARLEN),
        _RECOMPUTE_WU_HINTS,
    )
    return w, u, qg, kg


@input_guard
def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    BC = 32 if BT == 64 and K >= 64 else 16
    G = HV // H
    IS_VARLEN = cu_seqlens is not None

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = _cdiv(BT, BC)

    # Fused inter-solve + recompute_w_u path (NC == 2 only): the solved A tile
    # stays in registers and feeds the w/u mmas directly, and the kernel stores
    # the FULL [BT, BT] merged tile (incl. zero upper blocks), so Akk needs no
    # torch.zeros fill. Other tilings (NC == 4) use the unfused kernel pair,
    # whose partial block stores rely on the zero fill.
    use_fused_wu = _cdiv(BT, BC) == 2

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    if use_fused_wu:
        Akk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    else:
        # Akk must be zero-initialized - kernel only writes lower triangular.
        Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)

    BT_total = B * T

    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = _cached_dummy(k.device, torch.int32, 1, 2)
        cs = _cached_dummy(k.device, torch.int32, 1)
    use_split_diag_compute_solve = (not safe_gate) and BT == 64 and K >= 64
    use_solved_diagonal = safe_gate or use_split_diag_compute_solve

    if use_fused_wu and use_solved_diagonal and BC == 32 and K <= 64:
        # SINGLE fully-fused kernel (diagonal + inter + solve + w/u): no
        # intra_sub_chunk launch, no Akkd round trip (see kernel docstring).
        V = v.shape[-1]
        BK_FF = 64  # single K tile (path is gated to K <= 64)
        BV = 64
        STORE_QG = disable_recompute
        w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
        u = torch.empty_like(v)
        kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
        qg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype) if STORE_QG else None
        qf = q.contiguous().reshape(-1)
        qgf = qg.reshape(-1) if STORE_QG else _cached_dummy(k.device, k.dtype, 1)
        grid = (NT, B * HV, 1)
        _check_flat_addressing(q, k, gk, beta, v, Aqk, Akk, w, u, kg)
        _launch_fwd_full(
            torch.cuda.current_stream(),
            grid,
            (
                qf,
                k.contiguous().reshape(-1),
                gk.contiguous().reshape(-1),
                beta.contiguous().reshape(-1),
                v.contiguous().reshape(-1),
                Aqk.reshape(-1),
                Akk.reshape(-1),
                w.reshape(-1),
                u.reshape(-1),
                qgf,
                kg.reshape(-1),
                ci.reshape(-1),
                cs.reshape(-1),
                scale,
                T,
                HV,
                K,
                V,
                BT,
                BC,
                BK_FF,
                BV,
                G,
                STORE_QG,
                IS_VARLEN,
                IS_TF32_SUPPORTED,
            ),
            (T, HV, K, V, BT, BC, BK_FF, BV, G, STORE_QG, IS_VARLEN, IS_TF32_SUPPORTED),
        )
        return w, u, qg, kg, Aqk, Akk

    # Separate fp32 buffer for diagonal 16x16 blocks (precision in solve_tril).
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    # Step 1: diagonal blocks into Akkd (fp32) + diagonal Aqk into Aqk.
    if use_solved_diagonal:
        BK = _next_power_of_2(K)
        grid = (NT, NC, B * HV)
        _check_flat_addressing(q, k, gk, beta, Aqk, Akkd)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            chunk_kda_fwd_kernel_intra_sub_chunk,
            (
                # The kernel uses element-offset addressing on flat views.
                q.contiguous().reshape(-1),
                k.contiguous().reshape(-1),
                gk.contiguous().reshape(-1),
                beta.contiguous().reshape(-1),
                Aqk.reshape(-1),
                Akkd.reshape(-1),
                ci.reshape(-1),
                cs.reshape(-1),
                scale,
                T,
                HV,
                K,
                BT,
                BC,
                BK,
                G,
                IS_VARLEN,
            ),
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: fused inter-subchunk compute + solve_tril (+ recompute_w_u when
    # NC == 2 -- the solved A tile never round-trips DRAM, see the fused
    # kernel's docstring).
    BK = 32
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    g3 = gk.reshape(BT_total, HV, K)
    beta2 = beta.reshape(BT_total, HV)
    Aqk3 = Aqk.reshape(BT_total, HV, BT)
    Akkd3 = Akkd.reshape(BT_total, HV, BC)
    Akk3 = Akk.reshape(BT_total, HV, BT)
    grid = (NT, B * HV, 1)

    if use_fused_wu:
        V = v.shape[-1]
        BKW = 64  # w/qg/kg K tile (mirrors the unfused recompute_w_u BK)
        BV = 64
        # BK=64 (vs the unfused kernel's 32) halves the serial off-diagonal
        # mma chain: at small grids (~12 CTAs) the fused kernel is
        # pure per-CTA latency, so fewer, larger barrier-synchronized tile
        # ops win. Only when K is a multiple of 64 (otherwise keep the
        # unfused kernel's exact-fit tile and boundary behavior).
        BK_F = 64 if K % 64 == 0 else 32
        STORE_QG = disable_recompute
        w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
        u = torch.empty_like(v)
        kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
        qg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype) if STORE_QG else None
        # Flat 1-D views for the fused w/u tail (see the kernel docstring).
        vf = v.contiguous().reshape(-1)
        wf = w.reshape(-1)
        uf = u.reshape(-1)
        kgf = kg.reshape(-1)
        if STORE_QG:
            qf = q.contiguous().reshape(-1)
            qgf = qg.reshape(-1)
        else:
            qf = _cached_dummy(k.device, k.dtype, 1)
            qgf = _cached_dummy(k.device, k.dtype, 1)
        _check_flat_addressing(q, k, gk, beta, v, Akk, w, u, kg)
        _tuned_launch_hints(
            chunk_kda_fwd_kernel_inter_solve_wu_fused,
            torch.cuda.current_stream(),
            grid,
            (
                q3,
                k3,
                g3,
                beta2,
                Aqk3,
                Akkd3,
                qf,
                k3.reshape(-1),
                g3.reshape(-1),
                beta2.reshape(-1),
                vf,
                Akk3.reshape(-1),
                wf,
                uf,
                qgf,
                kgf,
                ci,
                cs,
                scale,
                T,
                HV,
                K,
                V,
                BT,
                BC,
                BK_F,
                BKW,
                BV,
                G,
                STORE_QG,
                IS_VARLEN,
                use_solved_diagonal,
                IS_TF32_SUPPORTED,
            ),
            # cache_key = the compile-affecting ConstInt/ConstBool args (occ/nww
            # is tuned per unique kernel instantiation).
            (T, HV, K, V, BT, BC, BK_F, BKW, BV, G, STORE_QG, IS_VARLEN, use_solved_diagonal, IS_TF32_SUPPORTED),
            _RECOMPUTE_WU_HINTS,
        )
        return w, u, qg, kg, Aqk, Akk

    _tuned_launch_hints(
        chunk_kda_fwd_kernel_inter_solve_fused,
        torch.cuda.current_stream(),
        grid,
        (
            q3,
            k3,
            g3,
            beta2,
            Aqk3,
            Akkd3,
            Akk3,
            ci,
            cs,
            scale,
            T,
            HV,
            K,
            BT,
            BC,
            NC,
            BK,
            G,
            IS_VARLEN,
            use_solved_diagonal,
            IS_TF32_SUPPORTED,
        ),
        # cache_key = the compile-affecting ConstInt/ConstBool args (occ/nww is
        # tuned per unique kernel instantiation).
        (T, HV, K, BT, BC, NC, BK, G, IS_VARLEN, use_solved_diagonal, IS_TF32_SUPPORTED),
        _INTER_SOLVE_HINTS,
    )

    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


@input_guard
def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, _next_power_of_2(K))
    G = HV // H
    IS_VARLEN = cu_seqlens is not None

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = _cdiv(BT, BC)
    NK = _cdiv(K, BK)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)

    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    g3 = g.reshape(BT_total, HV, K)
    beta2 = beta.reshape(BT_total, HV)
    dAqk3 = dAqk.reshape(BT_total, HV, BT)
    dAkk3 = dAkk.reshape(BT_total, HV, BT)
    dq3 = dq.reshape(BT_total, HV, K)
    dq23 = dq2.reshape(BT_total, HV, K)
    dk3 = dk.reshape(BT_total, HV, K)
    dk23 = dk2.reshape(BT_total, HV, K)
    dg3 = dg.reshape(BT_total, HV, K)
    dg23 = dg2.reshape(BT_total, HV, K)
    # db2 : [NK, B, T, HV] -> [NK * BT_total, HV]
    db23 = db2.reshape(NK * BT_total, HV)

    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = _cached_dummy(k.device, torch.int32, 1, 2)
        cs = _cached_dummy(k.device, torch.int32, 1)

    grid = (NK * NC, NT, B * HV)
    _launch_bwd_intra(
        torch.cuda.current_stream(),
        grid,
        (
            q3,
            k3,
            g3,
            beta2,
            dAqk3,
            dAkk3,
            dq3,
            dq23,
            dk3,
            dk23,
            dg3,
            dg23,
            db23,
            ci,
            cs,
            B,
            T,
            HV,
            K,
            BT,
            BC,
            BK,
            NC,
            G,
            IS_VARLEN,
            safe_gate,
        ),
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = dg2

    return dq, dk, db, dg
