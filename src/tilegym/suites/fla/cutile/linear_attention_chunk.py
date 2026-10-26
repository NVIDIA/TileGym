# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools

import cuda.tile as ct
import torch
import torch.nn.functional as F

from tilegym.backend import register_impl
from tilegym.suites.fla.cutile._host_utils import _segmented_arange
from tilegym.suites.fla.cutile._host_utils import _sm_split_threshold
from tilegym.suites.fla.cutile._host_utils import input_guard
from tilegym.suites.fla.cutile._normalization import StatefulNormalizeFunction

RCP_LN2 = 1.4426950216  # 1/ln(2), fla.ops.utils.constant.RCP_LN2

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad((prepare_lens(cu_seqlens) + chunk_size - 1) // chunk_size, (1, 0), value=0).cumsum(-1)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _cdiv_kernel(a, b):
    # In-kernel ceil-div over compile-time constants. Plain Python arithmetic on
    # ConstInt is fine for static loop bounds.
    return (a + b - 1) // b


# ===========================================================================
# CuTile Kernels
# ===========================================================================


# ---------------------------------------------------------------------------
# fla.ops.utils.cumsum -- local / global cumulative sum, scalar (3D) variants.
# ---------------------------------------------------------------------------


@ct.function
def _normalization_product_stages(q, kc, scale):
    qs = ct.astype(ct.astype(q, ct.float32) * scale, q.dtype)
    product = ct.astype(ct.astype(qs, ct.float32) * ct.astype(kc, ct.float32), q.dtype)
    return qs, product


@ct.kernel
def chunk_local_cumsum_scalar_kernel(
    s,  # [B, T, H]  (or [1, total_T, H] varlen)
    o,  # [B, T, H]  output
    cu_seqlens,  # [N+1] int (dummy [1] when not varlen)
    chunk_indices,  # [NT, 2] int (dummy [1, 2] when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    BT: ConstInt,
    REVERSE: ConstBool,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Per-chunk inclusive cumulative sum along time (scalar / no feature dim).

    grid = (NT, B * H). Layout is HEAD_FIRST=False ([B, T, H]) only -- the
    linear-attn entry never uses head_first.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=True, padding_value=0).item()
        i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=True, padding_value=0).item()
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        Tloc = eos - bos
        row = 0
    else:
        Tloc = T
        bos = 0
        row = i_b

    t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
    glob_t = bos + t_local
    in_bounds = t_local < Tloc

    b_s = ct.gather(s, (row, glob_t, i_h), mask=in_bounds, check_bounds=True, padding_value=0.0)
    b_s = ct.astype(b_s, ct.float32)
    # Zero the padded tail so it does not pollute the cumsum / sum.
    b_s = ct.where(in_bounds, b_s, 0.0)

    b_o = ct.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = ct.sum(b_s, axis=0, keepdims=True)
        b_o = -b_o + b_z + b_s
    if HAS_SCALE:
        b_o = b_o * scale

    ct.scatter(o, (row, glob_t, i_h), ct.astype(b_o, o.dtype), mask=in_bounds, check_bounds=True)


@ct.kernel
def chunk_local_cumsum_vector_kernel(
    s,  # [B, T, H, S]  (or [1, total_T, H, S] varlen)
    o,  # [B, T, H, S]  output
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    chunk_indices,  # [NT, 2] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    S: ConstInt,
    BT: ConstInt,
    BS: ConstInt,
    REVERSE: ConstBool,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Per-chunk inclusive cumulative sum along time for a vector feature dim S.

    grid = (cdiv(S, BS), NT, B * H). Layout [B, T, H, S] (head_first=False).
    """
    i_s = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=True, padding_value=0).item()
        i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=True, padding_value=0).item()
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        Tloc = eos - bos
        row = 0
    else:
        Tloc = T
        bos = 0
        row = i_b

    t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
    glob_t = bos + t_local
    s_off = i_s * BS + ct.arange(BS, dtype=ct.int32)
    m_t = t_local < Tloc
    m_s = s_off < S
    mask = m_t[:, None] & m_s[None, :]

    b_s = ct.gather(
        s,
        (row, glob_t[:, None], i_h, s_off[None, :]),
        mask=mask,
        check_bounds=True,
        padding_value=0.0,
    )
    b_s = ct.astype(b_s, ct.float32)
    b_s = ct.where(mask, b_s, 0.0)

    b_o = ct.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale

    ct.scatter(
        o,
        (row, glob_t[:, None], i_h, s_off[None, :]),
        ct.astype(b_o, o.dtype),
        mask=mask,
        check_bounds=True,
    )


@ct.kernel
def chunk_global_cumsum_scalar_kernel(
    s,  # [B, T, H]
    o,  # [B, T, H] output
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    BT: ConstInt,
    REVERSE: ConstBool,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Whole-sequence (across chunks) inclusive cumsum along time (scalar).

    grid = (N * H,). Sequential running carry b_z across the NT chunks.
    """
    i_nh = ct.bid(0)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        row = 0
    else:
        bos = 0
        eos = T
        row = i_n
    Tloc = eos - bos

    b_z = ct.zeros((), dtype=ct.float32)
    NT = ct.cdiv(Tloc, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
        glob_t = bos + t_local
        in_bounds = t_local < Tloc

        b_s = ct.gather(s, (row, glob_t, i_h), mask=in_bounds, check_bounds=True, padding_value=0.0)
        b_s = ct.astype(b_s, ct.float32)
        b_s = ct.where(in_bounds, b_s, 0.0)

        b_o = ct.cumsum(b_s, axis=0)
        b_ss = ct.sum(b_s, axis=0, keepdims=True)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o = b_o + b_z
        b_z = b_z + ct.sum(b_ss, axis=0)
        if HAS_SCALE:
            b_o = b_o * scale
        ct.scatter(o, (row, glob_t, i_h), ct.astype(b_o, o.dtype), mask=in_bounds, check_bounds=True)


@ct.kernel
def chunk_global_cumsum_vector_kernel(
    s,  # [B, T, H, S]
    o,  # [B, T, H, S] output
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    S: ConstInt,
    BT: ConstInt,
    BS: ConstInt,
    REVERSE: ConstBool,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Whole-sequence inclusive cumsum along time for a vector feature dim S.

    grid = (cdiv(S, BS), N * H). Running carry b_z of shape [BS].
    """
    i_s = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        row = 0
    else:
        bos = 0
        eos = T
        row = i_n
    Tloc = eos - bos

    s_off = i_s * BS + ct.arange(BS, dtype=ct.int32)
    m_s = s_off < S

    b_z = ct.zeros((BS,), dtype=ct.float32)
    NT = ct.cdiv(Tloc, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
        glob_t = bos + t_local
        m_t = t_local < Tloc
        mask = m_t[:, None] & m_s[None, :]

        b_s = ct.gather(
            s,
            (row, glob_t[:, None], i_h, s_off[None, :]),
            mask=mask,
            check_bounds=True,
            padding_value=0.0,
        )
        b_s = ct.astype(b_s, ct.float32)
        b_s = ct.where(mask, b_s, 0.0)

        b_c = b_z[None, :] + ct.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_c = b_c * scale
        ct.scatter(
            o,
            (row, glob_t[:, None], i_h, s_off[None, :]),
            ct.astype(b_c, o.dtype),
            mask=mask,
            check_bounds=True,
        )
        b_z = b_z + ct.sum(b_s, axis=0)


# ---------------------------------------------------------------------------
# fla.ops.common.chunk_h -- forward state h, backward state-gradient dh.
# Layout: k,v are [B, T, H, K]/[B, T, H, V]; h is [B, NS, H, K, V]
# (STATE_V_FIRST=False). USE_G/GK/GV and GQA paths are not reached by the
# linear-attn entry, so they are omitted (see module header).
# ---------------------------------------------------------------------------


@ct.kernel
def chunk_fwd_kernel_h(
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    h,  # [B, NS, H, K, V] output (states per chunk)
    h0,  # initial state [B, H, K, V] (dummy [1,1,1,1] when unused)
    ht,  # final state [N, H, K, V] (dummy when unused)
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    split_offsets,  # [N+1] int (dummy when not varlen)
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    USE_INITIAL_STATE: ConstBool,
    STORE_FINAL_STATE: ConstBool,
    IS_VARLEN: ConstBool,
    USE_IEEE_DOT: ConstBool,
    STATE_FP32: ConstBool,
):
    """Forward chunk state recurrence: b_h += k_chunk^T @ v_chunk per chunk.

    grid = (cdiv(K, BK), cdiv(V, BV), N * H). The state written for chunk i_t is
    the running state BEFORE that chunk's contribution (prefix), matching fla.
    """
    i_k = ct.bid(0)
    i_v = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    # h is [B, NS, H, K, V]: dense -> (i_n, chunk_in_batch); varlen (B==1) ->
    # (0, split_offsets[i_n] + chunk). BS == BT here so chunk_in_batch == i_t.
    if IS_VARLEN:
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        boh = ct.gather(split_offsets, (i_n,), check_bounds=True, padding_value=0).item()
        h_b = 0
    else:
        bos = 0
        eos = T
        boh = 0
        h_b = i_n
    Tloc = eos - bos
    NT = ct.cdiv(Tloc, BT)

    k_seg = k.slice(axis=1, start=bos, stop=eos)
    v_seg = v.slice(axis=1, start=bos, stop=eos)

    b_h = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        # h0 : [B, H, K, V]; load [BK, BV] block at tile (i_n, i_h, i_k, i_v).
        b_h0 = ct.load(h0, index=(i_n, i_h, i_k, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BK, BV)
        )
        b_h = ct.astype(b_h0, ct.float32)

    for i_t in range(NT):
        t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
        m_t = t_local < Tloc
        # Store the running prefix state for this chunk (h: [B, NS, H, K, V]).
        o_chunk = boh + i_t
        ct.store(
            h,
            index=(h_b, o_chunk, i_h, i_k, i_v),
            tile=ct.astype(b_h, h.dtype).reshape((1, 1, 1, BK, BV)),
        )

        # b_k : [BK, BT]  (transposed),  b_v : [BT, BV]
        # Batch block index is h_b (== i_n for dense, 0 for varlen/B==1). The
        # slice on axis=1 only windows T, so the batch dim must be selected here
        # rather than hardcoded to 0 (otherwise every batch reads batch 0's k/v).
        b_k = ct.load(
            k_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BK))
        b_v = ct.load(
            v_seg, index=(h_b, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BV))
        # zero padded rows so they do not contribute to the state
        b_k = ct.where(m_t[:, None], b_k, ct.zeros((BT, BK), dtype=b_k.dtype))
        b_v = ct.where(m_t[:, None], b_v, ct.zeros((BT, BV), dtype=b_v.dtype))

        b_kt = ct.transpose(b_k)  # [BK, BT]
        zero_acc = ct.zeros((BK, BV), dtype=ct.float32)
        if USE_IEEE_DOT:
            b_delta = ct.mma(b_kt, b_v, zero_acc)
        elif STATE_FP32:
            a = ct.astype(b_kt, ct.tfloat32)
            b = ct.astype(b_v, ct.tfloat32)
            b_delta = ct.mma(a, b, zero_acc)
        else:
            a = ct.astype(b_kt, ct.tfloat32) if b_kt.dtype == ct.float32 else b_kt
            b = ct.astype(b_v, ct.tfloat32) if b_v.dtype == ct.float32 else b_v
            b_delta = ct.mma(a, b, zero_acc)
        b_h = b_h + b_delta

    if STORE_FINAL_STATE:
        ct.store(
            ht,
            index=(i_n, i_h, i_k, i_v),
            tile=ct.astype(b_h, ht.dtype).reshape((1, 1, BK, BV)),
        )


@ct.kernel
def chunk_bwd_kernel_dh(
    q,  # [B, T, H, K]
    do,  # [B, T, H, V]
    dh,  # [B, NS, H, K, V] output
    dht,  # final-state grad [N, H, K, V] (dummy when unused)
    dh0,  # initial-state grad [B, H, K, V] (dummy when unused)
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    split_offsets,  # [N+1] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    STORE_INITIAL_STATE_GRADIENT: ConstBool,
    USE_FINAL_STATE_GRADIENT: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Backward state-grad recurrence (reverse over chunks): b_dh += q^T @ do.

    grid = (cdiv(K, BK), cdiv(V, BV), N * H). NG == 1 (HQ == H).
    """
    i_k = ct.bid(0)
    i_v = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    # dh is [B, NS, H, K, V]: dense -> (i_n, chunk); varlen (B==1) -> (0, boh+chunk).
    if IS_VARLEN:
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        boh = ct.gather(split_offsets, (i_n,), check_bounds=True, padding_value=0).item()
        h_b = 0
    else:
        bos = 0
        eos = T
        boh = 0
        h_b = i_n
    Tloc = eos - bos
    NT = ct.cdiv(Tloc, BT)

    q_seg = q.slice(axis=1, start=bos, stop=eos)
    do_seg = do.slice(axis=1, start=bos, stop=eos)

    b_dh = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_FINAL_STATE_GRADIENT:
        b_dht = ct.load(
            dht, index=(i_n, i_h, i_k, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BK, BV))
        b_dh = b_dh + ct.astype(b_dht, ct.float32)

    for i_c in range(NT):
        i_t = NT - 1 - i_c
        t_local = i_t * BT + ct.arange(BT, dtype=ct.int32)
        m_t = t_local < Tloc
        o_chunk = boh + i_t
        ct.store(
            dh,
            index=(h_b, o_chunk, i_h, i_k, i_v),
            tile=ct.astype(b_dh, dh.dtype).reshape((1, 1, 1, BK, BV)),
        )

        # b_q : [BT, BK] (transposed in mma), b_do : [BT, BV]
        b_q = ct.load(
            q_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BK))
        b_do = ct.load(
            do_seg, index=(h_b, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BV))
        b_q = b_q * scale
        b_q = ct.where(m_t[:, None], b_q, ct.zeros((BT, BK), dtype=b_q.dtype))
        b_do = ct.where(m_t[:, None], b_do, ct.zeros((BT, BV), dtype=b_do.dtype))

        b_qt = ct.transpose(b_q)  # [BK, BT]
        a = ct.astype(b_qt, ct.tfloat32) if b_qt.dtype == ct.float32 else b_qt
        b = ct.astype(b_do, ct.tfloat32) if b_do.dtype == ct.float32 else b_do
        zero_acc = ct.zeros((BK, BV), dtype=ct.float32)
        b_delta = ct.mma(a, b, zero_acc)
        b_dh = b_dh + b_delta

    if STORE_INITIAL_STATE_GRADIENT:
        ct.store(
            dh0,
            index=(i_n, i_h, i_k, i_v),
            tile=ct.astype(b_dh, dh0.dtype).reshape((1, 1, BK, BV)),
        )


# ---------------------------------------------------------------------------
# fla.ops.common.chunk_o -- forward output o, backward dq/dk, backward dv.
# Layout [B, T, H, *]; h is [B, NS, H, K, V]. HV == H, state_v_first=False,
# USE_G=False for the linear-attn entry.
# ---------------------------------------------------------------------------


@ct.kernel(occupancy=2, num_worker_warps=4)
def chunk_fwd_kernel_o(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    h,  # [B, NS, H, K, V]
    o,  # [B, T, H, V] output
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    chunk_indices,  # [NT, 2] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    IS_VARLEN: ConstBool,
    USE_IEEE_DOT: ConstBool,
    STATE_FP32: ConstBool,
):
    """Forward output: o = (q @ h) * scale + tril(q @ k^T) @ v * scale.

    grid = (cdiv(V, BV), NT, B * H).
    """
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // H
    i_h = i_bh % H

    # h is [B, NS, H, K, V]: dense -> (i_b, i_t); varlen -> (0, global_chunk).
    if IS_VARLEN:
        i_hg = i_t  # global chunk id (row index into chunk_indices)
        i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=True, padding_value=0).item()
        i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=True, padding_value=0).item()
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        h_b = 0
        h_c = i_hg
    else:
        bos = 0
        eos = T
        h_b = i_b
        h_c = i_t
    Tloc = eos - bos

    q_seg = q.slice(axis=1, start=bos, stop=eos)
    k_seg = k.slice(axis=1, start=bos, stop=eos)
    v_seg = v.slice(axis=1, start=bos, stop=eos)
    o_seg = o.slice(axis=1, start=bos, stop=eos)

    b_o = ct.zeros((BT, BV), dtype=ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(_cdiv_kernel(K, BK)):
        # b_q : [BT, BK], b_k : [BT, BK] (transposed in mma), b_h : [BK, BV]
        # h_b is the batch block index (== i_b for dense, 0 for varlen/B==1).
        b_q = ct.load(
            q_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BK))
        b_k = ct.load(
            k_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BK))
        b_h = ct.load(
            h, index=(h_b, h_c, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BK, BV))

        if USE_IEEE_DOT:
            qa = ct.astype(b_q, ct.tfloat32) if b_q.dtype == ct.float32 else b_q
            ha = ct.astype(b_h, ct.tfloat32) if b_h.dtype == ct.float32 else b_h
            b_o = ct.mma(qa, ha, b_o)

            b_kt = ct.transpose(b_k)
            ka = ct.astype(b_kt, ct.tfloat32) if b_kt.dtype == ct.float32 else b_kt
            b_A = ct.mma(qa, ka, b_A)
        elif STATE_FP32:
            qa = ct.astype(b_q, ct.tfloat32)
            ha = ct.astype(b_h, ct.tfloat32)
            b_o = ct.mma(qa, ha, b_o)

            b_kt = ct.transpose(b_k)
            ka = ct.astype(b_kt, ct.tfloat32)
            b_A = ct.mma(qa, ka, b_A)
        else:
            qa = ct.astype(b_q, ct.tfloat32) if b_q.dtype == ct.float32 else b_q
            ha = ct.astype(b_h, ct.tfloat32) if b_h.dtype == ct.float32 else b_h
            b_o = ct.mma(qa, ha, b_o)

            b_kt = ct.transpose(b_k)
            ka = ct.astype(b_kt, ct.tfloat32) if b_kt.dtype == ct.float32 else b_kt
            b_A = ct.mma(qa, ka, b_A)

    # causal mask (lower-triangular incl. diagonal) + row/col bounds
    o_t = i_t * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < Tloc
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=ct.float32))

    b_v = ct.load(v_seg, index=(h_b, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
        (BT, BV)
    )
    acc = ct.zeros((BT, BV), dtype=ct.float32)
    if USE_IEEE_DOT:
        b_av = ct.mma(b_A, b_v, acc)
    elif STATE_FP32:
        # keep A in fp32 (no bf16 downcast); tf32 dot for the intra-chunk term.
        Aa = ct.astype(b_A, ct.tfloat32)
        va = ct.astype(b_v, ct.tfloat32)
        b_av = ct.mma(Aa, va, acc)
    else:
        b_Av = ct.astype(b_A, b_v.dtype)
        va = ct.astype(b_v, ct.tfloat32) if b_v.dtype == ct.float32 else b_v
        Aa = ct.astype(b_Av, ct.tfloat32) if b_Av.dtype == ct.float32 else b_Av
        b_av = ct.mma(Aa, va, acc)
    b_o = b_o * scale + b_av * scale

    ct.store(
        o_seg,
        index=(h_b, i_t, i_h, i_v),
        tile=ct.astype(b_o, o.dtype).reshape((1, BT, 1, BV)),
    )


@ct.kernel
def chunk_bwd_kernel_dqkwg(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    h,  # [B, NS, H, K, V]
    do,  # [B, T, H, V]
    dh,  # [B, NS, H, K, V]
    dq,  # [B, T, H, K] output
    dk,  # [B, T, H, K] output
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    chunk_indices,  # [NT, 2] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    IS_VARLEN: ConstBool,
):
    i_k = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // H
    i_h = i_bh % H

    # h/dh are [B, NS, H, K, V]: dense -> (i_b, i_t); varlen -> (0, global_chunk).
    if IS_VARLEN:
        i_hg = i_t
        i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=True, padding_value=0).item()
        i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=True, padding_value=0).item()
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        h_b = 0
        h_c = i_hg
    else:
        bos = 0
        eos = T
        h_b = i_b
        h_c = i_t
    Tloc = eos - bos

    q_seg = q.slice(axis=1, start=bos, stop=eos)
    k_seg = k.slice(axis=1, start=bos, stop=eos)
    v_seg = v.slice(axis=1, start=bos, stop=eos)
    do_seg = do.slice(axis=1, start=bos, stop=eos)
    dq_seg = dq.slice(axis=1, start=bos, stop=eos)
    dk_seg = dk.slice(axis=1, start=bos, stop=eos)

    b_dq = ct.zeros((BT, BK), dtype=ct.float32)
    b_dk = ct.zeros((BT, BK), dtype=ct.float32)
    b_ds = ct.zeros((BT, BT), dtype=ct.float32)

    for i_v in range(_cdiv_kernel(V, BV)):
        b_v = ct.load(
            v_seg, index=(h_b, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BV))
        b_do = ct.load(
            do_seg, index=(h_b, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BV))
        b_h = ct.transpose(
            ct.load(
                h, index=(h_b, h_c, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO
            ).reshape((BK, BV))
        )  # [BV, BK]
        b_dh = ct.transpose(
            ct.load(
                dh, index=(h_b, h_c, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO
            ).reshape((BK, BV))
        )  # [BV, BK]

        # b_ds += b_do @ b_v^T   -> [BT, BT]
        vt = ct.transpose(b_v)  # [BV, BT]
        doa = ct.astype(b_do, ct.tfloat32) if b_do.dtype == ct.float32 else b_do
        vta = ct.astype(vt, ct.tfloat32) if vt.dtype == ct.float32 else vt
        b_ds = ct.mma(doa, vta, b_ds)
        ha = ct.astype(b_h, doa.dtype)
        b_dq = ct.mma(doa, ha, b_dq)
        # b_dk += b_v @ b_dh   ([BT, BV] @ [BV, BK] = [BT, BK])
        va = ct.astype(b_v, ct.tfloat32) if b_v.dtype == ct.float32 else b_v
        dha = ct.astype(b_dh, va.dtype)
        b_dk = ct.mma(va, dha, b_dk)

    b_q = ct.load(q_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
        (BT, BK)
    )
    b_k = ct.load(k_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
        (BT, BK)
    )

    o_t = i_t * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < Tloc
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])

    # no-gate branch
    b_ds = ct.where(m_A, b_ds, ct.zeros((BT, BT), dtype=ct.float32))
    b_dsk = ct.astype(b_ds, b_k.dtype)
    dsa = ct.astype(b_dsk, ct.tfloat32) if b_dsk.dtype == ct.float32 else b_dsk
    ka = ct.astype(b_k, ct.tfloat32) if b_k.dtype == ct.float32 else b_k
    b_dq = ct.mma(dsa, ka, b_dq)

    dst = ct.transpose(b_dsk)
    dsta = ct.astype(dst, ct.tfloat32) if dst.dtype == ct.float32 else dst
    qa = ct.astype(b_q, ct.tfloat32) if b_q.dtype == ct.float32 else b_q
    acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
    b_dk_add = ct.mma(dsta, qa, acc_dk)
    b_dk = b_dk + b_dk_add * scale
    b_dq = b_dq * scale

    ct.store(dq_seg, index=(h_b, i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((1, BT, 1, BK)))
    ct.store(dk_seg, index=(h_b, i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((1, BT, 1, BK)))


@ct.kernel
def chunk_bwd_kernel_dv(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    do,  # [B, T, H, V]
    dv,  # [B, T, H, V] output
    dh,  # [B, NS, H, K, V]
    cu_seqlens,  # [N+1] int (dummy when not varlen)
    chunk_indices,  # [NT, 2] int (dummy when not varlen)
    scale: ct.Constant[float],
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    IS_VARLEN: ConstBool,
):
    """Backward dv (no-gate path): dv = tril(k @ q^T)^T @ do * scale + k @ dh.

    grid = (cdiv(V, BV), NT, B * H).
    """
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // H
    i_h = i_bh % H

    # dh is [B, NS, H, K, V]: dense -> (i_b, i_t); varlen -> (0, global_chunk).
    if IS_VARLEN:
        i_hg = i_t
        i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=True, padding_value=0).item()
        i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=True, padding_value=0).item()
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        h_b = 0
        h_c = i_hg
    else:
        bos = 0
        eos = T
        h_b = i_b
        h_c = i_t
    Tloc = eos - bos

    q_seg = q.slice(axis=1, start=bos, stop=eos)
    k_seg = k.slice(axis=1, start=bos, stop=eos)
    do_seg = do.slice(axis=1, start=bos, stop=eos)
    dv_seg = dv.slice(axis=1, start=bos, stop=eos)

    b_dv = ct.zeros((BT, BV), dtype=ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(_cdiv_kernel(K, BK)):
        b_q = ct.load(
            q_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BK))
        b_k = ct.load(
            k_seg, index=(h_b, i_t, i_h, i_k), shape=(1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BT, BK))
        b_dh = ct.load(
            dh, index=(h_b, h_c, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO
        ).reshape((BK, BV))

        # b_A += b_k @ b_q^T   ([BT, BT])
        qt = ct.transpose(b_q)  # [BK, BT]
        ka = ct.astype(b_k, ct.tfloat32) if b_k.dtype == ct.float32 else b_k
        qta = ct.astype(qt, ct.tfloat32) if qt.dtype == ct.float32 else qt
        b_A = ct.mma(ka, qta, b_A)
        dha = ct.astype(b_dh, ka.dtype)
        b_dv = ct.mma(ka, dha, b_dv)

    o_t = i_t * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < Tloc
    # upper-triangular incl. diagonal (k<=q) for dv
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.where(m_A, b_A * scale, ct.zeros((BT, BT), dtype=ct.float32))

    b_do = ct.load(do_seg, index=(h_b, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
        (BT, BV)
    )
    b_Ad = ct.astype(b_A, b_do.dtype)
    Aa = ct.astype(b_Ad, ct.tfloat32) if b_Ad.dtype == ct.float32 else b_Ad
    doa = ct.astype(b_do, ct.tfloat32) if b_do.dtype == ct.float32 else b_do
    b_dv = ct.mma(Aa, doa, b_dv)

    ct.store(dv_seg, index=(h_b, i_t, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((1, BT, 1, BV)))


@ct.function
def _normalization_divide_stages(o, s):
    sf = ct.astype(s, ct.float32)
    eps = ct.full((), 1e-10, dtype=ct.float32)
    denom = ct.astype(ct.add(sf, eps, rounding_mode=ct.RoundingMode.RN, flush_to_zero=False), s.dtype)
    out = ct.astype(
        ct.truediv(
            ct.astype(o, ct.float32),
            ct.astype(denom, ct.float32),
            rounding_mode=ct.RoundingMode.RN,
            flush_to_zero=False,
        ),
        o.dtype,
    )
    return denom, out


@ct.kernel
def _normalization_divide_kernel(O, S, OUT, D, BR: ct.Constant[int], BC: ct.Constant[int]):
    row = ct.bid(0)
    column = ct.bid(1)
    o = ct.load(O, index=(row, column), shape=(BR, BC), padding_mode=ct.PaddingMode.ZERO)
    s = ct.load(S, index=(row,), shape=(BR,), padding_mode=ct.PaddingMode.ZERO)
    d, out = _normalization_divide_stages(o, s[:, None])
    ct.store(OUT, index=(row, column), tile=out)
    if column == 0:
        ct.store(D, index=(row,), tile=d.reshape((BR,)))


@ct.kernel
def _normalization_product_kernel(Q, KC, P, SCALE: ct.Constant[float], BLOCK: ConstInt):
    block = ct.bid(0)
    q = ct.load(Q, index=(block,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    kc = ct.load(KC, index=(block,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    scale = ct.full((), SCALE, dtype=ct.float32)
    _, product = _normalization_product_stages(q, kc, scale)
    ct.store(P, index=(block,), tile=product)


# ===========================================================================
# Host wrappers
# ===========================================================================


@functools.lru_cache(maxsize=None)
def _dummy_int(device):
    return torch.zeros(1, device=device, dtype=torch.int32)


@functools.lru_cache(maxsize=None)
def _dummy_int2(device):
    return torch.zeros(1, 2, device=device, dtype=torch.int32)


@functools.lru_cache(maxsize=None)
def _dummy_state(device, dtype):
    return torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Cumsum wrappers (fla.ops.utils.cumsum)
# ---------------------------------------------------------------------------


def chunk_local_cumsum_scalar(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float,
    chunk_indices=None,
):
    assert not head_first, "head_first not supported in cuTile port"
    B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(g.device)
    ci_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy_int2(g.device)

    grid = (NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_local_cumsum_scalar_kernel,
        (
            g_org,
            g,
            cu_arg,
            ci_arg,
            float(scale) if scale is not None else 1.0,
            T,
            H,
            BT,
            bool(reverse),
            scale is not None,
            is_varlen,
        ),
    )
    return g


def chunk_local_cumsum_vector(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float,
    chunk_indices=None,
):
    assert not head_first, "head_first not supported in cuTile port"
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BS = min(32, _next_power_of_2(S))

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(g.device)
    ci_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy_int2(g.device)

    grid = (_cdiv(S, BS), NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_local_cumsum_vector_kernel,
        (
            g_org,
            g,
            cu_arg,
            ci_arg,
            float(scale) if scale is not None else 1.0,
            T,
            H,
            S,
            BT,
            BS,
            bool(reverse),
            scale is not None,
            is_varlen,
        ),
    )
    return g


@input_guard
def chunk_global_cumsum_scalar(
    s,
    reverse=False,
    cu_seqlens=None,
    scale=None,
    head_first=False,
    output_dtype=torch.float,
):
    assert not head_first, "head_first not supported in cuTile port"
    B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = 64

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(s.device)

    grid = (N * H,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_global_cumsum_scalar_kernel,
        (
            s,
            z,
            cu_arg,
            float(scale) if scale is not None else 1.0,
            T,
            H,
            BT,
            bool(reverse),
            scale is not None,
            is_varlen,
        ),
    )
    return z


@input_guard
def chunk_global_cumsum_vector(
    s,
    reverse=False,
    cu_seqlens=None,
    scale=None,
    head_first=False,
    output_dtype=torch.float,
):
    assert not head_first, "head_first not supported in cuTile port"
    B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, _next_power_of_2(S))
    BT = 128 if not reverse and T >= 128 else 64

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(s.device)

    grid = (_cdiv(S, BS), N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_global_cumsum_vector_kernel,
        (
            s,
            z,
            cu_arg,
            float(scale) if scale is not None else 1.0,
            T,
            H,
            S,
            BT,
            BS,
            bool(reverse),
            scale is not None,
            is_varlen,
        ),
    )
    return z


@input_guard
def chunk_global_cumsum(
    s,
    reverse=False,
    cu_seqlens=None,
    scale=None,
    head_first=False,
    output_dtype=torch.float,
):
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s, reverse=reverse, cu_seqlens=cu_seqlens, scale=scale, head_first=head_first, output_dtype=output_dtype
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s, reverse=reverse, cu_seqlens=cu_seqlens, scale=scale, head_first=head_first, output_dtype=output_dtype
        )
    else:
        raise ValueError(f"Unsupported input shape {s.shape}")


@input_guard
def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float,
    chunk_indices=None,
    **kwargs,
):
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(f"Unsupported input shape {g.shape}")


# ---------------------------------------------------------------------------
# chunk_h wrappers (fla.ops.common.chunk_h)
# ---------------------------------------------------------------------------


def _tile_kv(K, V):
    BK = min(max(_next_power_of_2(K), 16), 128)
    BV = min(max(_next_power_of_2(V), 16), 128)
    return BK, BV


def _state_tile_kv(K, V, groups, device):
    """Split independent state tiles when the wider grid fits within one SM wave."""
    BK, BV = _tile_kv(K, V)
    split_k, split_v = max(16, BK // 2), max(16, BV // 2)
    blocks = _cdiv(K, BK) * _cdiv(V, BV) * groups
    split_blocks = _cdiv(K, split_k) * _cdiv(V, split_v) * groups
    try:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return BK, BV
    return (split_k, split_v) if blocks < split_blocks <= sm_count else (BK, BV)


def chunk_fwd_h(
    k,
    v,
    g=None,
    g_gamma=None,
    gk=None,
    gv=None,
    h0=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    split_size=None,
    states_in_fp32=False,
    state_fp32_dot=False,
):
    assert g is None and g_gamma is None and gk is None and gv is None, "gated chunk_fwd_h not supported"
    assert not state_v_first, "state_v_first not supported in cuTile port"
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    if cu_seqlens is None:
        N, NS, split_offsets = B, _cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, int(split_offsets[-1].item())

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None

    BK, BV = _state_tile_kv(K, V, N * H, k.device)

    if (
        k.dtype == v.dtype == torch.bfloat16
        and states_in_fp32
        and state_fp32_dot
        and (cu_seqlens is None)
        and (h0 is None)
        and (not output_final_state)
        and (K == V == 64)
        and (BT == 128)
        and (T % BT == 0)
        and (N * H <= _sm_split_threshold(k.device.index))
    ):
        BV = 32
    use_h0 = h0 is not None
    store_ht = ht is not None
    is_varlen = cu_seqlens is not None

    h0_arg = h0 if use_h0 else _dummy_state(k.device, k.dtype)
    ht_arg = ht if store_ht else _dummy_state(k.device, torch.float)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(k.device)
    so_arg = split_offsets.to(torch.int32) if is_varlen else _dummy_int(k.device)

    grid = (_cdiv(K, BK), _cdiv(V, BV), N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_fwd_kernel_h,
        (
            k,
            v,
            h,
            h0_arg,
            ht_arg,
            cu_arg,
            so_arg,
            T,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            use_h0,
            store_ht,
            is_varlen,
            bool(k.dtype is torch.float32),  # USE_IEEE_DOT
            bool(state_fp32_dot),  # STATE_FP32
        ),
    )
    return h, ht


def chunk_bwd_dh(
    q,
    k,
    v,
    do,
    h0,
    dht,
    scale,
    g=None,
    g_gamma=None,
    gk=None,
    gv=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    split_size=None,
    states_in_fp32=False,
):
    assert g is None and g_gamma is None and gk is None and gv is None, "gated chunk_bwd_dh not supported"
    assert not state_v_first, "state_v_first not supported in cuTile port"
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    assert HQ == H, "GQA (HQ != H) not supported in cuTile port"
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0
    if cu_seqlens is None:
        N, NS, split_offsets = B, _cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, int(split_offsets[-1].item())

    dh = k.new_empty(B, NS, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    BK, BV = _state_tile_kv(K, V, N * HQ, k.device)

    store_dh0 = dh0 is not None
    use_dht = dht is not None
    is_varlen = cu_seqlens is not None

    dht_arg = dht if use_dht else _dummy_state(k.device, torch.float)
    dh0_arg = dh0 if store_dh0 else _dummy_state(k.device, torch.float)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(k.device)
    so_arg = split_offsets.to(torch.int32) if is_varlen else _dummy_int(k.device)

    grid = (_cdiv(K, BK), _cdiv(V, BV), N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_bwd_kernel_dh,
        (
            q,
            do,
            dh,
            dht_arg,
            dh0_arg,
            cu_arg,
            so_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            store_dh0,
            use_dht,
            is_varlen,
        ),
    )
    return dh, dh0


# ---------------------------------------------------------------------------
# chunk_o wrappers (fla.ops.common.chunk_o)
# ---------------------------------------------------------------------------


# Cache the occupancy-4 `replace_hints` variant of chunk_fwd_kernel_o (its JIT
# cache lives on the hinted-kernel object, so reuse one instance per process).
@functools.lru_cache(maxsize=None)
def _o_kernel_occ4():
    return chunk_fwd_kernel_o.replace_hints(occupancy=4)


def _o_kernel_for_tile(BK, BV):
    # Larger tiles retain the default occupancy to limit register spills.
    if BK == 64 and BV == 64:
        return _o_kernel_occ4()
    return chunk_fwd_kernel_o


def chunk_fwd_o(
    q,
    k,
    v,
    h,
    g=None,
    g_gamma=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
    state_fp32_dot=False,
    out_dtype=None,
):
    assert g is None and g_gamma is None, "gated chunk_fwd_o not supported"
    assert not state_v_first, "state_v_first not supported in cuTile port"
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    assert HV == H, "GQA (HV != H) not supported in cuTile port"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v, dtype=out_dtype) if out_dtype is not None else torch.empty_like(v)

    BK, BV = _tile_kv(K, V)
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(q.device)
    ci_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy_int2(q.device)

    grid = (_cdiv(V, BV), NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _o_kernel_for_tile(BK, BV),
        (
            q,
            k,
            v,
            h,
            o,
            cu_arg,
            ci_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            is_varlen,
            bool(q.dtype is torch.float32),  # USE_IEEE_DOT
            bool(state_fp32_dot),  # STATE_FP32
        ),
    )
    return o


def chunk_bwd_dv(
    q,
    k,
    do,
    dh,
    g=None,
    g_gamma=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    assert g is None and g_gamma is None, "gated chunk_bwd_dv not supported"
    assert not state_v_first, "state_v_first not supported in cuTile port"
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    assert HV == H, "GQA (HV != H) not supported in cuTile port"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BK, BV = _tile_kv(K, V)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dv = torch.empty_like(do)
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(q.device)
    ci_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy_int2(q.device)

    grid = (_cdiv(V, BV), NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_bwd_kernel_dv,
        (
            q,
            k,
            do,
            dv,
            dh,
            cu_arg,
            ci_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            is_varlen,
        ),
    )
    return dv


def chunk_bwd_dqkwg(
    q,
    k,
    v,
    do,
    h,
    dh,
    w=None,
    g=None,
    g_gamma=None,
    dv=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    assert g is None and g_gamma is None, "gated chunk_bwd_dqkwg not supported"
    assert w is None and dv is None, "USE_DW path of chunk_bwd_dqkwg not supported"
    assert not state_v_first, "state_v_first not supported in cuTile port"
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    assert HV == H, "GQA (HV != H) not supported in cuTile port"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK, BV = _tile_kv(K, V)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)

    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy_int(q.device)
    ci_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy_int2(q.device)

    grid = (_cdiv(K, BK), NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_bwd_kernel_dqkwg,
        (
            q,
            k,
            v,
            h,
            do,
            dh,
            dq,
            dk,
            cu_arg,
            ci_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            is_varlen,
        ),
    )
    return dq, dk, None, None


# ===========================================================================
# fla.ops.simple_gla.chunk -- forward/backward orchestration (pure PyTorch)
# ===========================================================================


def chunk_simple_gla_fwd(
    q,
    k,
    v,
    g=None,
    g_gamma=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
        state_v_first=state_v_first,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    return o, ht


def chunk_simple_gla_bwd(
    q,
    k,
    v,
    g,
    g_gamma,
    initial_state,
    do,
    dht,
    scale,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
        state_v_first=state_v_first,
    )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
        state_v_first=state_v_first,
    )
    dq, dk, _, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    dv = chunk_bwd_dv(
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    return dq, dk, dv, dg, dh0


class ChunkSimpleGLAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
    ):
        T = q.shape[1]
        chunk_size = min(64, max(16, _next_power_of_2(T)))

        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        else:
            chunk_indices = None

        if g is not None:
            g = chunk_local_cumsum(
                g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        if g_gamma is not None:
            g_gamma = g_gamma * RCP_LN2
        o, ht = chunk_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, initial_state, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.state_v_first = state_v_first
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        q, k, v, g, g_gamma, initial_state, chunk_indices = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            initial_state=initial_state,
            do=do,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=ctx.state_v_first,
        )
        if g is not None:
            dg = chunk_local_cumsum(
                dg,
                chunk_size=chunk_size,
                reverse=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            ).to(g)
        else:
            dg = None
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None, None, None


@torch.compiler.disable
def chunk_simple_gla(
    q,
    k,
    v,
    g=None,
    g_gamma=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    **kwargs,
):
    if "head_first" in kwargs:
        raise DeprecationWarning("head_first has been removed. Inputs must be in `[B, T, H, ...]` format.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkSimpleGLAFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    return o, final_state


# ===========================================================================
# fla.ops.linear_attn.utils -- running normalization (pure PyTorch)
# ===========================================================================


def _normalization_product_or_eager(o, q, k_cum, scale, cu_seqlens, z_init, reverse):
    if (
        q.dtype == k_cum.dtype == o.dtype == torch.bfloat16
        and q.ndim == k_cum.ndim == o.ndim == 4
        and q.shape == k_cum.shape
        and o.shape[:3] == q.shape[:3]
        and 0 < q.numel() <= 2147483647
        and q.is_cuda
        and q.device == k_cum.device == o.device
        and q.is_contiguous()
        and k_cum.is_contiguous()
        and q.stride()
        == k_cum.stride()
        == (q.shape[1] * q.shape[2] * q.shape[3], q.shape[2] * q.shape[3], q.shape[3], 1)
        and q.data_ptr() % 16 == 0
        and k_cum.data_ptr() % 16 == 0
        and type(scale) in (int, float)
        and -3.4028234663852886e38 <= scale <= 3.4028234663852886e38
        and cu_seqlens is None
        and z_init is None
        and not reverse
        and torch.cuda.get_device_capability(q.device) == (10, 0)
    ):
        product = torch.empty_like(q)
        if product.stride() == q.stride() and product.data_ptr() % 16 == 0:
            ct.launch(
                torch.cuda.current_stream(),
                (_cdiv(q.numel(), 1024),),
                _normalization_product_kernel,
                (q.view(-1), k_cum.view(-1), product.view(-1), float(scale), 1024),
            )
            return product
    return q * scale * k_cum


def _normalization_divide_or_eager(o, s, cu_seqlens, z_init, reverse):
    if (
        o.dtype == s.dtype == torch.bfloat16
        and o.ndim == 4
        and s.shape == o.shape[:-1] + (1,)
        and 0 < o.shape[-1] <= 128 * 65535
        and 0 < o.numel() <= 2147483647
        and o.is_cuda
        and o.device == s.device
        and all(x.is_contiguous() and x.data_ptr() % 16 == 0 for x in (o, s))
        and cu_seqlens is None
        and z_init is None
        and not reverse
        and torch.cuda.get_device_capability(o.device) == (10, 0)
    ):
        out = torch.empty_like(o)
        denom = torch.empty_like(s)
        V = o.shape[-1]
        BC = min(128, 1 << (V - 1).bit_length())
        ct.launch(
            torch.cuda.current_stream(),
            (_cdiv(s.numel(), 8), _cdiv(V, BC)),
            _normalization_divide_kernel,
            (o.view(-1, V), s.view(-1), out.view(-1, V), denom.view(-1), 8, BC),
        )
        return out, denom
    denom = s + 1e-10
    return o / denom, denom


def normalize_with_z_state(o, q, k, scale, z_init, reverse, cu_seqlens):
    """Apply running normalization Z_t = sum_{s<=t} k_s (or sum_{s>=t} for reverse)."""
    return StatefulNormalizeFunction.apply(
        o,
        q,
        k,
        scale,
        z_init,
        reverse,
        cu_seqlens,
        chunk_global_cumsum,
        _normalization_product_or_eager,
        _normalization_divide_or_eager,
    )


# ===========================================================================
# fla.ops.linear_attn.chunk (entry)
# ===========================================================================


@register_impl("fla.linear_attention_chunk", backend="cutile")
@torch.compiler.disable
def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    normalize: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Chunked linear attention. Behavior identical to
    ``fla.ops.linear_attn.chunk.chunk_linear_attn`` (cuTile backend)."""
    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if kv_init is not None and kv_init.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {kv_init.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=kv_init,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    if normalize:
        o, z_state = normalize_with_z_state(o, q, k, scale, z_init, reverse=False, cu_seqlens=cu_seqlens)
        return o, (final_state, z_state) if output_final_state else None
    return o, final_state
