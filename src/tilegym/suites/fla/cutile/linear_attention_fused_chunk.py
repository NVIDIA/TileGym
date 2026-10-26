# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.suites.fla.cutile._host_utils import input_guard
from tilegym.suites.fla.cutile._normalization import StatefulNormalizeFunction

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]

# Approximate value of 1/ln(2), used for log/exp base conversion (matches source).
RCP_LN2 = 1.4426950216


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


_DUMMY_CACHE: dict[tuple[str, torch.dtype, tuple[int, ...]], torch.Tensor] = {}


def _dummy(device: torch.device, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """Return an uninitialized placeholder for compile-time-disabled inputs."""
    key = (str(device), dtype, shape)
    value = _DUMMY_CACHE.get(key)
    if value is None:
        value = torch.empty(shape, device=device, dtype=dtype)
        _DUMMY_CACHE[key] = value
    return value


# torch>=2.4 autocast custom fwd/bwd, bound to cuda.
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


# ======================== In-kernel helper functions ========================


@ct.function
def _exp2(x):
    return ct.exp2(ct.astype(x, ct.float32))


@ct.function
def _store_dg_partial(dg_seg, i_k, NV, i_v, i_t, i_h, b_dg_t, m_t, out_dtype, BT: ConstInt):
    """Store the [BT] dg partial for partial-group (i_k*NV + i_v) at chunk i_t.

    Only reached when USE_G (never from the public API). The dg buffer is
    [NK*NV, B*T, H]; here `dg_seg` is already token-sliced to [NK*NV, Tloc, H].
    """
    o_t = i_t * BT + ct.arange(BT, dtype=ct.int32)
    grp = i_k * NV + i_v
    ct.scatter(dg_seg, (grp, o_t, i_h), ct.astype(b_dg_t, out_dtype), mask=m_t, check_bounds=True)


# ======================== cuTile Kernels ========================


@ct.kernel
def fused_chunk_fwd_kernel(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    g,  # [B*T, H]   (dummy [1, 1] when not USE_G)
    g_gamma,  # [H]   (dummy [1] when not USE_G_GAMMA)
    o,  # [NK, B*T, H, V] (NK==1 -> [B*T, H, V])
    h0,  # [N*H, K, V] flat state (dummy [1, 1] when not USE_INITIAL_STATE)
    ht,  # [N*H, K, V] flat state (dummy [1, 1] when not STORE_FINAL_STATE)
    cu_seqlens,  # [N+1] int (dummy [1] when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    NK: ConstInt,
    USE_G: ConstBool,
    USE_G_GAMMA: ConstBool,
    USE_INITIAL_STATE: ConstBool,
    STORE_FINAL_STATE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Fused-chunk forward recurrence.

    grid = (cdiv(V, BV), NK, N*H). Each program owns a (BK, BV) state block and
    sweeps the chunk loop producing the [BT, BV] output for its value column.
    """
    i_v = ct.bid(0)
    i_k = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_n * T
        eos = bos + T
    Tloc = eos - bos
    NT = ct.cdiv(Tloc, BT)

    o_i = ct.arange(BT, dtype=ct.int32)
    # [BT, BT] lower-triangular (causal) mask.
    m_s = o_i[:, None] >= o_i[None, :]

    # decay scaffolding (USE_G_GAMMA computes per-head fixed decay up front)
    b_gamma = ct.zeros((1,), dtype=ct.float32)
    b_g = ct.zeros((BT,), dtype=ct.float32)
    b_gq = ct.ones((BT,), dtype=ct.float32)
    b_gk = ct.ones((BT,), dtype=ct.float32)
    b_gn = ct.zeros((), dtype=ct.float32)
    if USE_G_GAMMA:
        b_gamma = ct.astype(ct.load(g_gamma, index=(i_h,), shape=()), ct.float32)
        b_g = b_gamma * (ct.astype(o_i, ct.float32) + 1.0)
        b_gq = _exp2(b_g)

    # Per-segment token window views (zero-copy, TMA-compatible).
    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    o_seg = o.slice(axis=1 if NK > 1 else 0, start=bos, stop=eos)

    # [BK, BV] running KV state.
    b_h = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        b_h = ct.astype(
            ct.load(h0, index=(i_nh, i_k, i_v), shape=(1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape((BK, BV)),
            ct.float32,
        )

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < Tloc
        last_idx = min(i_t * BT + BT, Tloc) - 1

        # [BT, BK]
        b_q = ct.load(q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_q = ct.astype(ct.astype(b_q, ct.float32) * scale, b_q.dtype)
        # [BT, BK] (k loaded natural, transposed to [BK, BT] in the mma below)
        b_k = ct.load(k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_kt = ct.transpose(b_k)
        # [BT, BV]
        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )

        # [BT, BT] = b_q @ b_k^T
        acc_s = ct.zeros((BT, BT), dtype=ct.float32)
        b_s = ct.mma(b_q, b_kt, acc_s)

        if USE_G:
            g_seg = g.slice(axis=0, start=bos, stop=eos)
            b_g = ct.astype(
                ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)),
                ct.float32,
            )
            b_g = ct.where(m_t, b_g, 0.0)
            b_g_last = ct.load(g_seg, index=(last_idx, i_h), shape=()).astype(ct.float32)
            b_gq = _exp2(b_g)
            b_gk = _exp2(b_g_last - b_g)
            b_gn = _exp2(b_g_last)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, Tloc - i_t * BT)
            b_gk = _exp2(b_g_last - b_g)
            b_gn = _exp2(b_g_last)

        if USE_G or USE_G_GAMMA:
            b_gs = ct.where(
                (m_s & m_t[None, :]),
                _exp2(b_g[:, None] - b_g[None, :]),
                0.0,
            )
            b_s = b_s * b_gs
            # [BT, BV] = b_s @ b_v + (b_q @ b_h) * b_gq
            acc_o = ct.zeros((BT, BV), dtype=ct.float32)
            acc_o = ct.mma(ct.astype(b_s, b_q.dtype), b_v, acc_o)
            acc_h = ct.zeros((BT, BV), dtype=ct.float32)
            acc_h = ct.mma(b_q, ct.astype(b_h, b_q.dtype), acc_h)
            b_o = acc_o + acc_h * b_gq[:, None]
            b_v = ct.astype(ct.astype(b_v, ct.float32) * b_gk[:, None], b_v.dtype)
        else:
            b_s = b_s * ct.astype(m_s & m_t[None, :], ct.float32)
            acc_o = ct.zeros((BT, BV), dtype=ct.float32)
            acc_o = ct.mma(ct.astype(b_s, b_q.dtype), b_v, acc_o)
            acc_h = ct.zeros((BT, BV), dtype=ct.float32)
            acc_h = ct.mma(b_q, ct.astype(b_h, b_q.dtype), acc_h)
            b_o = acc_o + acc_h

        # b_h += b_k^T @ b_v   (state update: [BK, BV])
        if USE_G or USE_G_GAMMA:
            b_h = b_h * b_gn
        acc_h_state = ct.zeros((BK, BV), dtype=ct.float32)
        b_h = b_h + ct.mma(b_kt, b_v, acc_h_state)

        if NK > 1:
            ct.store(o_seg, index=(i_k, i_t, i_h, i_v), tile=ct.astype(b_o, o.dtype).reshape((1, BT, 1, BV)))
        else:
            ct.store(o_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_o, o.dtype).reshape((BT, 1, BV)))

    if STORE_FINAL_STATE:
        ct.store(ht, index=(i_nh, i_k, i_v), tile=ct.astype(b_h, ht.dtype).reshape((1, BK, BV)))


@ct.kernel
def fused_chunk_bwd_kernel(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    g,  # [B*T, H] (dummy when not USE_G)
    g_gamma,  # [H] (dummy when not USE_G_GAMMA)
    do,  # [B*T, H, V]
    dq,  # [NV, B*T, H, K] (NV==1 -> [B*T, H, K])
    dk,  # [NV, B*T, H, K]
    dv,  # [NK, B*T, H, V]
    dg,  # [NK*NV, B*T, H] (dummy when not USE_G)
    h0,  # [N*H, K, V] (dummy when not USE_INITIAL_STATE)
    dht,  # [N*H, K, V] (dummy when not USE_FINAL_STATE)
    dh0,  # [N*H, K, V] (dummy when not USE_INITIAL_STATE)
    cu_seqlens,  # [N+1] int (dummy when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    NK: ConstInt,
    NV: ConstInt,
    USE_G: ConstBool,
    USE_G_GAMMA: ConstBool,
    IS_VARLEN: ConstBool,
    USE_INITIAL_STATE: ConstBool,
    USE_FINAL_STATE: ConstBool,
):
    """Fused-chunk backward recurrence.

    Two passes over the chunk loop: forward sweep computes dq while building the
    [BV, BK] state b_h; reverse sweep computes dk/dv while building the [BK, BV]
    state b_dh. Only the no-gate path is exercised by the public API, but the
    USE_G / USE_G_GAMMA math is ported for parity.
    """
    i_v = ct.bid(0)
    i_k = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_n * T
        eos = bos + T
    Tloc = eos - bos
    NT = ct.cdiv(Tloc, BT)

    o_i = ct.arange(BT, dtype=ct.int32)
    m_s = o_i[:, None] >= o_i[None, :]

    b_gamma = ct.zeros((1,), dtype=ct.float32)
    b_g = ct.zeros((BT,), dtype=ct.float32)
    b_gq = ct.ones((BT,), dtype=ct.float32)
    if USE_G_GAMMA:
        b_gamma = ct.astype(ct.load(g_gamma, index=(i_h,), shape=()), ct.float32)
        b_g = b_gamma * (ct.astype(o_i, ct.float32) + 1.0)
        b_gq = _exp2(b_g)

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dq_seg = dq.slice(axis=1 if NV > 1 else 0, start=bos, stop=eos)

    # [BV, BK] running state for the dq pass.
    b_h = ct.zeros((BV, BK), dtype=ct.float32)
    if USE_INITIAL_STATE:
        # h0 stored [K, V]; load and transpose to [V, K].
        b_h = ct.transpose(
            ct.astype(
                ct.load(h0, index=(i_nh, i_k, i_v), shape=(1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BK, BV)
                ),
                ct.float32,
            )
        )

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < Tloc
        last_idx = min(i_t * BT + BT, Tloc) - 1

        b_k = ct.load(k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        b_vt = ct.transpose(b_v)  # [BV, BT]
        b_do = ct.load(do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )

        # [BT, BT] = (b_do @ b_v^T) * scale
        acc_ds = ct.zeros((BT, BT), dtype=ct.float32)
        b_ds = ct.mma(b_do, b_vt, acc_ds) * scale

        if USE_G or USE_G_GAMMA:
            if USE_G:
                g_seg = g.slice(axis=0, start=bos, stop=eos)
                b_g = ct.astype(
                    ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)),
                    ct.float32,
                )
                b_g = ct.where(m_t, b_g, 0.0)
                b_g_last = ct.load(g_seg, index=(last_idx, i_h), shape=()).astype(ct.float32)
                b_gq = _exp2(b_g)
                b_gk = _exp2(b_g_last - b_g)
                b_gn = _exp2(b_g_last)
            else:
                b_g_last = b_gamma * min(BT, Tloc - i_t * BT)
                b_gk = _exp2(b_g_last - b_g)
                b_gn = _exp2(b_g_last)

            b_gs = ct.where(
                (m_s & m_t[None, :]),
                _exp2(b_g[:, None] - b_g[None, :]),
                0.0,
            )
            b_ds = b_ds * b_gs
            b_q = ct.load(q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            )
            acc_dq = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dq = ct.mma(ct.astype(b_ds, b_k.dtype), b_k, acc_dq)
            acc_dq2 = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dq2 = ct.mma(
                ct.astype(ct.astype(b_do, ct.float32) * b_gq[:, None] * scale, b_k.dtype),
                ct.astype(b_h, b_k.dtype),
                acc_dq2,
            )
            b_dq = acc_dq + acc_dq2
            if USE_G:
                dg_seg = dg.slice(axis=1, start=bos, stop=eos)
                b_dg_t = ct.sum(ct.astype(b_q, ct.float32) * b_dq, axis=1)
                _store_dg_partial(dg_seg, i_k, NV, i_v, i_t, i_h, b_dg_t, m_t, dg.dtype, BT)
            # b_h = b_h * b_gn + b_v^T @ (b_k * b_gk)
            b_h = b_h * b_gn
            b_kg = ct.astype(ct.astype(b_k, ct.float32) * b_gk[:, None], b_k.dtype)
            # Accumulate into a fresh tile before updating the recurrent state.
            acc_h_fwd = ct.zeros((BV, BK), dtype=ct.float32)
            b_h = b_h + ct.mma(b_vt, b_kg, acc_h_fwd)
        else:
            b_ds = b_ds * ct.astype(m_s & m_t[None, :], ct.float32)
            acc_dq = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dq = ct.mma(ct.astype(b_ds, b_k.dtype), b_k, acc_dq)
            acc_dq2 = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dq2 = ct.mma(
                ct.astype(ct.astype(b_do, ct.float32) * scale, b_k.dtype),
                ct.astype(b_h, b_k.dtype),
                acc_dq2,
            )
            b_dq = acc_dq + acc_dq2
            # b_h += b_v^T @ b_k
            # Accumulate into a fresh tile before updating the recurrent state.
            acc_h_fwd = ct.zeros((BV, BK), dtype=ct.float32)
            b_h = b_h + ct.mma(b_vt, b_k, acc_h_fwd)

        if NV > 1:
            ct.store(dq_seg, index=(i_v, i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((1, BT, 1, BK)))
        else:
            ct.store(dq_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))

    # ---- reverse sweep: dk / dv ----
    dk_seg = dk.slice(axis=1 if NV > 1 else 0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=1 if NK > 1 else 0, start=bos, stop=eos)

    b_dh = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_FINAL_STATE:
        b_dh = b_dh + ct.astype(
            ct.load(dht, index=(i_nh, i_k, i_v), shape=(1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape((BK, BV)),
            ct.float32,
        )

    # dg w.r.t. the raw gate: reverse cumsum of the per-token terms, seeded with
    # the final-state contribution sum(h_T * dh_T).
    b_dg_last = ct.zeros((), dtype=ct.float32)
    if USE_G:
        b_dg_last = ct.sum(ct.transpose(b_h) * b_dh)

    for i_c in range(NT):
        i_t = NT - 1 - i_c
        o_t = i_t * BT + o_i
        m_t = o_t < Tloc

        b_q = ct.load(q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_qt = ct.transpose(b_q)  # [BK, BT]
        b_k = ct.load(k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        b_do = ct.load(do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )

        # b_s = b_k @ b_q^T  ; b_ds = b_v @ b_do^T   -> [BT, BT]
        acc_s = ct.zeros((BT, BT), dtype=ct.float32)
        b_s = ct.mma(b_k, b_qt, acc_s)
        acc_ds = ct.zeros((BT, BT), dtype=ct.float32)
        b_ds = ct.mma(b_v, ct.transpose(b_do), acc_ds)

        if USE_G or USE_G_GAMMA:
            if USE_G:
                g_seg = g.slice(axis=0, start=bos, stop=eos)
                last_idx = min(i_t * BT + BT, Tloc) - 1
                b_g = ct.astype(
                    ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)),
                    ct.float32,
                )
                b_g = ct.where(m_t, b_g, 0.0)
                b_g_last = ct.load(g_seg, index=(last_idx, i_h), shape=()).astype(ct.float32)
                b_gq = _exp2(b_g)
                b_gk = _exp2(b_g_last - b_g)
                b_gn = _exp2(b_g_last)
                dg_seg = dg.slice(axis=1, start=bos, stop=eos)
            else:
                b_g_last = b_gamma * min(BT, Tloc - i_t * BT)
                b_gk = _exp2(b_g_last - b_g)
                b_gn = _exp2(b_g_last)

            # b_gs = trans(where(m_s & (m_t & m_t), exp2(b_g[:,None]-b_g[None,:]), 0)) * scale
            b_gs = (
                ct.transpose(
                    ct.where(
                        m_s & (m_t[:, None] & m_t[None, :]),
                        _exp2(b_g[:, None] - b_g[None, :]),
                        0.0,
                    )
                )
                * scale
            )
            b_s = b_s * b_gs
            b_ds = b_ds * b_gs

            # b_dk = b_ds @ b_q + b_v @ b_dh^T * b_gk
            acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dk = ct.mma(ct.astype(b_ds, b_k.dtype), b_q, acc_dk)
            acc_dk2 = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dk2 = ct.mma(b_v, ct.astype(ct.transpose(b_dh), b_v.dtype), acc_dk2)
            b_dk = acc_dk + acc_dk2 * b_gk[:, None]

            if USE_G:
                # b_dg_t = where(m_t, load(dg) - sum(b_k*b_dk, 1), 0); cumulative reduce
                cur_dg = ct.astype(
                    ct.load(dg_seg, index=(i_k * NV + i_v, i_t, i_h), shape=(1, BT, 1)).reshape((BT,)),
                    ct.float32,
                )
                b_dg_t = ct.where(m_t, cur_dg - ct.sum(ct.astype(b_k, ct.float32) * b_dk, axis=1), 0.0)
                b_dg_last = b_dg_last + ct.sum(b_dg_t, axis=0)
                b_dg = b_dg_last + b_dg_t - ct.cumsum(b_dg_t, axis=0)
                _store_dg_partial(dg_seg, i_k, NV, i_v, i_t, i_h, b_dg, m_t, dg.dtype, BT)

            # b_dv = b_s @ b_do + b_k @ b_dh * b_gk
            acc_dv = ct.zeros((BT, BV), dtype=ct.float32)
            acc_dv = ct.mma(ct.astype(b_s, b_do.dtype), b_do, acc_dv)
            acc_dv2 = ct.zeros((BT, BV), dtype=ct.float32)
            acc_dv2 = ct.mma(b_k, ct.astype(b_dh, b_k.dtype), acc_dv2)
            b_dv = acc_dv + acc_dv2 * b_gk[:, None]

            # b_dh = b_dh * b_gn + b_q^T @ (b_do * b_gq * scale)
            # Accumulate into a fresh tile before updating the recurrent state.
            b_do_s = ct.astype(ct.astype(b_do, ct.float32) * b_gq[:, None] * scale, b_do.dtype)
            acc_dh = ct.zeros((BK, BV), dtype=ct.float32)
            b_dh = b_dh * b_gn + ct.mma(b_qt, b_do_s, acc_dh)
        else:
            # mask = trans(m_s & (m_t[:,None] & m_t))
            mask2d = ct.transpose(m_s & (m_t[:, None] & m_t[None, :]))
            maskf = ct.astype(mask2d, ct.float32)
            b_s = ct.astype(b_s * scale * maskf, b_do.dtype)
            b_ds = ct.astype(b_ds * scale * maskf, b_q.dtype)

            # b_dk = b_ds @ b_q + b_v @ b_dh^T
            acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dk = ct.mma(b_ds, b_q, acc_dk)
            acc_dk2 = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dk2 = ct.mma(b_v, ct.astype(ct.transpose(b_dh), b_v.dtype), acc_dk2)
            b_dk = acc_dk + acc_dk2

            # b_dv = b_s @ b_do + b_k @ b_dh
            acc_dv = ct.zeros((BT, BV), dtype=ct.float32)
            acc_dv = ct.mma(b_s, b_do, acc_dv)
            acc_dv2 = ct.zeros((BT, BV), dtype=ct.float32)
            acc_dv2 = ct.mma(b_k, ct.astype(b_dh, b_k.dtype), acc_dv2)
            b_dv = acc_dv + acc_dv2

            # b_dh += b_q^T @ (b_do * scale)
            # Accumulate into a fresh tile before updating the recurrent state.
            b_do_s = ct.astype(ct.astype(b_do, ct.float32) * scale, b_do.dtype)
            acc_dh = ct.zeros((BK, BV), dtype=ct.float32)
            b_dh = b_dh + ct.mma(b_qt, b_do_s, acc_dh)

        if NV > 1:
            ct.store(dk_seg, index=(i_v, i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((1, BT, 1, BK)))
        else:
            ct.store(dk_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))
        if NK > 1:
            ct.store(dv_seg, index=(i_k, i_t, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((1, BT, 1, BV)))
        else:
            ct.store(dv_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((BT, 1, BV)))

    if USE_INITIAL_STATE:
        ct.store(dh0, index=(i_nh, i_k, i_v), tile=ct.astype(b_dh, dh0.dtype).reshape((1, BK, BV)))


_SPLIT_LATENCY = 1


@ct.kernel
def fused_chunk_bwd_kernel_dq(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    do,  # [B*T, H, V]
    dq,  # [NV, B*T, H, K] (NV==1 -> [B*T, H, K])
    h0,  # [N*H, K, V] (dummy when not USE_INITIAL_STATE)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    NV: ConstInt,
    USE_INITIAL_STATE: ConstBool,
    EVEN_T: ConstBool,
):
    i_v = ct.bid(0)
    i_k = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    bos = i_n * T
    eos = bos + T
    NT = ct.cdiv(T, BT)

    o_i = ct.arange(BT, dtype=ct.int32)
    m_s = o_i[:, None] >= o_i[None, :]

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dq_seg = dq.slice(axis=1 if NV > 1 else 0, start=bos, stop=eos)

    # [BV, BK] running state (transposed KV).
    b_h = ct.zeros((BV, BK), dtype=ct.float32)
    if USE_INITIAL_STATE:
        b_h = ct.transpose(
            ct.astype(
                ct.load(h0, index=(i_nh, i_k, i_v), shape=(1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BK, BV)
                ),
                ct.float32,
            )
        )

    for i_t in range(NT):
        b_k = ct.load(
            k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BK))
        b_v = ct.load(
            v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BV))
        b_vt = ct.transpose(b_v)  # [BV, BT]
        b_do = ct.load(
            do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BV))

        # [BT, BT] = (b_do @ b_v^T) * scale
        acc_ds = ct.zeros((BT, BT), dtype=ct.float32)
        b_ds = ct.mma(b_do, b_vt, acc_ds) * scale
        if EVEN_T:
            b_ds = b_ds * ct.astype(m_s, ct.float32)
        else:
            m_t = (i_t * BT + o_i) < T
            b_ds = b_ds * ct.astype(m_s & m_t[None, :], ct.float32)

        # b_dq = b_ds @ b_k + (b_do * scale) @ b_h
        acc_dq = ct.zeros((BT, BK), dtype=ct.float32)
        acc_dq = ct.mma(ct.astype(b_ds, b_k.dtype), b_k, acc_dq)
        acc_dq2 = ct.zeros((BT, BK), dtype=ct.float32)
        acc_dq2 = ct.mma(
            ct.astype(ct.astype(b_do, ct.float32) * scale, b_k.dtype),
            ct.astype(b_h, b_k.dtype),
            acc_dq2,
        )
        b_dq = acc_dq + acc_dq2
        acc_h_fwd = ct.zeros((BV, BK), dtype=ct.float32)
        b_h = b_h + ct.mma(b_vt, b_k, acc_h_fwd)

        if NV > 1:
            ct.store(dq_seg, index=(i_v, i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((1, BT, 1, BK)))
        else:
            ct.store(dq_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))


@ct.kernel
def fused_chunk_bwd_kernel_dkv(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    do,  # [B*T, H, V]
    dk,  # [NV, B*T, H, K] (NV==1 -> [B*T, H, K])
    dv,  # [NK, B*T, H, V] (NK==1 -> [B*T, H, V])
    dht,  # [N*H, K, V] (dummy when not USE_FINAL_STATE)
    dh0,  # [N*H, K, V] (dummy when not USE_INITIAL_STATE)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    NK: ConstInt,
    NV: ConstInt,
    USE_INITIAL_STATE: ConstBool,
    USE_FINAL_STATE: ConstBool,
    EVEN_T: ConstBool,
):
    i_v = ct.bid(0)
    i_k = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    bos = i_n * T
    eos = bos + T
    NT = ct.cdiv(T, BT)

    o_i = ct.arange(BT, dtype=ct.int32)
    m_s = o_i[:, None] >= o_i[None, :]

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dk_seg = dk.slice(axis=1 if NV > 1 else 0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=1 if NK > 1 else 0, start=bos, stop=eos)

    b_dh = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_FINAL_STATE:
        b_dh = b_dh + ct.astype(
            ct.load(dht, index=(i_nh, i_k, i_v), shape=(1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape((BK, BV)),
            ct.float32,
        )

    for i_c in range(NT):
        i_t = NT - 1 - i_c

        b_q = ct.load(
            q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BK))
        b_qt = ct.transpose(b_q)  # [BK, BT]
        b_k = ct.load(
            k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BK))
        b_v = ct.load(
            v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BV))
        b_do = ct.load(
            do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO, latency=_SPLIT_LATENCY
        ).reshape((BT, BV))

        # b_s = b_k @ b_q^T ; b_ds = b_v @ b_do^T  -> [BT, BT]
        acc_s = ct.zeros((BT, BT), dtype=ct.float32)
        b_s = ct.mma(b_k, b_qt, acc_s)
        acc_ds = ct.zeros((BT, BT), dtype=ct.float32)
        b_ds = ct.mma(b_v, ct.transpose(b_do), acc_ds)

        if EVEN_T:
            maskf = ct.astype(ct.transpose(m_s), ct.float32)
        else:
            m_t = (i_t * BT + o_i) < T
            maskf = ct.astype(ct.transpose(m_s & (m_t[:, None] & m_t[None, :])), ct.float32)
        b_s = ct.astype(b_s * scale * maskf, b_do.dtype)
        b_ds = ct.astype(b_ds * scale * maskf, b_q.dtype)

        # b_dk = b_ds @ b_q + b_v @ b_dh^T
        acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
        acc_dk = ct.mma(b_ds, b_q, acc_dk)
        acc_dk2 = ct.zeros((BT, BK), dtype=ct.float32)
        acc_dk2 = ct.mma(b_v, ct.astype(ct.transpose(b_dh), b_v.dtype), acc_dk2)
        b_dk = acc_dk + acc_dk2

        # b_dv = b_s @ b_do + b_k @ b_dh
        acc_dv = ct.zeros((BT, BV), dtype=ct.float32)
        acc_dv = ct.mma(b_s, b_do, acc_dv)
        acc_dv2 = ct.zeros((BT, BV), dtype=ct.float32)
        acc_dv2 = ct.mma(b_k, ct.astype(b_dh, b_k.dtype), acc_dv2)
        b_dv = acc_dv + acc_dv2
        b_do_s = ct.astype(ct.astype(b_do, ct.float32) * scale, b_do.dtype)
        acc_dh = ct.zeros((BK, BV), dtype=ct.float32)
        b_dh = b_dh + ct.mma(b_qt, b_do_s, acc_dh)

        if NV > 1:
            ct.store(dk_seg, index=(i_v, i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((1, BT, 1, BK)))
        else:
            ct.store(dk_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))
        if NK > 1:
            ct.store(dv_seg, index=(i_k, i_t, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((1, BT, 1, BV)))
        else:
            ct.store(dv_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((BT, 1, BV)))

    if USE_INITIAL_STATE:
        ct.store(dh0, index=(i_nh, i_k, i_v), tile=ct.astype(b_dh, dh0.dtype).reshape((1, BK, BV)))


@ct.kernel
def fused_chunk_bwd_reduce_kernel(
    x,  # [NP, M] partials (row-major flat)
    y,  # [M] output (fp32 accumulation, optional final cast)
    N,
    NP: ConstInt,
    BLOCK: ConstInt,
):
    # index is a TILE index: dim0 tile size 1 (-> element p), dim1 tile size
    # BLOCK (-> element i*BLOCK). Pass i, NOT i*BLOCK.
    i = ct.bid(0)
    b_y = ct.zeros((BLOCK,), dtype=ct.float32)
    for p in range(NP):
        b_y = b_y + ct.astype(
            ct.load(x, index=(p, i), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK,)),
            ct.float32,
        )
    ct.store(y, index=(i,), tile=ct.astype(b_y, y.dtype))


def _reduce_partials(
    x_partials: torch.Tensor,
    out_shape,
    np_: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sum ``np_`` partials [NP, *out_shape] via the cuTile reduce kernel.

    Accumulation stays fp32; ``dtype`` only controls the final store so the
    autograd path can fuse its required input-gradient cast into this pass.

    The 1D block store is NOT tail-clamped by cuTile, so ``n`` is rounded up to a
    multiple of BLOCK on BOTH x (zero-padded rows -> harmless in the sum) and y;
    every block load/store is then in-bounds. y is sliced back to n on return.
    """
    n = 1
    for s in out_shape:
        n *= s
    BLOCK = 2048
    n_pad = _cdiv(n, BLOCK) * BLOCK
    x2 = x_partials.reshape(np_, n)
    if n_pad != n:
        x2 = torch.nn.functional.pad(x2, (0, n_pad - n))
    y = torch.empty(n_pad, device=x_partials.device, dtype=dtype)
    grid = (n_pad // BLOCK,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_chunk_bwd_reduce_kernel,
        (x2, y, n, np_, BLOCK),
    )
    return y[:n].reshape(*out_shape)


@ct.kernel
def chunk_global_cumsum_vector_kernel(
    s,  # [B*T, H, S] (or head_first [B*H*T, S] handled by host reshape)
    o,  # [B*T, H, S] output
    cu_seqlens,  # [N+1] int (dummy [1] when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    S: ConstInt,
    BT: ConstInt,
    BS: ConstInt,
    REVERSE: ConstBool,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """Global vector cumulative sum
    (HEAD_FIRST=False layout: [B, T, H, S]).

    grid = (cdiv(S, BS), N*H). Running prefix-sum along the time axis, in
    BT-sized chunks, carrying the running total b_z across chunks.
    """
    i_s = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_n * T
        eos = bos + T
    Tloc = eos - bos
    NT = ct.cdiv(Tloc, BT)

    s_seg = s.slice(axis=0, start=bos, stop=eos)
    o_seg = o.slice(axis=0, start=bos, stop=eos)

    b_z = ct.zeros((BS,), dtype=ct.float32)

    o_i = ct.arange(BT, dtype=ct.int32)
    for i_c in range(NT):
        i_t = (NT - 1 - i_c) if REVERSE else i_c
        m_t = (i_t * BT + o_i) < Tloc

        b_s = ct.astype(
            ct.load(s_seg, index=(i_t, i_h, i_s), shape=(BT, 1, BS), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BS)
            ),
            ct.float32,
        )
        # zero padded tail so it does not pollute the running sum / cumsum
        b_s = ct.where(m_t[:, None], b_s, 0.0)

        if REVERSE:
            b_c = b_z[None, :] + ct.cumsum(b_s, axis=0, reverse=True)
        else:
            b_c = b_z[None, :] + ct.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_c = b_c * scale
        ct.store(o_seg, index=(i_t, i_h, i_s), tile=ct.astype(b_c, o.dtype).reshape((BT, 1, BS)), allow_tma=False)
        b_z = b_z + ct.sum(b_s, axis=0)


@ct.kernel
def chunk_local_cumsum_scalar_kernel(
    s,  # [B*T, H]
    o,  # [B*T, H] output
    cu_seqlens,  # [N+1] int (dummy [1] when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    BT: ConstInt,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """cuTile equivalent of chunk_local_cumsum_scalar_kernel
    (HEAD_FIRST=False layout: [B, T, H], REVERSE=False).

    grid = (N*H,). Prefix-sum along the time axis that restarts at every
    BT-sized chunk boundary of each sequence.
    """
    i_nh = ct.bid(0)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_n * T
        eos = bos + T
    NT = ct.cdiv(eos - bos, BT)

    s_seg = s.slice(axis=0, start=bos, stop=eos)
    o_seg = o.slice(axis=0, start=bos, stop=eos)

    for i_t in range(NT):
        b_s = ct.astype(
            ct.load(s_seg, index=(i_t, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)),
            ct.float32,
        )
        b_o = ct.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_o = b_o * scale
        ct.store(o_seg, index=(i_t, i_h), tile=ct.astype(b_o, o.dtype).reshape((BT, 1)))


# ======================== Host Wrappers ========================


def fused_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    # A 128-wide tile keeps 128-wide heads in one block. Head-dimension
    # masks cover smaller and non-power-of-two dimensions.
    BK = min(max(_next_power_of_2(K), 16), 128)
    BV = min(max(_next_power_of_2(V), 16), 128)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NK = _cdiv(K, BK)
    NV = _cdiv(V, BV)

    o = v.new_empty(NK, *v.shape, dtype=torch.float) if NK > 1 else torch.empty_like(v)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None

    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    v3 = v.reshape(BT_total, H, V)
    o_arg = o.reshape(NK, BT_total, H, V) if NK > 1 else o.reshape(BT_total, H, V)

    use_g = g is not None
    use_g_gamma = g_gamma is not None
    use_init = initial_state is not None
    store_final = output_final_state
    is_varlen = cu_seqlens is not None

    g_arg = g.reshape(BT_total, H) if use_g else _dummy(q.device, q.dtype, (1, 1))
    g_gamma_arg = g_gamma if use_g_gamma else _dummy(q.device, torch.float32, (1,))
    h0_arg = initial_state.reshape(N * H, K, V) if use_init else _dummy(q.device, q.dtype, (1, 1, 1))
    ht_arg = ht.reshape(N * H, K, V) if store_final else _dummy(q.device, q.dtype, (1, 1, 1))
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy(q.device, torch.int32, (1,))

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_chunk_fwd_kernel,
        (
            q3,
            k3,
            v3,
            g_arg,
            g_gamma_arg,
            o_arg,
            h0_arg,
            ht_arg,
            cu_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            NK,
            use_g,
            use_g_gamma,
            use_init,
            store_final,
            is_varlen,
        ),
    )
    if NK > 1:
        o = _reduce_partials(o, v.shape, NK, v.dtype)
    return o, ht


def fused_chunk_bwd(
    q,
    k,
    v,
    g,
    g_gamma,
    do,
    scale,
    initial_state: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    grad_dtypes: tuple[torch.dtype, torch.dtype, torch.dtype] | None = None,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    use_split = g is None and g_gamma is None and cu_seqlens is None
    BT = chunk_size
    BK = min(max(_next_power_of_2(K), 16), 64)
    BV = min(max(_next_power_of_2(V), 16), 64)
    NK, NV = _cdiv(K, BK), _cdiv(V, BV)
    split_bf16_partials = use_split and q.dtype == torch.bfloat16
    partial_qk_dtype = q.dtype if split_bf16_partials else torch.float
    partial_v_dtype = v.dtype if split_bf16_partials else torch.float
    dq = q.new_empty(NV, *q.shape, dtype=partial_qk_dtype) if NV > 1 else torch.empty_like(q)
    dk = k.new_empty(NV, *k.shape, dtype=partial_qk_dtype) if NV > 1 else torch.empty_like(k)
    dv = v.new_empty(NK, *v.shape, dtype=partial_v_dtype) if NK > 1 else torch.empty_like(v)
    dg = g.new_empty(NK * NV, *g.shape, dtype=torch.float) if g is not None else None
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None

    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    v3 = v.reshape(BT_total, H, V)
    do3 = do.reshape(BT_total, H, V)
    dq_arg = dq.reshape(NV, BT_total, H, K) if NV > 1 else dq.reshape(BT_total, H, K)
    dk_arg = dk.reshape(NV, BT_total, H, K) if NV > 1 else dk.reshape(BT_total, H, K)
    dv_arg = dv.reshape(NK, BT_total, H, V) if NK > 1 else dv.reshape(BT_total, H, V)

    use_g = g is not None
    use_g_gamma = g_gamma is not None
    use_init = initial_state is not None
    use_final = dht is not None
    is_varlen = cu_seqlens is not None

    g_arg = g.reshape(BT_total, H) if use_g else _dummy(q.device, q.dtype, (1, 1))
    g_gamma_arg = g_gamma if use_g_gamma else _dummy(q.device, torch.float32, (1,))
    dg_arg = dg.reshape(NK * NV, BT_total, H) if use_g else _dummy(q.device, q.dtype, (1, 1, 1))
    h0_arg = initial_state.reshape(N * H, K, V) if use_init else _dummy(q.device, q.dtype, (1, 1, 1))
    dht_arg = dht.reshape(N * H, K, V) if use_final else _dummy(q.device, q.dtype, (1, 1, 1))
    dh0_arg = dh0.reshape(N * H, K, V) if use_init else _dummy(q.device, q.dtype, (1, 1, 1))
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy(q.device, torch.int32, (1,))

    grid = (NV, NK, N * H)
    if use_split:
        even_t = (T % BT) == 0
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            grid,
            fused_chunk_bwd_kernel_dq,
            (
                q3,
                k3,
                v3,
                do3,
                dq_arg,
                h0_arg,
                float(scale),
                T,
                H,
                K,
                V,
                BT,
                BK,
                BV,
                NV,
                use_init,
                even_t,
            ),
        )
        ct.launch(
            stream,
            grid,
            fused_chunk_bwd_kernel_dkv,
            (
                q3,
                k3,
                v3,
                do3,
                dk_arg,
                dv_arg,
                dht_arg,
                dh0_arg,
                float(scale),
                T,
                H,
                K,
                V,
                BT,
                BK,
                BV,
                NK,
                NV,
                use_init,
                use_final,
                even_t,
            ),
        )
    else:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fused_chunk_bwd_kernel,
            (
                q3,
                k3,
                v3,
                g_arg,
                g_gamma_arg,
                do3,
                dq_arg,
                dk_arg,
                dv_arg,
                dg_arg,
                h0_arg,
                dht_arg,
                dh0_arg,
                cu_arg,
                float(scale),
                T,
                H,
                K,
                V,
                BT,
                BK,
                BV,
                NK,
                NV,
                use_g,
                use_g_gamma,
                is_varlen,
                use_init,
                use_final,
            ),
        )
    if use_split:
        dq_dtype, dk_dtype, dv_dtype = grad_dtypes or (torch.float32, torch.float32, torch.float32)
        if NV > 1:
            dq = _reduce_partials(dq, q.shape, NV, dq_dtype)
            dk = _reduce_partials(dk, k.shape, NV, dk_dtype)
        if NK > 1:
            dv = _reduce_partials(dv, v.shape, NK, dv_dtype)
    else:
        dq = dq.sum(0) if NV > 1 else dq
        dk = dk.sum(0) if NV > 1 else dk
        dv = dv.sum(0) if NK > 1 else dv
    if dg is not None:
        dg = dg.sum(0).to(g)

    return dq, dk, dv, dg, dh0


class FusedChunkFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
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
        cu_seqlens,
    ):
        chunk_size = min(64, max(16, _next_power_of_2(q.shape[1])))
        # Pre-scale by RCP_LN2 so downstream kernels can use exp2 directly.
        if g is not None:
            g = chunk_local_cumsum(g, chunk_size=chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens)
        if g_gamma is not None:
            g_gamma = g_gamma * RCP_LN2
        o, ht = fused_chunk_fwd(
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
        )

        ctx.save_for_backward(q, k, v, g, g_gamma, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, g, g_gamma, initial_state = ctx.saved_tensors

        dq, dk, dv, dg, dh0 = fused_chunk_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            chunk_size=ctx.chunk_size,
            grad_dtypes=(q.dtype, k.dtype, v.dtype),
        )
        if g is not None:
            dg = dg.to(g)
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None


def fused_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if g is not None and g_gamma is not None:
        raise ValueError("Only one of `g` or `g_gamma` should be provided.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = FusedChunkFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
    return o, final_state


def fused_chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    g_gamma: torch.Tensor = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = fused_chunk(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


# ===========================================================================
# Global cumsum (vector) host wrappers
# ===========================================================================


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, _next_power_of_2(S))
    # The D64 suffix scan is latency-bound; the narrower tile cuts its loop-carried
    # cumsum cost, while forward scans and wider feature dims benefit from BT=128.
    BT = 32 if reverse and S == 64 else 128

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)

    # Normalise to a [B*T, H, S] token-major layout for the kernel.
    if head_first:
        # [B, H, T, S] -> [B, T, H, S]
        s_in = s.permute(0, 2, 1, 3).contiguous().reshape(B * T, H, S)
        z_buf = torch.empty(B * T, H, S, device=s.device, dtype=z.dtype)
    else:
        s_in = s.reshape(B * T, H, S)
        z_buf = z.reshape(B * T, H, S)

    has_scale = scale is not None
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy(s.device, torch.int32, (1,))

    grid = (_cdiv(S, BS), N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_global_cumsum_vector_kernel,
        (
            s_in,
            z_buf,
            cu_arg,
            float(scale) if has_scale else 1.0,
            T,
            H,
            S,
            BT,
            BS,
            bool(reverse),
            has_scale,
            is_varlen,
        ),
    )
    if head_first:
        z = z_buf.reshape(B, T, H, S).permute(0, 2, 1, 3).contiguous()
    return z


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) != 3:
        raise ValueError(f"Unsupported input shape {g.shape}, which should be [B, T, H]")
    B, T, H = g.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    has_scale = scale is not None
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy(g.device, torch.int32, (1,))

    ct.launch(
        torch.cuda.current_stream(),
        (N * H,),
        chunk_local_cumsum_scalar_kernel,
        (
            g.reshape(B * T, H),
            o.reshape(B * T, H),
            cu_arg,
            float(scale) if has_scale else 1.0,
            T,
            H,
            chunk_size,
            has_scale,
            is_varlen,
        ),
    )
    return o


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H, D] if `head_first=False` "
            f"or [B, H, T, D] otherwise",
        )


@ct.function
def _normalization_product_stages(q, kc, scale):
    qs = ct.astype(ct.astype(q, ct.float32) * scale, q.dtype)
    product = ct.astype(ct.astype(qs, ct.float32) * ct.astype(kc, ct.float32), q.dtype)
    return qs, product


@ct.kernel
def _normalization_product_kernel(Q, KC, P, SCALE: ConstFloat, BLOCK: ConstInt):
    block = ct.bid(0)
    q = ct.load(Q, index=(block,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    kc = ct.load(KC, index=(block,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    scale = ct.full((), SCALE, dtype=ct.float32)
    _, product = _normalization_product_stages(q, kc, scale)
    ct.store(P, index=(block,), tile=product)


def _normalization_product_or_eager(o, q, k_cum, scale, cu_seqlens, z_init, reverse):
    if (
        q.dtype == k_cum.dtype == torch.bfloat16
        and q.shape == k_cum.shape
        and 0 < q.numel() <= 2147483647
        and q.is_contiguous()
        and k_cum.is_contiguous()
        and q.data_ptr() % 16 == 0
        and k_cum.data_ptr() % 16 == 0
        and cu_seqlens is None
        and z_init is None
        and torch.cuda.get_device_capability(q.device)[0] == 10
    ):
        product = torch.empty_like(q)
        ct.launch(
            torch.cuda.current_stream(),
            (_cdiv(q.numel(), 1024),),
            _normalization_product_kernel,
            (q.view(-1), k_cum.view(-1), product.view(-1), float(scale), 1024),
        )
        return product
    return q * scale * k_cum


@ct.function
def _normalization_backward_stages(dd, q, kc, scale):
    # Keep the BF16 rounding after each product used by the eager backward.
    ds = ct.astype(ct.astype(dd, ct.float32) * scale, dd.dtype)
    dq = ct.astype(ct.astype(ds, ct.float32) * ct.astype(kc, ct.float32), q.dtype)
    dk = ct.astype(ct.astype(ds, ct.float32) * ct.astype(q, ct.float32), q.dtype)
    return ds, dq, dk


@ct.kernel
def _normalization_backward_products_kernel(DD, Q, KC, DQ, DK, SCALE: ConstFloat, K: ConstInt, BLOCK: ConstInt):
    idx = ct.bid(0) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    q = ct.gather(Q, (idx,), check_bounds=True, padding_value=0)
    kc = ct.gather(KC, (idx,), check_bounds=True, padding_value=0)
    dd = ct.gather(DD, (idx // K,), check_bounds=True, padding_value=0)
    _, dq, dk = _normalization_backward_stages(dd, q, kc, ct.full((), SCALE, dtype=ct.float32))
    ct.scatter(DQ, (idx,), dq, check_bounds=True)
    ct.scatter(DK, (idx,), dk, check_bounds=True)


def _normalization_backward_products_or_eager(ddenom, q, k_cum, scale, cu_seqlens, has_z_init):
    if (
        q.dtype == k_cum.dtype == ddenom.dtype == torch.bfloat16
        and q.shape == k_cum.shape
        and q.ndim == 4
        and q.shape[-1] > 0
        and ddenom.shape == q.shape[:-1] + (1,)
        and 0 < q.numel() <= 2147483647
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k_cum, ddenom))
        and cu_seqlens is None
        and not has_z_init
        and torch.cuda.get_device_capability(q.device)[0] == 10
    ):
        dq = torch.empty_like(q)
        dk = torch.empty_like(q)
        ct.launch(
            torch.cuda.current_stream(),
            (_cdiv(q.numel(), 1024),),
            _normalization_backward_products_kernel,
            (ddenom.view(-1), q.view(-1), k_cum.view(-1), dq.view(-1), dk.view(-1), float(scale), q.shape[-1], 1024),
        )
        return dq, dk
    return ddenom * scale * k_cum, ddenom * scale * q


def normalize_with_z_state(
    o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    z_init: torch.Tensor | None,
    reverse: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        None,
        _normalization_backward_products_or_eager,
    )


# ===========================================================================
# Public entry point
# ===========================================================================


@torch.compiler.disable
def fused_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    normalize: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor): queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        scale (Optional[float]): Scale factor for linear attention scores.
            Defaults to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor | tuple]): Initial recurrent state.
            When `normalize=False`, a tensor of shape `[B, H, K, V]`. When
            `normalize=True`, a tuple `(kv_state, z_state)`.
        output_final_state (Optional[bool]): Whether to output the final state.
        normalize (bool): Whether to normalize the output. Default: `True`.
        cu_seqlens (torch.LongTensor): Cumulative sequence lengths of shape `[N+1]`.

    Returns:
        o (torch.Tensor): Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor | tuple | None): Mirrors `initial_state`.
    """
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
        scale = q.shape[-1] ** -0.5
    o, final_state = fused_chunk_simple_gla(
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


# --- TileGym dispatch registration ---------------------------------------
register_impl("fla.linear_attention_fused_chunk", backend="cutile")(fused_chunk_linear_attn)
