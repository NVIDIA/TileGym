# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.suites.fla.cutile._host_utils import _segmented_arange
from tilegym.suites.fla.cutile._host_utils import input_guard
from tilegym.suites.fla.cutile._host_utils import validate_mamba_chunk_training
from tilegym.suites.fla.cutile._tuning import hint_configs
from tilegym.suites.fla.cutile._tuning import tune_launch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]

# Approximate value of 1/ln(2), used for log/exp base conversion (matches source).
RCP_LN2 = 1.4426950216


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Cached dummy placeholders for the unused-feature kernel arguments.
#
# The public Mamba-2 entry always drives a single feature combination (USE_G,
# no g_gamma / initial / final / varlen), so the kernels still receive small
# zero placeholders for inactive arguments. These tensors are not written by
# the kernels, so cache them by device, dtype and shape.
# ---------------------------------------------------------------------------
_dummy_cache: dict = {}


def _dummy(shape, device, dtype):
    key = (tuple(shape), device, dtype)
    t = _dummy_cache.get(key)
    if t is None:
        t = torch.zeros(shape, device=device, dtype=dtype)
        _dummy_cache[key] = t
    return t


def _autotuned_launch(kernel, cache_key, grid, args, occ_choices=(1,), nww_choices=(4, 8), timeout=60):
    choices, warps = tuple(occ_choices), tuple(nww_choices)
    tune_launch(
        kernel,
        torch.cuda.current_stream(),
        lambda c: grid,
        lambda c: args,
        lambda: hint_configs(choices, warps),
        cache_key,
        timeout=timeout,
        config_key=("hints", choices, warps),
    )


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

    Only reached when USE_G. The dg buffer is [NK*NV, B*T, H]; here `dg_seg` is
    already token-sliced to [NK*NV, Tloc, H].
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
    h0,  # [N*H, K, V] flat state (dummy [1, 1, 1] when not USE_INITIAL_STATE)
    ht,  # [N*H, K, V] flat state (dummy [1, 1, 1] when not STORE_FINAL_STATE)
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
            ct.load(h0, index=(i_nh, i_k, i_v), shape=(1, BK, BV)).reshape((BK, BV)),
            ct.float32,
        )

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < Tloc
        last_idx = min(i_t * BT + BT, Tloc) - 1

        # [BT, BK]
        b_q = ct.load(q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK)).reshape((BT, BK))
        b_q = ct.astype(ct.astype(b_q, ct.float32) * scale, b_q.dtype)
        # [BT, BK] (k loaded natural, transposed to [BK, BT] in the mma below)
        b_k = ct.load(k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK)).reshape((BT, BK))
        b_kt = ct.transpose(b_k)
        # [BT, BV]
        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV)).reshape((BT, BV))

        # [BT, BT] = b_q @ b_k^T
        acc_s = ct.zeros((BT, BT), dtype=ct.float32)
        b_s = ct.mma(b_q, b_kt, acc_s)

        if USE_G:
            g_seg = g.slice(axis=0, start=bos, stop=eos)
            b_g = ct.astype(ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1)).reshape((BT,)), ct.float32)
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

        # b_h (+= or decayed) b_k^T @ b_v   (state update: [BK, BV])
        if USE_G or USE_G_GAMMA:
            b_h = b_h * b_gn
        b_h = ct.mma(b_kt, b_v, b_h)

        if NK > 1:
            ct.store(o_seg, index=(i_k, i_t, i_h, i_v), tile=ct.astype(b_o, o.dtype).reshape((1, BT, 1, BV)))
        else:
            ct.store(o_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_o, o.dtype).reshape((BT, 1, BV)))

    if STORE_FINAL_STATE:
        ct.store(ht, index=(i_nh, i_k, i_v), tile=ct.astype(b_h, ht.dtype).reshape((1, BK, BV)))


@ct.kernel
def fused_chunk_bwd_dq_kernel(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    g,  # [B*T, H] (dummy when not USE_G)
    g_gamma,  # [H] (dummy when not USE_G_GAMMA)
    do,  # [B*T, H, V]
    dq,  # [NV, B*T, H, K] (NV==1 -> [B*T, H, K])
    dg,  # [NK*NV, B*T, H] (dummy when not USE_G)
    h0,  # [N*H, K, V] (dummy when not USE_INITIAL_STATE)
    h_final,  # [N*H*NK*NV, BV, BK] scratch: final b_h state of the dq sweep
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
):
    """dq forward sweep of the mamba2 fused-chunk backward (split kernel 1/2).

    Computes dq and the dg partial (USE_G) while building the [BV, BK] running
    state b_h. The final b_h is stored to ``h_final`` so the dk/dv kernel can
    seed its dg running-last scalar (USE_G). Holding only ONE running state per
    kernel (vs the monolithic two-pass kernel's b_h + b_dh + all score tiles)
    roughly halves live registers/SMEM, raising blocks/SM and SM utilisation.
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
            ct.astype(ct.load(h0, index=(i_nh, i_k, i_v), shape=(1, BK, BV)).reshape((BK, BV)), ct.float32)
        )

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < Tloc
        last_idx = min(i_t * BT + BT, Tloc) - 1

        b_k = ct.load(k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK)).reshape((BT, BK))
        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV)).reshape((BT, BV))
        b_vt = ct.transpose(b_v)  # [BV, BT]
        b_do = ct.load(do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV)).reshape((BT, BV))

        # [BT, BT] = (b_do @ b_v^T) * scale
        acc_ds = ct.zeros((BT, BT), dtype=ct.float32)
        b_ds = ct.mma(b_do, b_vt, acc_ds) * scale

        if USE_G or USE_G_GAMMA:
            if USE_G:
                g_seg = g.slice(axis=0, start=bos, stop=eos)
                b_g = ct.astype(ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1)).reshape((BT,)), ct.float32)
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
            b_q = ct.load(q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK)).reshape((BT, BK))
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
            b_h = ct.mma(b_vt, b_kg, b_h)
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
            b_h = ct.mma(b_vt, b_k, b_h)

        if NV > 1:
            ct.store(dq_seg, index=(i_v, i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((1, BT, 1, BK)))
        else:
            ct.store(dq_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))

    # Persist the final b_h state for the dk/dv kernel (USE_G dg seed).
    if USE_G:
        i_hf = i_nh * (NK * NV) + i_k * NV + i_v
        ct.store(h_final, index=(i_hf, 0, 0), tile=b_h.reshape((1, BV, BK)))


@ct.kernel
def fused_chunk_bwd_dkv_kernel(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    g,  # [B*T, H] (dummy when not USE_G)
    g_gamma,  # [H] (dummy when not USE_G_GAMMA)
    do,  # [B*T, H, V]
    dk,  # [NV, B*T, H, K]
    dv,  # [NK, B*T, H, V]
    dg,  # [NK*NV, B*T, H] (dummy when not USE_G)
    h_final,  # [N*H*NK*NV, BV, BK] final b_h from the dq kernel (USE_G dg seed)
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
    """dk/dv reverse sweep of the mamba2 fused-chunk backward (split kernel 2/2).

    Computes dk, dv (and the dg cumsum tail for USE_G) while building the
    [BK, BV] running state b_dh. The USE_G dg running-last scalar is seeded from
    the final b_h state stored by the dq kernel.
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
    if USE_G_GAMMA:
        b_gamma = ct.astype(ct.load(g_gamma, index=(i_h,), shape=()), ct.float32)
        b_g = b_gamma * (ct.astype(o_i, ct.float32) + 1.0)

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dk_seg = dk.slice(axis=1 if NV > 1 else 0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=1 if NK > 1 else 0, start=bos, stop=eos)

    # [BK, BV] running state for the reverse pass.
    b_dh = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_FINAL_STATE:
        b_dh = b_dh + ct.astype(ct.load(dht, index=(i_nh, i_k, i_v), shape=(1, BK, BV)).reshape((BK, BV)), ct.float32)

    # dg running-last scalar (only meaningful for USE_G), seeded from the final
    # b_h state of the dq kernel: b_dg_last = sum(trans(b_h) * b_dh).
    b_dg_last = ct.zeros((1,), dtype=ct.float32)
    if USE_G:
        i_hf = i_nh * (NK * NV) + i_k * NV + i_v
        b_h_final = ct.load(h_final, index=(i_hf, 0, 0), shape=(1, BV, BK)).reshape((BV, BK))
        b_dg_last = ct.sum(ct.transpose(b_h_final) * b_dh)

    for i_c in range(NT):
        i_t = NT - 1 - i_c
        o_t = i_t * BT + o_i
        m_t = o_t < Tloc
        last_idx = min(i_t * BT + BT, Tloc) - 1

        b_q = ct.load(q_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK)).reshape((BT, BK))
        b_qt = ct.transpose(b_q)  # [BK, BT]
        b_k = ct.load(k_seg, index=(i_t, i_h, i_k), shape=(BT, 1, BK)).reshape((BT, BK))
        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV)).reshape((BT, BV))
        b_do = ct.load(do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV)).reshape((BT, BV))

        # b_s = b_k @ b_q^T  ; b_ds = b_v @ b_do^T   -> [BT, BT]
        acc_s = ct.zeros((BT, BT), dtype=ct.float32)
        b_s = ct.mma(b_k, b_qt, acc_s)
        acc_ds = ct.zeros((BT, BT), dtype=ct.float32)
        b_ds = ct.mma(b_v, ct.transpose(b_do), acc_ds)

        if USE_G or USE_G_GAMMA:
            if USE_G:
                g_seg = g.slice(axis=0, start=bos, stop=eos)
                dg_seg = dg.slice(axis=1, start=bos, stop=eos)
                b_g = ct.astype(ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1)).reshape((BT,)), ct.float32)
                b_g = ct.where(m_t, b_g, 0.0)
                b_g_last = ct.load(g_seg, index=(last_idx, i_h), shape=()).astype(ct.float32)
                b_gq = _exp2(b_g)
                b_gk = _exp2(b_g_last - b_g)
                b_gn = _exp2(b_g_last)
            else:
                b_g_last = b_gamma * min(BT, Tloc - i_t * BT)
                b_gq = _exp2(b_g)
                b_gk = _exp2(b_g_last - b_g)
                b_gn = _exp2(b_g_last)

            # b_gs = trans(where(m_s & (m_t[:,None] & m_t), exp2(b_g[:,None]-b_g[None,:]), 0)) * scale
            b_gs = (
                ct.transpose(
                    ct.where(
                        (m_s & (m_t[:, None] & m_t[None, :])),
                        _exp2(b_g[:, None] - b_g[None, :]),
                        0.0,
                    )
                )
                * scale
            )

            b_s = b_s * b_gs
            b_ds = b_ds * b_gs

            # b_dk = b_ds @ b_q + (b_v @ b_dh^T) * b_gk
            acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dk = ct.mma(ct.astype(b_ds, b_k.dtype), b_q, acc_dk)
            acc_dk2 = ct.zeros((BT, BK), dtype=ct.float32)
            acc_dk2 = ct.mma(b_v, ct.astype(ct.transpose(b_dh), b_v.dtype), acc_dk2)
            b_dk = acc_dk + acc_dk2 * b_gk[:, None]

            if USE_G:
                # dg_t = where(m_t, dg_prev - sum(b_k * b_dk, 1), 0); then prefix.
                dg_prev = ct.astype(
                    ct.load(dg_seg, index=(i_k * NV + i_v, i_t, i_h), shape=(1, BT, 1)).reshape((BT,)),
                    ct.float32,
                )
                b_dg_t = ct.where(m_t, dg_prev - ct.sum(ct.astype(b_k, ct.float32) * b_dk, axis=1), 0.0)
                b_dg_last = b_dg_last + ct.sum(b_dg_t, axis=0)
                b_dg = b_dg_last + b_dg_t - ct.cumsum(b_dg_t, axis=0)

            # b_dv = b_s @ b_do + (b_k @ b_dh) * b_gk
            acc_dv = ct.zeros((BT, BV), dtype=ct.float32)
            acc_dv = ct.mma(ct.astype(b_s, b_do.dtype), b_do, acc_dv)
            acc_dv2 = ct.zeros((BT, BV), dtype=ct.float32)
            acc_dv2 = ct.mma(b_k, ct.astype(b_dh, b_k.dtype), acc_dv2)
            b_dv = acc_dv + acc_dv2 * b_gk[:, None]

            # b_dh = b_dh * b_gn + b_q^T @ (b_do * b_gq * scale)
            b_dh = b_dh * b_gn
            acc_dh = ct.zeros((BK, BV), dtype=ct.float32)
            b_dh = b_dh + ct.mma(
                b_qt,
                ct.astype(ct.astype(b_do, ct.float32) * b_gq[:, None] * scale, b_do.dtype),
                acc_dh,
            )

            if USE_G:
                _store_dg_partial(dg_seg, i_k, NV, i_v, i_t, i_h, b_dg, m_t, dg.dtype, BT)
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
            acc_dh = ct.zeros((BK, BV), dtype=ct.float32)
            b_dh = b_dh + ct.mma(b_qt, ct.astype(ct.astype(b_do, ct.float32) * scale, b_do.dtype), acc_dh)

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
def chunk_local_cumsum_scalar_kernel(
    s,  # [B*T, H] input scalar gate
    o,  # [B*T, H] output (chunk-local cumsum)
    cu_seqlens,  # [N+1] int (dummy [1] when not IS_VARLEN)
    chunk_indices,  # [NTtot, 2] int (dummy [1, 1] when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    BT: ConstInt,
    REVERSE: ConstBool,
    HAS_SCALE: ConstBool,
    IS_VARLEN: ConstBool,
):
    """cuTile equivalent of the non-head-first chunk_local_cumsum_scalar_kernel.

    grid = (NT, B*H). Chunk-LOCAL cumsum (no carry across chunks) along the time
    axis for the scalar gate g of shape [B, T, H] (head_first=False layout).
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_t = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
    Tloc = eos - bos

    s_seg = s.slice(axis=0, start=bos, stop=eos)
    o_seg = o.slice(axis=0, start=bos, stop=eos)

    o_i = ct.arange(BT, dtype=ct.int32)
    m_t = (i_t * BT + o_i) < Tloc

    # [BT, 1] -> [BT]
    b_s = ct.astype(ct.load(s_seg, index=(i_t, i_h), shape=(BT, 1)).reshape((BT,)), ct.float32)
    # zero out the padded tail so it does not pollute the running sum / cumsum
    b_s = ct.where(m_t, b_s, 0.0)

    b_o = ct.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = ct.sum(b_s, axis=0)
        b_o = -b_o + b_z + b_s
    if HAS_SCALE:
        b_o = b_o * scale

    ct.store(o_seg, index=(i_t, i_h), tile=ct.astype(b_o, o.dtype).reshape((BT, 1)), allow_tma=False)


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
    BK = min(max(_next_power_of_2(K), 16), 128)
    BV = min(max(_next_power_of_2(V), 16), 128)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NK = _cdiv(K, BK)
    if cu_seqlens is None and BV == 128 and torch.cuda.get_device_capability(q.device)[0] == 10:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        if NK * _cdiv(V, BV) * N * H * 2 <= num_sms:
            BV = 64
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

    g_arg = g.reshape(BT_total, H) if use_g else _dummy((1, 1), q.device, q.dtype)
    g_gamma_arg = g_gamma if use_g_gamma else _dummy((1,), q.device, torch.float32)
    h0_arg = initial_state.reshape(N * H, K, V) if use_init else _dummy((1, 1, 1), q.device, q.dtype)
    ht_arg = ht.reshape(N * H, K, V) if store_final else _dummy((1, 1, 1), q.device, q.dtype)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy((1,), q.device, torch.int32)

    grid = (NV, NK, N * H)
    fwd_args = (
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
    )
    # Launch-hint autotune (occupancy x num_worker_warps), shape-keyed so it is
    # tuned once per shape and the default config (occ=1, nww=4) equals the base
    # @ct.kernel launch -- non-improving shapes therefore keep original perf.
    # Helps low-head / long-T grids (e.g. N*H small, NT large) raise SM
    # utilisation by spreading the per-chunk MMA across more worker warps.
    fwd_key = (
        "fused_chunk_fwd",
        H,
        K,
        V,
        BT,
        BK,
        BV,
        NK,
        N,
        T,
        use_g,
        use_g_gamma,
        use_init,
        store_final,
        is_varlen,
        q.dtype,
    )
    _autotuned_launch(
        fused_chunk_fwd_kernel,
        fwd_key,
        grid,
        fwd_args,
        occ_choices=(1, 2, 4),
        nww_choices=(4, 8),
    )
    if NK > 1:
        o = o.sum(0).to(v)
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
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BT = chunk_size
    NT_est = _cdiv(T, BT) if cu_seqlens is None else 1 << 30
    _spill_bound = (K * V >= 256 * 256) or (NT_est >= 64)
    _cap = 64 if _spill_bound else 128
    BK = min(max(_next_power_of_2(K), 16), _cap)
    BV = min(max(_next_power_of_2(V), 16), _cap)
    NK, NV = _cdiv(K, BK), _cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float) if NV > 1 else torch.empty_like(q)
    dk = k.new_empty(NV, *k.shape, dtype=torch.float) if NV > 1 else torch.empty_like(k)
    dv = v.new_empty(NK, *v.shape, dtype=torch.float) if NK > 1 else torch.empty_like(v)
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

    g_arg = g.reshape(BT_total, H) if use_g else torch.zeros(1, 1, device=q.device, dtype=q.dtype)
    g_gamma_arg = g_gamma if use_g_gamma else torch.zeros(1, device=q.device, dtype=torch.float32)
    dg_arg = dg.reshape(NK * NV, BT_total, H).zero_() if use_g else torch.zeros(1, 1, 1, device=q.device, dtype=q.dtype)
    h0_arg = initial_state.reshape(N * H, K, V) if use_init else torch.zeros(1, 1, 1, device=q.device, dtype=q.dtype)
    dht_arg = dht.reshape(N * H, K, V) if use_final else torch.zeros(1, 1, 1, device=q.device, dtype=q.dtype)
    dh0_arg = dh0.reshape(N * H, K, V) if use_init else torch.zeros(1, 1, 1, device=q.device, dtype=q.dtype)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else torch.zeros(1, device=q.device, dtype=torch.int32)

    grid = (NV, NK, N * H)

    # Scratch holding the final b_h state of the dq sweep so the dk/dv kernel can
    # seed its dg running-last scalar (USE_G only). Tiny ([N*H*NK*NV, BV, BK]).
    if use_g:
        h_final = q.new_empty(N * H * NK * NV, BV, BK, dtype=torch.float)
    else:
        h_final = torch.zeros(1, 1, 1, device=q.device, dtype=torch.float)

    # --- split kernel 1/2: dq forward sweep (builds + persists b_h) ---
    dq_args = (
        q3,
        k3,
        v3,
        g_arg,
        g_gamma_arg,
        do3,
        dq_arg,
        dg_arg,
        h0_arg,
        h_final,
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
    )
    dq_key = (
        "fused_chunk_bwd_dq",
        N,
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
        q.dtype,
    )
    _autotuned_launch(fused_chunk_bwd_dq_kernel, dq_key, grid, dq_args)

    # --- split kernel 2/2: dk/dv reverse sweep (consumes b_h) ---
    dkv_args = (
        q3,
        k3,
        v3,
        g_arg,
        g_gamma_arg,
        do3,
        dk_arg,
        dv_arg,
        dg_arg,
        h_final,
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
    )
    dkv_key = (
        "fused_chunk_bwd_dkv",
        N,
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
        q.dtype,
    )
    _autotuned_launch(fused_chunk_bwd_dkv_kernel, dkv_key, grid, dkv_args)
    dq = dq.sum(0) if NV > 1 else dq
    dk = dk.sum(0) if NV > 1 else dk
    dv = dv.sum(0) if NK > 1 else dv
    if dg is not None:
        dg = dg.sum(0).to(g)

    return dq, dk, dv, dg, dh0


# ===========================================================================
# chunk_local_cumsum machinery (needed when g is not None)
# ===========================================================================


def _prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.diff(cu_seqlens)


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_counts = (_prepare_lens(cu_seqlens) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def chunk_local_cumsum_scalar(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    output_dtype=torch.float,
    chunk_indices=None,
):
    B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_org = g
    g_out = torch.empty_like(g, dtype=output_dtype or g.dtype)

    BT_total = B * T
    s_arg = g_org.reshape(BT_total, H)
    o_arg = g_out.reshape(BT_total, H)

    has_scale = scale is not None
    is_varlen = cu_seqlens is not None
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else _dummy((1,), g.device, torch.int32)
    ci_arg = chunk_indices.to(torch.int32) if is_varlen else _dummy((1, 1), g.device, torch.int32)

    grid = (NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        chunk_local_cumsum_scalar_kernel,
        (
            s_arg,
            o_arg,
            cu_arg,
            ci_arg,
            float(scale) if has_scale else 1.0,
            T,
            H,
            BT,
            bool(reverse),
            has_scale,
            is_varlen,
        ),
    )
    return g_out


@input_guard
def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
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
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(f"Unsupported input shape {g.shape}")


# ===========================================================================
# Autograd Function + public entry points
# ===========================================================================


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
            g = chunk_local_cumsum(
                g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
            )
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
# Public entry point
# ===========================================================================


@torch.compiler.disable
def fused_chunk_mamba2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused-chunk Mamba-2 (SSD) forward = simple-GLA with per-token scalar
    log-decay ``g`` (= ``log(alpha)``), shape ``[B, T, H]``.
    q/k/v: ``[B, T, H, K/K/V]``. Returns ``(o, final_state)``.
    """
    validate_mamba_chunk_training(g)
    return fused_chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


# --- TileGym dispatch registration ---------------------------------------
register_impl("fla.linear_attention_mamba2_fused_chunk", backend="cutile")(fused_chunk_mamba2)
