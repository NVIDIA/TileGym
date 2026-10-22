# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools
import logging
from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.suites.fla.cutile._host_utils import input_guard
from tilegym.suites.fla.cutile._host_utils import validate_mamba_chunk_training
from tilegym.suites.fla.cutile.linear_attention_mamba2_fused_chunk import (
    fused_chunk_simple_gla as _fused_chunk_simple_gla,
)

logger = logging.getLogger(__name__)

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# autocast custom fwd/bwd (assume torch >= 2.4 with cuda).
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Device helper (inlined, @ct.function -- NOT @ct.kernel).
# ---------------------------------------------------------------------------


@ct.function
def _exp(x):
    return ct.exp(ct.astype(x, ct.float32))


def _use_chunked_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    g_gamma: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    gv: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    reverse: bool,
    state_v_first: bool,
    cu_seqlens: Optional[torch.LongTensor],
) -> bool:
    return (
        q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and g is not None
        and g_gamma is None
        and gk is None
        and gv is None
        and initial_state is None
        and not output_final_state
        and not reverse
        and not state_v_first
        and cu_seqlens is None
    )


# ======================== cuTile Kernels ========================


@ct.kernel
def fused_recurrent_fwd_kernel(
    q,  # [B*T, H, K]  (token axis flattened on host)
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    g,  # [B*T, H]   scalar log-decay (dummy [1] when not USE_G)
    o,  # [NK, B*T, H, V] (float32)
    h0,  # [N, H, K, V] (float32, dummy [1,1,1,1] when not USE_INITIAL_STATE)
    ht,  # [N, H, K, V] (float32, dummy [1,1,1,1] when not STORE_FINAL_STATE)
    cu_seqlens,  # [N+1] int32 (dummy [1] when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    REVERSE: ConstInt,
    USE_G: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    STORE_FINAL_STATE: ConstInt,
    IS_VARLEN: ConstInt,
):
    i_v = ct.bid(0)
    i_k = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        Tloc = eos - bos
    else:
        bos = i_n * T
        Tloc = T

    o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
    o_v = i_v * BV + ct.arange(BV, dtype=ct.int32)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_h = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        # h0 layout [N, H, K, V]; load [BK, BV] head/state slice.
        b_h0 = ct.gather(
            h0,
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            mask=m_h,
            check_bounds=True,
            padding_value=0.0,
        )
        b_h = b_h + ct.astype(b_h0, ct.float32)

    for step in range(Tloc):
        t = (Tloc - 1 - step) if REVERSE else step
        row = bos + t

        b_q = ct.gather(q, (row, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0)
        b_k = ct.gather(k, (row, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0)
        b_v = ct.gather(v, (row, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0)
        b_q = ct.astype(b_q, ct.float32) * scale
        b_k = ct.astype(b_k, ct.float32)
        b_v = ct.astype(b_v, ct.float32)

        # per-token scalar gate decays the carried state BEFORE the update.
        if USE_G:
            b_g = ct.astype(ct.gather(g, (row, i_h), check_bounds=True, padding_value=0.0), ct.float32)
            b_h = b_h * _exp(b_g)

        # outer-product state update, then weighted sum over K -> [BV]
        b_h = b_h + b_k[:, None] * b_v[None, :]
        b_o = ct.sum(b_h * b_q[:, None], axis=0)

        # o layout [NK, B*T, H, V]
        ct.scatter(
            o,
            (i_k, row, i_h, o_v),
            ct.astype(b_o, o.dtype),
            mask=m_v,
            check_bounds=True,
        )

    if STORE_FINAL_STATE:
        ct.scatter(
            ht,
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            ct.astype(b_h, ht.dtype),
            mask=m_h,
            check_bounds=True,
        )


@ct.kernel
def fused_recurrent_bwd_kernel(
    q,  # [B*T, H, K]
    k,  # [B*T, H, K]
    v,  # [B*T, H, V]
    g,  # [B*T, H]   scalar log-decay (dummy [1] when not USE_G)
    h0,  # [N, H, K, V] (dummy when not USE_INITIAL_STATE)
    do,  # [B*T, H, V]
    dq,  # [NV, B*T, H, K] (float32)
    dk,  # [NV, B*T, H, K] (float32)
    dv,  # [NK, B*T, H, V] (float32)
    dg,  # [NK*NV, B*T, H] (float32, dummy when not USE_G/STORE_DG)
    dht,  # [N, H, K, V] (dummy when not USE_FINAL_STATE_GRADIENT)
    dh0,  # [N, H, K, V] (dummy when not STORE_INITIAL_STATE_GRADIENT)
    cu_seqlens,  # [N+1] int32 (dummy when not IS_VARLEN)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    NV: ConstInt,
    REVERSE: ConstInt,
    USE_G: ConstInt,
    STORE_DG: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    STORE_INITIAL_STATE_GRADIENT: ConstInt,
    USE_FINAL_STATE_GRADIENT: ConstInt,
    IS_VARLEN: ConstInt,
):
    i_v = ct.bid(0)
    i_k = ct.bid(1)
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        Tloc = eos - bos
    else:
        bos = i_n * T
        Tloc = T

    o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
    o_v = i_v * BV + ct.arange(BV, dtype=ct.int32)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    # ---- forward sweep: build state, emit dq ----
    b_h = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        b_h0 = ct.gather(
            h0,
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            mask=m_h,
            check_bounds=True,
            padding_value=0.0,
        )
        b_h = b_h + ct.astype(b_h0, ct.float32)

    for step in range(Tloc):
        t = (Tloc - 1 - step) if REVERSE else step
        row = bos + t

        b_k = ct.gather(k, (row, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0)
        b_v = ct.gather(v, (row, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0)
        b_do = ct.gather(do, (row, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0)
        b_k = ct.astype(b_k, ct.float32)
        b_v = ct.astype(b_v, ct.float32)
        b_do = ct.astype(b_do, ct.float32)

        if USE_G:
            b_g = ct.astype(ct.gather(g, (row, i_h), check_bounds=True, padding_value=0.0), ct.float32)
            b_h = b_h * _exp(b_g)

        b_h = b_h + b_k[:, None] * b_v[None, :]
        b_dq = ct.sum(b_h * b_do[None, :], axis=1) * scale

        ct.scatter(dq, (i_v, row, i_h, o_k), ct.astype(b_dq, dq.dtype), mask=m_k, check_bounds=True)

    # ---- reverse sweep: build state-gradient, emit dk / dv / dg ----
    b_dh = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_FINAL_STATE_GRADIENT:
        b_dht = ct.gather(
            dht,
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            mask=m_h,
            check_bounds=True,
            padding_value=0.0,
        )
        b_dh = b_dh + ct.astype(b_dht, ct.float32)

    if STORE_DG:
        if USE_FINAL_STATE_GRADIENT:
            b_dg_acc = ct.sum(b_h * b_dh)
        else:
            b_dg_acc = ct.full((), 0.0, ct.float32)

    for step in range(Tloc):
        # Reverse direction of the forward sweep. Forward visited t in
        # [0..T-1] (or reversed); the second sweep walks the opposite way.
        t = step if REVERSE else (Tloc - 1 - step)
        row = bos + t

        b_q = ct.gather(q, (row, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0)
        b_k = ct.gather(k, (row, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0)
        b_v = ct.gather(v, (row, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0)
        b_do = ct.gather(do, (row, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0)
        b_q = ct.astype(b_q, ct.float32)
        b_k = ct.astype(b_k, ct.float32)
        b_v = ct.astype(b_v, ct.float32)
        b_do = ct.astype(b_do, ct.float32)

        b_dh = b_dh + (b_q * scale)[:, None] * b_do[None, :]
        b_dk = ct.sum(b_dh * b_v[None, :], axis=1)
        b_dv = ct.sum(b_dh * b_k[:, None], axis=0)

        if USE_G:
            b_g = ct.astype(ct.gather(g, (row, i_h), check_bounds=True, padding_value=0.0), ct.float32)
            if STORE_DG:
                b_dq = ct.astype(
                    ct.gather(dq, (i_v, row, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0), ct.float32
                )
                b_dg_acc = b_dg_acc + ct.sum(b_q * b_dq - b_k * b_dk)
            b_dh = b_dh * _exp(b_g)
            if STORE_DG:
                # dg layout [NK*NV, B*T, H]; the (i_k*NV + i_v) partial is summed host-side.
                ct.scatter(dg, (i_k * NV + i_v, row, i_h), ct.astype(b_dg_acc, dg.dtype), check_bounds=True)

        ct.scatter(dk, (i_v, row, i_h, o_k), ct.astype(b_dk, dk.dtype), mask=m_k, check_bounds=True)
        ct.scatter(dv, (i_k, row, i_h, o_v), ct.astype(b_dv, dv.dtype), mask=m_v, check_bounds=True)

    if STORE_INITIAL_STATE_GRADIENT:
        ct.scatter(
            dh0,
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            ct.astype(b_dh, dh0.dtype),
            mask=m_h,
            check_bounds=True,
        )


# ======================== Wrapper Functions ========================


def fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if g_gamma is not None or gk is not None or gv is not None or state_v_first:
        raise NotImplementedError(
            "The mamba2 implementation only supports the scalar-gate, K-major-state "
            "variant (g_gamma/gk/gv must be None, state_v_first=False).",
        )

    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = min(_next_power_of_2(K), 64), min(_next_power_of_2(V), 64)
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)

    is_varlen = cu_seqlens is not None
    use_initial_state = initial_state is not None
    use_g = g is not None

    if output_final_state:
        ht = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        ht = q.new_empty(1, 1, 1, 1, dtype=torch.float32)

    h0_arg = initial_state if use_initial_state else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else torch.empty(1, device=q.device, dtype=torch.int32)
    BT = B * T
    q3 = q.reshape(BT, H, K)
    k3 = k.reshape(BT, H, K)
    v3 = v.reshape(BT, H, V)
    g2 = g.reshape(BT, H) if use_g else torch.zeros(1, device=q.device, dtype=torch.float32)
    o = q.new_empty(NK, BT, H, V, dtype=torch.float32)

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_recurrent_fwd_kernel,
        (
            q3,
            k3,
            v3,
            g2,
            o,
            h0_arg,
            ht,
            cu_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BK,
            BV,
            int(reverse),
            int(use_g),
            int(use_initial_state),
            int(output_final_state),
            int(is_varlen),
        ),
    )
    o = o.sum(0).reshape(B, T, H, V)
    return o, (ht if output_final_state else None)


def fused_recurrent_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    compute_dg: bool = True,
):
    if g_gamma is not None or gk is not None or gv is not None or state_v_first:
        raise NotImplementedError(
            "The mamba2 implementation only supports the scalar-gate, K-major-state "
            "variant (g_gamma/gk/gv must be None, state_v_first=False).",
        )

    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = min(_next_power_of_2(K), 64), min(_next_power_of_2(V), 64)
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)

    is_varlen = cu_seqlens is not None
    use_initial_state = initial_state is not None
    use_final_state_gradient = dht is not None
    use_g = g is not None
    store_dg = bool(use_g and compute_dg)

    dh0 = torch.empty_like(initial_state) if use_initial_state else q.new_empty(1, 1, 1, 1, dtype=torch.float32)

    h0_arg = initial_state if use_initial_state else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    dht_arg = dht if use_final_state_gradient else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else torch.empty(1, device=q.device, dtype=torch.int32)

    # Flatten (B, T) -> (B*T) for direct token-axis indexing.
    BT = B * T
    q3 = q.reshape(BT, H, K)
    k3 = k.reshape(BT, H, K)
    v3 = v.reshape(BT, H, V)
    do3 = do.reshape(BT, H, V)
    g2 = g.reshape(BT, H) if use_g else torch.zeros(1, device=q.device, dtype=torch.float32)
    dq = q.new_empty(NV, BT, H, K, dtype=torch.float32)
    dk = q.new_empty(NV, BT, H, K, dtype=torch.float32)
    dv = q.new_empty(NK, BT, H, V, dtype=torch.float32)
    if store_dg:
        dg = g.new_empty(NK * NV, BT, H, dtype=torch.float32)
    else:
        dg = torch.zeros(1, device=q.device, dtype=torch.float32)

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_recurrent_bwd_kernel,
        (
            q3,
            k3,
            v3,
            g2,
            h0_arg,
            do3,
            dq,
            dk,
            dv,
            dg,
            dht_arg,
            dh0,
            cu_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BK,
            BV,
            NV,
            int(reverse),
            int(use_g),
            int(store_dg),
            int(use_initial_state),
            int(use_initial_state),  # STORE_INITIAL_STATE_GRADIENT
            int(use_final_state_gradient),
            int(is_varlen),
        ),
    )
    dq = dq.sum(0).reshape(B, T, H, K)
    dk = dk.sum(0).reshape(B, T, H, K)
    dv = dv.sum(0).reshape(B, T, H, V)
    if store_dg:
        dg_out = dg.sum(0).reshape(B, T, H).to(g)
    else:
        dg_out = None
    dh0_out = dh0 if use_initial_state else None
    return dq, dk, dv, dg_out, None, None, dh0_out


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        g_gamma: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        gv: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        state_v_first: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        o, ht = fused_recurrent_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        ctx.state_v_first = state_v_first
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, g_gamma, gk, gv, initial_state, o = ctx.saved_tensors
        dq, dk, dv, dg, dgk, dgv, dh0 = fused_recurrent_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            o=o,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            cu_seqlens=ctx.cu_seqlens,
            state_v_first=ctx.state_v_first,
            compute_dg=ctx.needs_input_grad[3],
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg, None, dgk, dgv, None, dh0, None, None, None, None


def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        gk,
        gv,
        scale,
        initial_state,
        output_final_state,
        reverse,
        state_v_first,
        cu_seqlens,
    )


def fused_recurrent_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    g_gamma: torch.Tensor = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please pre-pack variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if _use_chunked_fast_path(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
    ):
        validate_mamba_chunk_training(g)
        return _fused_chunk_simple_gla(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=None,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
        )

    o, final_state = fused_recurrent(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        state_v_first=state_v_first,
    )
    return o, final_state


# =============================================================================
# Public API.
# =============================================================================


def fused_recurrent_mamba2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused-recurrent Mamba-2 (SSD) forward = simple-GLA with per-token scalar
    log-decay ``g`` (= ``log(alpha)``), shape ``[B, T, H]``.
    q/k/v: ``[B, T, H, K/K/V]``. Returns ``(o, final_state)``.
    """
    return fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


register_impl("fla.linear_attention_mamba2_fused_recurrent", backend="cutile")(fused_recurrent_mamba2)
