# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.suites.fla.cutile._host_utils import input_guard

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# =============================================================================
# L2 normalization (used only when use_qk_l2norm_in_kernel=True).
# One program per row of the flattened [T, D] view, gathering the whole
# D-vector.
# =============================================================================


@ct.kernel
def l2norm_fwd_kernel_ct(
    x,  # [T, D]
    y,  # [T, D] output
    rstd,  # [T] output (float32)
    eps: ConstFloat,
    D: ConstInt,
    BD: ConstInt,
):
    i_t = ct.bid(0)
    cols = ct.arange(BD, dtype=ct.int32)
    mask = cols < D
    b_x = ct.gather(x, (i_t, cols), mask=mask, check_bounds=True, padding_value=0.0)
    b_x = ct.astype(b_x, ct.float32)
    b_rstd = 1.0 / ct.sqrt(ct.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    ct.scatter(y, (i_t, cols), ct.astype(b_y, y.dtype), mask=mask, check_bounds=True)
    ct.scatter(rstd, (i_t,), b_rstd, check_bounds=True)


@ct.kernel
def l2norm_bwd_kernel_ct(
    y,  # [T, D]
    rstd,  # [T] (float32)
    dy,  # [T, D]
    dx,  # [T, D] output
    D: ConstInt,
    BD: ConstInt,
):
    i_t = ct.bid(0)
    cols = ct.arange(BD, dtype=ct.int32)
    mask = cols < D
    b_y = ct.astype(ct.gather(y, (i_t, cols), mask=mask, check_bounds=True, padding_value=0.0), ct.float32)
    b_dy = ct.astype(ct.gather(dy, (i_t, cols), mask=mask, check_bounds=True, padding_value=0.0), ct.float32)
    b_rstd = ct.astype(ct.gather(rstd, (i_t,), check_bounds=True, padding_value=0.0), ct.float32)
    b_dx = b_dy * b_rstd - ct.sum(b_dy * b_y) * b_y * b_rstd
    ct.scatter(dx, (i_t, cols), ct.astype(b_dx, dx.dtype), mask=mask, check_bounds=True)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
    x_shape_og = x.shape
    x = x.contiguous().view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    T, D = x.shape[0], x.shape[-1]
    BD = _next_power_of_2(D)
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)

    grid = (T,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        l2norm_fwd_kernel_ct,
        (x, y, rstd, float(eps), D, BD),
    )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6):
    y_shape_og = y.shape
    y = y.contiguous().view(-1, dy.shape[-1])
    dy = dy.contiguous().view(-1, dy.shape[-1])
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    BD = _next_power_of_2(D)

    grid = (T,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        l2norm_bwd_kernel_ct,
        (y, rstd.contiguous().view(-1), dy, dx, D, BD),
    )
    return dx.view(y_shape_og)


@ct.kernel
def fused_recurrent_delta_rule_fwd_kernel_ct(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    u,  # [B, T, H, V] output
    beta,  # [B, T, H] scalar, or [B, T, H, V] headwise
    o,  # [NK, B, T, H, V] output
    h0,  # [N, H, K, V] initial state (dummy [1,1,1,1] when unused)
    ht,  # [N, H, K, V] final state (dummy [1,1,1,1] when unused)
    cu_seqlens,  # [N+1] int32 (dummy [1] when not varlen)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    STORE_FINAL_STATE: ConstInt,
    IS_BETA_HEADWISE: ConstInt,
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
        # varlen inputs are pre-flattened to [1, total_T, H, *]: the batch dim is
        # always 0 and the per-sequence offset lives in the time index (gt = bos + t).
        b = 0
    else:
        bos = 0
        Tloc = T
        # non-varlen inputs are [B, T, H, *]: index the real batch on dim 0 and the
        # real timestep on dim 1 (gt = t). Folding the batch into the time index
        # (gt = i_n*T + t) overflows dim 1 for i_n >= 1, so the bounds-checked
        # gather reads padding (0) and the scatter is dropped (uninit output).
        b = i_n

    o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
    o_v = i_v * BV + ct.arange(BV, dtype=ct.int32)
    m_k = o_k < K
    m_v = o_v < V
    # state orientation [BV, BK]: rows = V, cols = K
    m_h = m_k[None, :] & m_v[:, None]

    b_h = ct.zeros((BV, BK), dtype=ct.float32)
    if USE_INITIAL_STATE:
        # h0 [N, H, K, V] -> [BV, BK] tile: K along cols, V along rows.
        b_h0 = ct.gather(
            h0,
            (i_n, i_h, o_k[None, :], o_v[:, None]),
            mask=m_h,
            check_bounds=True,
            padding_value=0.0,
        )
        b_h = b_h + ct.astype(b_h0, ct.float32)

    for t in range(0, Tloc):
        gt = bos + t

        b_k = ct.astype(ct.gather(k, (b, gt, i_h, o_k), mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)
        b_v = ct.astype(ct.gather(v, (b, gt, i_h, o_v), mask=m_v, check_bounds=False, padding_value=0.0), ct.float32)
        b_q = ct.astype(ct.gather(q, (b, gt, i_h, o_k), mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)
        b_q = b_q * scale

        # b_v_minus = sum(b_h * b_k[None, :], axis=1)  -> [BV]
        b_v_minus = ct.sum(b_h * b_k[None, :], axis=1)
        b_v = b_v - b_v_minus

        if IS_BETA_HEADWISE:
            b_beta = ct.astype(
                ct.gather(beta, (b, gt, i_h, o_v), mask=m_v, check_bounds=False, padding_value=0.0), ct.float32
            )
        else:
            b_beta = ct.astype(ct.gather(beta, (b, gt, i_h), check_bounds=False, padding_value=0.0), ct.float32)

        # u = b_v (delta correction, before beta scaling)
        ct.scatter(u, (b, gt, i_h, o_v), ct.astype(b_v, u.dtype), mask=m_v, check_bounds=False)

        b_v = b_v * b_beta
        # b_h += b_k[None, :] * b_v[:, None]  -> [BV, BK]
        b_h = b_h + b_k[None, :] * b_v[:, None]

        # b_o = sum(b_h * b_q[None, :], axis=1)  -> [BV]
        b_o = ct.sum(b_h * b_q[None, :], axis=1)
        ct.scatter(o, (i_k, b, gt, i_h, o_v), ct.astype(b_o, o.dtype), mask=m_v, check_bounds=False)

    if STORE_FINAL_STATE:
        ct.scatter(
            ht,
            (i_n, i_h, o_k[None, :], o_v[:, None]),
            ct.astype(b_h, ht.dtype),
            mask=m_h,
            check_bounds=True,
        )


@ct.kernel
def fused_recurrent_delta_rule_bwd_kernel_ct(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    beta,  # [B, T, H] scalar, or [B, T, H, V] headwise
    h0,  # [N, H, K, V] (dummy when unused)
    dh0,  # [N, H, K, V] output (dummy when unused)
    dht,  # [N, H, K, V] (dummy when unused)
    do,  # [B, T, H, V]
    dq,  # [NV, B, T, H, K] output
    dk,  # [NV, B, T, H, K] output
    dv,  # [NK, B, T, H, V] output
    db_scalar,  # [NV, B, T, H] output (scalar-beta path; dummy when headwise/no-db)
    db_vec,  # [NV, NK, B, T, H, V] output (headwise-beta path; dummy otherwise)
    cu_seqlens,  # [N+1] int32 (dummy when not varlen)
    scale: ConstFloat,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    IS_BETA_HEADWISE: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    COMPUTE_INITIAL_STATE_GRAD: ConstInt,
    USE_FINAL_STATE_GRADIENT: ConstInt,
    IS_VARLEN: ConstInt,
    COMPUTE_DB: ConstInt,
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
        # varlen inputs are pre-flattened to [1, total_T, H, *]: the batch dim is
        # always 0 and the per-sequence offset lives in the time index (gt = bos + t).
        b = 0
    else:
        bos = 0
        Tloc = T
        # non-varlen inputs are [B, T, H, *]: index the real batch on dim 0 and the
        # real timestep on dim 1 (gt = t). Folding the batch into the time index
        # (gt = i_n*T + t) overflows dim 1 for i_n >= 1, so the bounds-checked
        # gather reads padding (0) and the scatter is dropped (uninit output).
        b = i_n

    o_k = i_k * BK + ct.arange(BK, dtype=ct.int32)
    o_v = i_v * BV + ct.arange(BV, dtype=ct.int32)
    m_k = o_k < K
    m_v = o_v < V
    # state orientation [BK, BV]: rows = K, cols = V
    m_h = m_k[:, None] & m_v[None, :]

    # ----- reverse sweep: accumulate b_dh, emit dk / dv / db -----
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

    for j in range(0, Tloc):
        t = Tloc - 1 - j
        gt = bos + t

        b_q = ct.astype(ct.gather(q, (b, gt, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0), ct.float32)
        b_q = b_q * scale
        b_k = ct.astype(ct.gather(k, (b, gt, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0), ct.float32)
        b_v = ct.astype(ct.gather(v, (b, gt, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0), ct.float32)
        b_do = ct.astype(ct.gather(do, (b, gt, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0), ct.float32)
        if IS_BETA_HEADWISE:
            b_beta = ct.astype(
                ct.gather(beta, (b, gt, i_h, o_v), mask=m_v, check_bounds=False, padding_value=0.0), ct.float32
            )
        else:
            b_beta = ct.astype(ct.gather(beta, (b, gt, i_h), check_bounds=False, padding_value=0.0), ct.float32)

        # b_dh += b_q[:, None] * b_do[None, :]  -> [BK, BV]
        b_dh = b_dh + b_q[:, None] * b_do[None, :]
        # b_dk = sum(b_dh * (b_v * b_beta)[None, :], axis=1)  -> [BK]
        b_vb = b_v * b_beta
        b_dk = ct.sum(b_dh * b_vb[None, :], axis=1)
        # b_dv = sum(b_dh * b_k[:, None], axis=0)  -> [BV]
        b_dv = ct.sum(b_dh * b_k[:, None], axis=0)

        if COMPUTE_DB:
            if IS_BETA_HEADWISE:
                b_db = b_dv * b_v  # [BV]
            else:
                b_db = ct.sum(b_dv * b_v)  # scalar
        b_dv = b_dv * b_beta

        ct.scatter(dk, (i_v, b, gt, i_h, o_k), ct.astype(b_dk, dk.dtype), mask=m_k, check_bounds=True)
        ct.scatter(dv, (i_k, b, gt, i_h, o_v), ct.astype(b_dv, dv.dtype), mask=m_v, check_bounds=True)
        if COMPUTE_DB:
            if IS_BETA_HEADWISE:
                ct.scatter(
                    db_vec, (i_v, i_k, b, gt, i_h, o_v), ct.astype(b_db, db_vec.dtype), mask=m_v, check_bounds=True
                )
            else:
                ct.scatter(db_scalar, (i_v, b, gt, i_h), ct.astype(b_db, db_scalar.dtype), check_bounds=True)

        # b_dh -= b_k[:, None] * b_dv[None, :]
        b_dh = b_dh - b_k[:, None] * b_dv[None, :]

    if COMPUTE_INITIAL_STATE_GRAD:
        ct.scatter(
            dh0,
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            ct.astype(b_dh, dh0.dtype),
            mask=m_h,
            check_bounds=True,
        )

    # ----- forward sweep: rebuild b_h, correct dk, produce dq -----
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

    for t in range(0, Tloc):
        gt = bos + t

        b_dk = ct.astype(
            ct.gather(dk, (i_v, b, gt, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0), ct.float32
        )
        b_dv = ct.astype(
            ct.gather(dv, (i_k, b, gt, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0), ct.float32
        )
        # b_dk -= sum(b_dv[None, :] * b_h, axis=1)  (b_h [BK, BV]) -> [BK]
        b_dk = b_dk - ct.sum(b_dv[None, :] * b_h, axis=1)
        ct.scatter(dk, (i_v, b, gt, i_h, o_k), ct.astype(b_dk, dk.dtype), mask=m_k, check_bounds=True)

        b_k = ct.astype(ct.gather(k, (b, gt, i_h, o_k), mask=m_k, check_bounds=True, padding_value=0.0), ct.float32)
        b_v = ct.astype(ct.gather(v, (b, gt, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0), ct.float32)
        b_do = ct.astype(ct.gather(do, (b, gt, i_h, o_v), mask=m_v, check_bounds=True, padding_value=0.0), ct.float32)
        if IS_BETA_HEADWISE:
            b_beta = ct.astype(
                ct.gather(beta, (b, gt, i_h, o_v), mask=m_v, check_bounds=False, padding_value=0.0), ct.float32
            )
        else:
            b_beta = ct.astype(ct.gather(beta, (b, gt, i_h), check_bounds=False, padding_value=0.0), ct.float32)
        b_v = b_v * b_beta

        # b_h += b_k[:, None] * b_v[None, :]  -> [BK, BV]
        b_h = b_h + b_k[:, None] * b_v[None, :]
        # d_q = sum(b_h * b_do[None, :], axis=1) * scale  -> [BK]
        d_q = ct.sum(b_h * b_do[None, :], axis=1) * scale
        ct.scatter(dq, (i_v, b, gt, i_h, o_k), ct.astype(d_q, dq.dtype), mask=m_k, check_bounds=True)


# ======================== Wrapper Functions ========================


def fused_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = _next_power_of_2(K)
    BV = min(_next_power_of_2(V), 8)
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV
    assert NK == 1, "NK > 1 is not supported yet"

    o = q.new_empty(NK, *v.shape)
    u = torch.empty_like(v)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    is_beta_headwise = beta.ndim == v.ndim
    use_initial = initial_state is not None
    store_final = final_state is not None
    is_varlen = cu_seqlens is not None

    # cuTile launch args cannot be None: substitute dummy tensors + flags.
    h0_arg = initial_state if use_initial else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    ht_arg = final_state if store_final else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else torch.empty(1, device=q.device, dtype=torch.int32)

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_recurrent_delta_rule_fwd_kernel_ct,
        (
            q,
            k,
            v,
            u,
            beta,
            o,
            h0_arg,
            ht_arg,
            cu_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BK,
            BV,
            int(use_initial),
            int(store_final),
            int(is_beta_headwise),
            int(is_varlen),
        ),
    )
    o = o.squeeze(0)
    return o, u, final_state


def fused_recurrent_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    compute_db: bool = True,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    beta_vector = beta.ndim == v.ndim
    BK = _next_power_of_2(K)
    BV = min(_next_power_of_2(V), 32)
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)

    use_initial = initial_state is not None
    compute_initial_state_grad = use_initial and initial_state.requires_grad
    if compute_initial_state_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    use_final_grad = dht is not None
    is_varlen = cu_seqlens is not None

    # db output: scalar path [NV, B, T, H], headwise path [NV, NK, B, T, H, V].
    if beta_vector and compute_db:
        db_vec = q.new_empty(NV, NK, B, T, H, V)
        db_scalar = q.new_empty(1, 1, 1, 1, dtype=q.dtype)
    elif compute_db:
        db_scalar = q.new_empty(NV, B, T, H)
        db_vec = q.new_empty(1, 1, 1, 1, 1, 1, dtype=q.dtype)
    else:
        db_scalar = q.new_empty(1, 1, 1, 1, dtype=q.dtype)
        db_vec = q.new_empty(1, 1, 1, 1, 1, 1, dtype=q.dtype)

    h0_arg = initial_state if use_initial else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    dh0_arg = dh0 if compute_initial_state_grad else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    dht_arg = dht if use_final_grad else q.new_empty(1, 1, 1, 1, dtype=torch.float32)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else torch.empty(1, device=q.device, dtype=torch.int32)

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_recurrent_delta_rule_bwd_kernel_ct,
        (
            q,
            k,
            v,
            beta,
            h0_arg,
            dh0_arg,
            dht_arg,
            do,
            dq,
            dk,
            dv,
            db_scalar,
            db_vec,
            cu_arg,
            float(scale),
            T,
            H,
            K,
            V,
            BK,
            BV,
            int(beta_vector),
            int(use_initial),
            int(compute_initial_state_grad),
            int(use_final_grad),
            int(is_varlen),
            int(compute_db),
        ),
    )

    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.squeeze(0)
    if compute_db:
        db = db_vec.sum((0, 1)) if beta_vector else db_scalar.sum(0)
    else:
        db = None

    return dq, dk, dv, db, dh0


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        o, u, final_state = fused_recurrent_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, u, beta, initial_state)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, q_rstd, k, k_rstd, v, beta, initial_state = ctx.saved_tensors
        dq, dk, dv, db, dh0 = fused_recurrent_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            dht=dht,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
            compute_db=ctx.needs_input_grad[3],
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        db = db.to(beta) if db is not None else None
        return dq.to(q), dk.to(k), dv.to(v), db, None, dh0, None, None, None


@torch.compiler.disable
def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    r"""
    Args:
        q (torch.Tensor): queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        beta (torch.Tensor): betas of shape `[B, T, H]` (or `[B, T, H, V]` headwise).
        scale (Optional[float]): Scale factor; defaults to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]): `[N, H, K, V]` initial state.
        output_final_state (Optional[bool]): Whether to output the final state.
        use_qk_l2norm_in_kernel (Optional[bool]): Whether to L2-normalize q, k in-kernel.
        cu_seqlens (torch.LongTensor): `[N+1]` cumulative seqlens for varlen.

    Returns:
        o (torch.Tensor): Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor): `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
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
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if q.dtype in (torch.float16, torch.bfloat16) and beta.ndim == 3 and k.shape[-1] <= 256 and cu_seqlens is None:
        from tilegym.suites.fla.cutile.linear_attention_deltanet_chunk import chunk_delta_rule

        return chunk_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state


register_impl("fla.linear_attention_deltanet_fused_recurrent", backend="cutile")(fused_recurrent_delta_rule)
