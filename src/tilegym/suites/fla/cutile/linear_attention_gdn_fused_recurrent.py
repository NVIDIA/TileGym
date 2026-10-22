# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Port of the flash-linear-attention fused_recurrent_gated_delta_rule op.
# Dense BF16 calls use the shared chunk implementation; other supported
# configurations use the general fused-recurrent kernels.

import warnings
from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl  # noqa: F401
from tilegym.suites.fla.cutile._host_utils import input_guard

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
ConstBool = ct.Constant[bool]


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# Device helpers


@ct.function
def _exp(x):
    return ct.exp(ct.astype(x, ct.float32))


@ct.function
def _softplus(x):
    return ct.where(x < 20.0, ct.log(1.0 + ct.exp(x)), x)


@ct.kernel
def fused_recurrent_gated_delta_rule_fwd_kernel_ct(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, HV, V]
    g,  # [B, T, HV]   (dummy [1] when not USE_G)
    gk,  # [B, T, HV, K] (dummy [1] when not USE_GK)
    gv,  # [B, T, HV, V] (dummy [1] when not USE_GV)
    beta,  # [B, T, HV] scalar or [B, T, HV, V] headwise
    A_log,  # [HV]  (dummy [1] when not USE_GATE_IN_KERNEL)
    dt_bias,  # [HV]  (dummy [1] when not HAS_DT_BIAS)
    o,  # [B, T, HV, V] output
    h0,  # [N, HV, K, V] or [N, HV, V, K] (dummy when not USE_INITIAL_STATE)
    ht,  # [N, HV, K, V] or [N, HV, V, K] (dummy when not STORE_FINAL_STATE)
    cu_seqlens,  # [N+1] int32 (dummy [1] when not varlen)
    scale: ConstFloat,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    USE_G: ConstBool,
    USE_GK: ConstBool,
    USE_GV: ConstBool,
    USE_QK_L2NORM_IN_KERNEL: ConstBool,
    IS_BETA_HEADWISE: ConstBool,
    USE_INITIAL_STATE: ConstBool,
    STORE_FINAL_STATE: ConstBool,
    STATE_V_FIRST: ConstBool,
    IS_VARLEN: ConstBool,
    USE_GATE_IN_KERNEL: ConstBool,
    HAS_DT_BIAS: ConstBool,
    APPLY_BETA_SIGMOID: ConstBool,
    ALLOW_NEG_EIGVAL: ConstBool,
):
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        # varlen packs all sequences into batch dim 0: q is [1, sum(T), H, K].
        bos = ct.gather(cu_seqlens, (i_n,), check_bounds=True, padding_value=0).item()
        eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=True, padding_value=0).item()
        Tloc = eos - bos
        b = 0
    else:
        bos = 0
        Tloc = T
        b = i_n

    o_k = ct.arange(BK, dtype=ct.int32)
    o_v = i_v * BV + ct.arange(BV, dtype=ct.int32)
    mask_k = o_k < K
    mask_v = o_v < V
    if STATE_V_FIRST:
        # state [BV, BK]: rows = V, cols = K
        mask_h = mask_v[:, None] & mask_k[None, :]
        b_h = ct.zeros((BV, BK), dtype=ct.float32)
    else:
        # state [BK, BV]: rows = K, cols = V
        mask_h = mask_k[:, None] & mask_v[None, :]
        b_h = ct.zeros((BK, BV), dtype=ct.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            b_h0 = ct.gather(
                h0, (i_n, i_hv, o_v[:, None], o_k[None, :]), mask=mask_h, check_bounds=True, padding_value=0.0
            )
        else:
            b_h0 = ct.gather(
                h0, (i_n, i_hv, o_k[:, None], o_v[None, :]), mask=mask_h, check_bounds=True, padding_value=0.0
            )
        b_h = b_h + ct.astype(b_h0, ct.float32)

    for t in range(0, Tloc):
        # global time index into the (B, T, ...) tensors (varlen: bos = cu_seqlens[i_n]).
        gt = bos + t

        b_q = ct.astype(ct.gather(q, (b, gt, i_h, o_k), mask=mask_k, check_bounds=True, padding_value=0.0), ct.float32)
        b_k = ct.astype(ct.gather(k, (b, gt, i_h, o_k), mask=mask_k, check_bounds=True, padding_value=0.0), ct.float32)
        b_v = ct.astype(ct.gather(v, (b, gt, i_hv, o_v), mask=mask_v, check_bounds=True, padding_value=0.0), ct.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * ct.rsqrt(ct.sum(b_q * b_q) + 1e-6)
            b_k = b_k * ct.rsqrt(ct.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        if IS_BETA_HEADWISE:
            # scalar/per-head beta [B, T, HV]: single value
            b_beta = ct.astype(ct.gather(beta, (b, gt, i_hv), check_bounds=True, padding_value=0.0), ct.float32)
        else:
            # vector beta [B, T, HV, V]: per-V-channel
            b_beta = ct.astype(
                ct.gather(beta, (b, gt, i_hv, o_v), mask=mask_v, check_bounds=True, padding_value=0.0), ct.float32
            )
        if APPLY_BETA_SIGMOID:
            b_beta = 1.0 / (1.0 + ct.exp(-b_beta))
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2.0

        if USE_G:
            b_g = ct.astype(ct.gather(g, (b, gt, i_hv), check_bounds=True, padding_value=0.0), ct.float32)
            if USE_GATE_IN_KERNEL:
                b_A = ct.astype(ct.gather(A_log, (i_hv,), check_bounds=True, padding_value=0.0), ct.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + ct.astype(ct.gather(dt_bias, (i_hv,), check_bounds=True, padding_value=0.0), ct.float32)
                b_g = -_exp(b_A) * _softplus(b_g)
            b_h = b_h * _exp(b_g)

        if USE_GK:
            b_gk = ct.astype(
                ct.gather(gk, (b, gt, i_hv, o_k), mask=mask_k, check_bounds=True, padding_value=0.0), ct.float32
            )
            if STATE_V_FIRST:
                b_h = b_h * _exp(b_gk[None, :])
            else:
                b_h = b_h * _exp(b_gk[:, None])

        if USE_GV:
            b_gv = ct.astype(
                ct.gather(gv, (b, gt, i_hv, o_v), mask=mask_v, check_bounds=True, padding_value=0.0), ct.float32
            )
            if STATE_V_FIRST:
                b_h = b_h * _exp(b_gv[:, None])
            else:
                b_h = b_h * _exp(b_gv[None, :])

        if STATE_V_FIRST:
            # b_v = beta * (v - sum(h * k[None, :], axis=1))  -> [BV]
            b_v = b_beta * (b_v - ct.sum(b_h * b_k[None, :], axis=1))
            b_h = b_h + b_v[:, None] * b_k[None, :]
            b_o = ct.sum(b_h * b_q[None, :], axis=1)
        else:
            # b_v = beta * (v - sum(h * k[:, None], axis=0))  -> [BV]
            b_v = b_beta * (b_v - ct.sum(b_h * b_k[:, None], axis=0))
            b_h = b_h + b_k[:, None] * b_v
            b_o = ct.sum(b_h * b_q[:, None], axis=0)
        ct.scatter(o, (b, gt, i_hv, o_v), ct.astype(b_o, o.dtype), mask=mask_v, check_bounds=True)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            ct.scatter(
                ht, (i_n, i_hv, o_v[:, None], o_k[None, :]), ct.astype(b_h, ht.dtype), mask=mask_h, check_bounds=True
            )
        else:
            ct.scatter(
                ht, (i_n, i_hv, o_k[:, None], o_v[None, :]), ct.astype(b_h, ht.dtype), mask=mask_h, check_bounds=True
            )


# Wrapper Functions


@input_guard
def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = _next_power_of_2(K)
    BV = min(8, _next_power_of_2(V)) if gv is None else _next_power_of_2(V)
    o = torch.empty_like(v)
    if output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    is_beta_headwise = beta.ndim != v.ndim
    use_g = g is not None
    use_gk = gk is not None
    use_gv = gv is not None
    use_initial = initial_state is not None
    store_final = final_state is not None
    is_varlen = cu_seqlens is not None
    use_gate_in_kernel = A_log is not None
    has_dt_bias = dt_bias is not None

    # cuTile launch args cannot be None: substitute dummy tensors + flags.
    def _dummy(*shape, dtype=q.dtype):
        return q.new_empty(*shape, dtype=dtype)

    g_arg = g if use_g else _dummy(1)
    gk_arg = gk if use_gk else _dummy(1)
    gv_arg = gv if use_gv else _dummy(1)
    A_log_arg = A_log if use_gate_in_kernel else _dummy(1)
    dt_bias_arg = dt_bias if has_dt_bias else _dummy(1)
    h0_arg = initial_state if use_initial else _dummy(1, 1, 1, 1, dtype=torch.float32)
    ht_arg = final_state if store_final else _dummy(1, 1, 1, 1, dtype=torch.float32)
    cu_arg = cu_seqlens.to(torch.int32) if is_varlen else torch.empty(1, device=q.device, dtype=torch.int32)

    NV = _cdiv(V, BV)
    grid = (NV, N * HV)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_recurrent_gated_delta_rule_fwd_kernel_ct,
        (
            q,
            k,
            v,
            g_arg,
            gk_arg,
            gv_arg,
            beta,
            A_log_arg,
            dt_bias_arg,
            o,
            h0_arg,
            ht_arg,
            cu_arg,
            float(scale),
            T,
            H,
            HV,
            K,
            V,
            BK,
            BV,
            bool(use_g),
            bool(use_gk),
            bool(use_gv),
            bool(use_qk_l2norm_in_kernel),
            bool(is_beta_headwise),
            bool(use_initial),
            bool(store_final),
            bool(state_v_first),
            bool(is_varlen),
            bool(use_gate_in_kernel),
            bool(has_dt_bias),
            bool(use_beta_sigmoid_in_kernel),
            bool(allow_neg_eigval),
        ),
    )
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        gv: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
        )

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps.",
        )


@input_guard
def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`, where `HV` must be divisible by `H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
            When `use_gate_in_kernel=False` (default), `g` must be in log space (pre-computed decay).
            When `use_gate_in_kernel=True`, `g` is the raw pre-activation input; the kernel fuses
            `-exp(A_log) * softplus(g + dt_bias)` internally per step.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space GDN decay internally.
            When `True`, `g` is the raw input and `A_log` must be provided; the kernel fuses
            gate activation into the recurrence. Default: `False`.
        A_log (Optional[torch.Tensor]):
            Decay parameter of shape `[HV]`. Required when `use_gate_in_kernel=True`.
        dt_bias (Optional[torch.Tensor]):
            Bias added to `g` before activation, of shape `[HV]`.
            Only used when `use_gate_in_kernel=True`.
        use_beta_sigmoid_in_kernel (Optional[bool]):
            Whether to apply `torch.sigmoid(beta)` inside the kernel.
            - If `True`, the passed `beta` acts as the raw beta logits.
            - If `False`, `beta` is expected to already be in post-sigmoid space.
            Default: `False`.
        allow_neg_eigval (Optional[bool]):
            Whether to allow negative eigenvalues by scaling `beta` to `[0, 2)`.
            Only takes effect together with `use_beta_sigmoid_in_kernel=True`, in which case
            the kernel computes `2 * sigmoid(beta)` instead of `sigmoid(beta)`. Default: `False`.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first ``[V, K]`` layout instead of the default ``[K, V]``. Default: ``False``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Training is supported only by the BF16 dense chunk route with equal heads,
    scalar beta, q/k normalization, precomputed gates and no state or fused options.
    Other configurations are inference-only and reject inputs requiring gradients.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    """
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    if q.shape[2] != k.shape[2] or v.shape[2] % q.shape[2]:
        raise ValueError("q/k heads must match and divide the value head count")
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
    if beta is None:
        beta = q.new_ones(v.shape[:-1])
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
        if g is None:
            raise ValueError("`g` (raw pre-activation) must be provided when `use_gate_in_kernel=True`.")
    else:
        A_log = None
        dt_bias = None
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    supports_training = (
        q.dtype == torch.bfloat16
        and g is not None
        and gk is None
        and gv is None
        and beta is not None
        and beta.ndim != v.ndim
        and A_log is None
        and dt_bias is None
        and initial_state is None
        and not output_final_state
        and use_qk_l2norm_in_kernel
        and not use_beta_sigmoid_in_kernel
        and not allow_neg_eigval
        and not state_v_first
        and cu_seqlens is None
        and k.shape[2] == v.shape[2]
    )
    requires_backward = torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in (q, k, v, g, gk, gv, beta, A_log, dt_bias, initial_state)
    )
    if requires_backward and not supports_training:
        raise NotImplementedError(
            "GDN recurrent training requires BF16 dense equal-head inputs, scalar beta, "
            "q/k normalization, precomputed scalar gates, no initial/final state and no fused options. "
            "Use linear_attention_gdn_chunk for other supported training configurations."
        )
    if supports_training:
        from tilegym.suites.fla.cutile.linear_attention_gdn_chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        state_v_first,
        cu_seqlens,
    )
    return o, final_state


fused_recurrent_gdn = fused_recurrent_gated_delta_rule


# TileGym dispatch registration


@register_impl("fla.linear_attention_gdn_fused_recurrent", backend="cutile")
def _gdn_fused_recurrent_dispatch(
    q,
    k,
    v,
    g=None,
    beta=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **kwargs,
):
    # Forward by keyword: positional forwarding would misroute beta into gk.
    return fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )
