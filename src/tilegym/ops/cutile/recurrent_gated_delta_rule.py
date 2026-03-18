# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel
def recurrent_gated_delta_rule_fwd_cutile_kernel(
    # Tensors — native (B, T, H, D) layout for Q/K/V; (B, T, H) for G/Beta
    Q,
    K,
    V,
    G,
    Beta,
    Output,  # (B, T, H, V)
    InitState,  # (B, H, K, V) or dummy
    FinalState,  # (B, H, K, V) or dummy
    # Runtime scalars
    scale: float,
    seq_len: int,
    # Compile-time constants
    NUM_HEADS: ConstInt,
    HAS_INITIAL_STATE: ConstBool,
    OUTPUT_FINAL_STATE: ConstBool,
    USE_QK_L2NORM: ConstBool,
    BLOCK_K: ConstInt,
    BLOCK_V: ConstInt,
):
    """
    Grid: (batch_size * num_heads, ceil(v_head_dim / BLOCK_V), 1)

    Each program owns a (BLOCK_K, BLOCK_V) tile of recurrent state
    and loops over timesteps sequentially.  Uses TMA for all loads/stores.
    """
    pid_bh = ct.bid(0)
    pid_v = ct.bid(1)

    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    _ZERO = ct.PaddingMode.ZERO

    # ---- Initialize state (BLOCK_K, BLOCK_V) ----
    state = ct.zeros((BLOCK_K, BLOCK_V), dtype=ct.float32)
    if HAS_INITIAL_STATE:
        state = ct.load(
            InitState,
            index=(b, h, 0, pid_v),
            shape=(1, 1, BLOCK_K, BLOCK_V),
            padding_mode=_ZERO,
        ).reshape((BLOCK_K, BLOCK_V))
        state = ct.astype(state, ct.float32)

    # ---- Recurrent loop ----
    for t in range(seq_len):
        # Load q_t, k_t: (BLOCK_K,)
        q_t = ct.load(
            Q,
            index=(b, t, h, 0),
            shape=(1, 1, 1, BLOCK_K),
            padding_mode=_ZERO,
        ).reshape((BLOCK_K,))
        q_t = ct.astype(q_t, ct.float32)

        k_t = ct.load(
            K,
            index=(b, t, h, 0),
            shape=(1, 1, 1, BLOCK_K),
            padding_mode=_ZERO,
        ).reshape((BLOCK_K,))
        k_t = ct.astype(k_t, ct.float32)

        # Optional L2-normalize q and k
        if USE_QK_L2NORM:
            q_t = q_t * ct.rsqrt(ct.sum(q_t * q_t, axis=0) + 1e-6)
            k_t = k_t * ct.rsqrt(ct.sum(k_t * k_t, axis=0) + 1e-6)

        q_t = q_t * scale

        # Load v_t: (BLOCK_V,)
        v_t = ct.load(
            V,
            index=(b, t, h, pid_v),
            shape=(1, 1, 1, BLOCK_V),
            padding_mode=_ZERO,
        ).reshape((BLOCK_V,))
        v_t = ct.astype(v_t, ct.float32)

        # Load g_t, beta_t: scalars
        g_t = ct.astype(ct.load(G, index=(b, t, h), shape=(), padding_mode=_ZERO), ct.float32)
        beta_t = ct.astype(ct.load(Beta, index=(b, t, h), shape=(), padding_mode=_ZERO), ct.float32)

        # 1. Decay state
        state = state * ct.exp(g_t)

        # 2. Retrieve from memory: kv_mem[v] = sum_k state[k,v] * k_t[k]
        k_col = ct.expand_dims(k_t, axis=1)  # (BLOCK_K, 1)
        kv_mem = ct.sum(state * k_col, axis=0)  # (BLOCK_V,)

        # 3. Compute delta
        delta = (v_t - kv_mem) * beta_t

        # 4. Rank-1 state update: state += outer(k_t, delta)
        state = state + k_col * ct.expand_dims(delta, axis=0)

        # 5. Query: out_t[v] = sum_k state[k,v] * q_t[k]
        out_t = ct.sum(state * ct.expand_dims(q_t, axis=1), axis=0)

        # Store output (cast to output dtype)
        out_tile = ct.astype(
            ct.reshape(out_t, (1, 1, 1, BLOCK_V)),
            Output.dtype,
        )
        ct.store(Output, index=(b, t, h, pid_v), tile=out_tile)

    # ---- Store final state ----
    if OUTPUT_FINAL_STATE:
        ct.store(
            FinalState,
            index=(b, h, 0, pid_v),
            tile=ct.reshape(state, (1, 1, BLOCK_K, BLOCK_V)),
        )


class RecurrentGatedDeltaRuleCuTile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel):
        initial_dtype = query.dtype
        B, T, H, K = query.shape
        V = value.shape[-1]
        scale = 1.0 / math.sqrt(K)

        # Ensure contiguous for TMA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        output = torch.empty(B, T, H, V, device=query.device, dtype=initial_dtype)

        has_initial_state = initial_state is not None
        if has_initial_state:
            initial_state = initial_state.contiguous().float()

        final_state = None
        if output_final_state:
            final_state = torch.empty(B, H, K, V, device=query.device, dtype=torch.float32)

        BLOCK_K = 1 << (K - 1).bit_length()
        BLOCK_V = min(64, 1 << (V - 1).bit_length())

        grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V, 1)

        # cuTile cannot accept None — use dummy tensors for absent state
        dummy = torch.empty(1, 1, 1, 1, device=query.device, dtype=torch.float32)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            recurrent_gated_delta_rule_fwd_cutile_kernel,
            (
                query,
                key,
                value,
                g,
                beta,
                output,
                initial_state if has_initial_state else dummy,
                final_state if output_final_state else dummy,
                scale,
                T,
                H,
                has_initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                BLOCK_K,
                BLOCK_V,
            ),
        )

        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        raise NotImplementedError("Backward pass not implemented for RecurrentGatedDeltaRuleCuTile")


@register_impl("recurrent_gated_delta_rule", backend="cutile")
def recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs
):
    """Drop-in cuTile replacement for torch_recurrent_gated_delta_rule."""
    return RecurrentGatedDeltaRuleCuTile.apply(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )
