# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _ct_mm(a, b):
    """ct.matmul with fp32→tf32 cast for tensor core usage.
    Returns fp32 result (cast back from tf32) for compatibility with
    subsequent fp32 element-wise ops."""
    if a.dtype == ct.float32:
        a = ct.astype(a, ct.tfloat32)
    if b.dtype == ct.float32:
        b = ct.astype(b, ct.tfloat32)
    return ct.astype(ct.matmul(a, b), ct.float32)


def _ct_solve_tril_serial(A, CS):
    """Invert a unit lower-triangular system in reference operation order."""
    offs = ct.arange(CS, dtype=ct.int32)
    for i in range(1, CS):
        is_row = offs == i
        is_row_col = ct.expand_dims(is_row, axis=1)
        row = ct.sum(ct.where(is_row_col, A, 0.0), axis=0)
        correction = ct.sum(ct.expand_dims(row, axis=1) * A, axis=0)
        A = A + ct.where(is_row_col, ct.expand_dims(correction, axis=0), 0.0)
    eye = ct.where(
        ct.expand_dims(offs, axis=1) == ct.expand_dims(offs, axis=0),
        1.0,
        0.0,
    )
    return A + eye


def _ct_solve_tril_neumann_guarded(A, CS, n_steps):
    """Use fast Neumann-by-squaring only when its powers cannot grow.

    Full-fp32 Neumann products still create cancellation terms up to 1e15 on
    correlated model keys, so changing MMA precision alone is insufficient.
    The matrix infinity norm is submultiplicative, so ||A||_inf < 1 bounds every
    squared power instead of discovering instability after the expensive solve.
    """
    offs = ct.arange(CS, dtype=ct.int32)
    eye = ct.where(
        ct.expand_dims(offs, axis=1) == ct.expand_dims(offs, axis=0),
        1.0,
        0.0,
    )
    norm_inf = ct.max(ct.sum(ct.abs(A), axis=1))
    result = A
    if norm_inf < 1.0:
        result = eye + A
        A_power = A
        for _ in range(1, n_steps):
            A_power = _ct_mm(A_power, A_power)
            result = _ct_mm(result, eye + A_power)
    else:
        result = _ct_solve_tril_serial(A, CS)
    return result


@ct.kernel(occupancy=2)
def _intra_chunk_prepare_kernel(
    Q,
    K,
    V,
    Beta,
    G,  # raw inputs (B,T,H,D) / (B,T,H)
    Q_out,
    K_out,
    V_corr,
    K_cumdecay,
    G_cum_out,  # 5D outputs
    seq_len: int,
    K_dim: int,
    V_dim: int,
    scale: float,
    NUM_HEADS: ConstInt,
    USE_QK_L2NORM: ConstBool,
    CHUNK_SIZE: ConstInt,
    BLOCK_K: ConstInt,
    SOLVE_STEPS: ConstInt,
    INTER_BF16: ConstBool,  # bf16 intermediates (only when input is bf16/fp16)
):
    """Grid: (B*H, num_chunks, 1).  Each program handles one (b, h, chunk)."""
    pid_bh = ct.bid(0)
    pid_chunk = ct.bid(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    _ZERO = ct.PaddingMode.ZERO

    # General path (any chunk size): full reference-order triangular solve.
    k_tile = ct.astype(
        ct.load(
            K,
            index=(b, pid_chunk, h, 0),
            shape=(1, CHUNK_SIZE, 1, BLOCK_K),
            padding_mode=_ZERO,
        ).reshape((CHUNK_SIZE, BLOCK_K)),
        ct.float32,
    )

    if USE_QK_L2NORM:
        k_tile = k_tile * ct.rsqrt(ct.sum(k_tile * k_tile, axis=1, keepdims=True) + 1e-6)

    beta_tile = ct.astype(
        ct.load(
            Beta,
            index=(b, pid_chunk, h),
            shape=(1, CHUNK_SIZE, 1),
            padding_mode=_ZERO,
        ).reshape((CHUNK_SIZE,)),
        ct.float32,
    )

    g_raw = ct.astype(
        ct.load(
            G,
            index=(b, pid_chunk, h),
            shape=(1, CHUNK_SIZE, 1),
            padding_mode=_ZERO,
        ).reshape((CHUNK_SIZE,)),
        ct.float32,
    )
    g_cum = ct.cumsum(g_raw, axis=0)

    offs_c = ct.arange(CHUNK_SIZE, dtype=ct.int32)
    offs_c_row = ct.expand_dims(offs_c, axis=1)
    offs_c_col = ct.expand_dims(offs_c, axis=0)
    lower_tri = offs_c_row >= offs_c_col
    strict_lower = offs_c_row > offs_c_col

    gc_row = ct.expand_dims(g_cum, axis=1)
    gc_col = ct.expand_dims(g_cum, axis=0)
    decay_mask = ct.where(lower_tri, ct.exp(gc_row - gc_col), 0.0)

    kb_tile = k_tile * ct.expand_dims(beta_tile, axis=1)
    base_attn = _ct_mm(kb_tile, ct.transpose(k_tile))
    attn = ct.where(strict_lower, -(base_attn * decay_mask), 0.0)
    if USE_QK_L2NORM:
        # Model-normalized keys can be nearly collinear; keep their exact
        # reference-order solve so small prefill differences cannot alter decode.
        attn = _ct_solve_tril_serial(attn, CHUNK_SIZE)
    else:
        attn = _ct_solve_tril_neumann_guarded(attn, CHUNK_SIZE, SOLVE_STEPS)

    num_v_tiles = ct.cdiv(V_dim, BLOCK_K)
    for vt in range(num_v_tiles):
        v_tile = ct.astype(
            ct.load(
                V,
                index=(b, pid_chunk, h, vt),
                shape=(1, CHUNK_SIZE, 1, BLOCK_K),
                padding_mode=_ZERO,
            ).reshape((CHUNK_SIZE, BLOCK_K)),
            ct.float32,
        )
        vb_tile = v_tile * ct.expand_dims(beta_tile, axis=1)
        vc_out = _ct_mm(attn, vb_tile)
        if INTER_BF16:
            ct.store(
                V_corr,
                index=(b, h, pid_chunk, 0, vt),
                tile=ct.reshape(ct.astype(vc_out, ct.bfloat16), (1, 1, 1, CHUNK_SIZE, BLOCK_K)),
            )
        else:
            ct.store(V_corr, index=(b, h, pid_chunk, 0, vt), tile=ct.reshape(vc_out, (1, 1, 1, CHUNK_SIZE, BLOCK_K)))

    kbe_tile = kb_tile * ct.expand_dims(ct.exp(g_cum), axis=1)
    kc_out = _ct_mm(attn, kbe_tile)
    if INTER_BF16:
        ct.store(
            K_cumdecay,
            index=(b, h, pid_chunk, 0, 0),
            tile=ct.reshape(ct.astype(kc_out, ct.bfloat16), (1, 1, 1, CHUNK_SIZE, BLOCK_K)),
        )
    else:
        ct.store(K_cumdecay, index=(b, h, pid_chunk, 0, 0), tile=ct.reshape(kc_out, (1, 1, 1, CHUNK_SIZE, BLOCK_K)))
    ct.store(G_cum_out, index=(b, h, pid_chunk, 0), tile=ct.reshape(g_cum, (1, 1, 1, CHUNK_SIZE)))

    q_tile = ct.astype(
        ct.load(
            Q,
            index=(b, pid_chunk, h, 0),
            shape=(1, CHUNK_SIZE, 1, BLOCK_K),
            padding_mode=_ZERO,
        ).reshape((CHUNK_SIZE, BLOCK_K)),
        ct.float32,
    )
    if USE_QK_L2NORM:
        q_tile = q_tile * ct.rsqrt(ct.sum(q_tile * q_tile, axis=1, keepdims=True) + 1e-6)
    q_tile = q_tile * scale
    if INTER_BF16:
        ct.store(
            Q_out,
            index=(b, h, pid_chunk, 0, 0),
            tile=ct.reshape(ct.astype(q_tile, ct.bfloat16), (1, 1, 1, CHUNK_SIZE, BLOCK_K)),
        )
        ct.store(
            K_out,
            index=(b, h, pid_chunk, 0, 0),
            tile=ct.reshape(ct.astype(k_tile, ct.bfloat16), (1, 1, 1, CHUNK_SIZE, BLOCK_K)),
        )
    else:
        ct.store(Q_out, index=(b, h, pid_chunk, 0, 0), tile=ct.reshape(q_tile, (1, 1, 1, CHUNK_SIZE, BLOCK_K)))
        ct.store(K_out, index=(b, h, pid_chunk, 0, 0), tile=ct.reshape(k_tile, (1, 1, 1, CHUNK_SIZE, BLOCK_K)))


@ct.kernel
def _chunk_inter_recurrence_kernel(
    Q_ch,
    K_ch,
    V_corr,
    K_cumdecay,
    G_cum_in,
    Output,
    InitState,
    FinalState,
    num_chunks: int,
    K_dim: int,
    V_dim: int,
    NUM_HEADS: ConstInt,
    HAS_INITIAL_STATE: ConstBool,
    OUTPUT_FINAL_STATE: ConstBool,
    CHUNK_SIZE: ConstInt,
    BLOCK_K: ConstInt,
    BLOCK_V: ConstInt,
):
    """Grid: (B*H, cdiv(V, BLOCK_V), 1).  Loops over chunks."""
    pid_bh = ct.bid(0)
    pid_v = ct.bid(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    _ZERO = ct.PaddingMode.ZERO

    state = ct.zeros((BLOCK_K, BLOCK_V), dtype=ct.float32)
    if HAS_INITIAL_STATE:
        state = ct.load(
            InitState,
            index=(b, h, 0, pid_v),
            shape=(1, 1, BLOCK_K, BLOCK_V),
            padding_mode=_ZERO,
        ).reshape((BLOCK_K, BLOCK_V))
        state = ct.astype(state, ct.float32)

    # Pre-compute causal mask (constant across chunks)
    offs_c = ct.arange(CHUNK_SIZE, dtype=ct.int32)
    causal = ct.expand_dims(offs_c, axis=1) >= ct.expand_dims(offs_c, axis=0)

    for ci in range(num_chunks):
        # Load intermediates from 5D tensors
        kc_chunk = ct.astype(
            ct.load(
                K_cumdecay,
                index=(b, h, ci, 0, 0),
                shape=(1, 1, 1, CHUNK_SIZE, BLOCK_K),
                padding_mode=_ZERO,
            ).reshape((CHUNK_SIZE, BLOCK_K)),
            ct.float32,
        )
        v_prime = _ct_mm(kc_chunk, state)  # (CS, BV)

        v_corr = ct.astype(
            ct.load(
                V_corr,
                index=(b, h, ci, 0, pid_v),
                shape=(1, 1, 1, CHUNK_SIZE, BLOCK_V),
                padding_mode=_ZERO,
            ).reshape((CHUNK_SIZE, BLOCK_V)),
            ct.float32,
        )
        v_new = v_corr - v_prime

        q_chunk = ct.astype(
            ct.load(
                Q_ch,
                index=(b, h, ci, 0, 0),
                shape=(1, 1, 1, CHUNK_SIZE, BLOCK_K),
                padding_mode=_ZERO,
            ).reshape((CHUNK_SIZE, BLOCK_K)),
            ct.float32,
        )

        g_cum = ct.load(
            G_cum_in,
            index=(b, h, ci, 0),
            shape=(1, 1, 1, CHUNK_SIZE),
            padding_mode=_ZERO,
        ).reshape((CHUNK_SIZE,))

        # Inter-chunk: q_weighted @ state
        q_weighted = q_chunk * ct.expand_dims(ct.exp(g_cum), axis=1)
        attn_inter = _ct_mm(q_weighted, state)  # (CS, BV)

        # Intra-chunk: QK^T with causal decay mask
        k_chunk = ct.astype(
            ct.load(
                K_ch,
                index=(b, h, ci, 0, 0),
                shape=(1, 1, 1, CHUNK_SIZE, BLOCK_K),
                padding_mode=_ZERO,
            ).reshape((CHUNK_SIZE, BLOCK_K)),
            ct.float32,
        )

        qk = _ct_mm(q_chunk, ct.transpose(k_chunk))  # (CS, CS)
        gc_row = ct.expand_dims(g_cum, axis=1)
        gc_col = ct.expand_dims(g_cum, axis=0)
        decay = ct.where(causal, ct.exp(gc_row - gc_col), 0.0)
        qk_masked = ct.where(causal, qk * decay, 0.0)

        intra_out = _ct_mm(qk_masked, v_new)  # (CS, BV)
        out_chunk = attn_inter + intra_out

        ct.store(
            Output,
            index=(b, h, ci, 0, pid_v),
            tile=ct.reshape(out_chunk, (1, 1, 1, CHUNK_SIZE, BLOCK_V)),
        )

        # State update
        g_last = ct.load(
            G_cum_in,
            index=(b, h, ci, CHUNK_SIZE - 1),
            shape=(),
            padding_mode=_ZERO,
        )
        g_last = ct.astype(g_last, ct.float32)
        k_weighted = k_chunk * ct.expand_dims(ct.exp(g_last - g_cum), axis=1)
        state = state * ct.exp(g_last) + _ct_mm(ct.transpose(k_weighted), v_new)

    if OUTPUT_FINAL_STATE:
        ct.store(
            FinalState,
            index=(b, h, 0, pid_v),
            tile=ct.reshape(state, (1, 1, BLOCK_K, BLOCK_V)),
        )


class _ChunkGatedDeltaRuleCuTile(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        g,
        beta,
        chunk_size,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    ):
        initial_dtype = query.dtype
        B, T, H, K = query.shape
        V = value.shape[-1]
        num_chunks = (T + chunk_size - 1) // chunk_size
        scale = 1.0 / math.sqrt(K)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        device = query.device
        BLOCK_K = 1 << (K - 1).bit_length()
        solve_steps = max(1, (chunk_size - 1).bit_length())

        # Allocate intermediates (5D). The four (chunk_size, K/V) buffers are the
        # dominant write/read traffic between the intra and inter kernels; store
        # them as bf16 to halve that traffic (the intra kernel is ~memory-bound at
        # low occupancy). g_cum stays fp32 (used as exp(g_last - g_cum) decay).
        # Only reduce precision when the op runs reduced-precision; fp32 inputs
        # keep fp32 intermediates so no accuracy is lost.
        inter_bf16 = initial_dtype in (torch.bfloat16, torch.float16)
        inter_dtype = torch.bfloat16 if inter_bf16 else torch.float32
        q_chunked = torch.empty(B, H, num_chunks, chunk_size, K, device=device, dtype=inter_dtype)
        k_chunked = torch.empty(B, H, num_chunks, chunk_size, K, device=device, dtype=inter_dtype)
        v_corrected = torch.empty(B, H, num_chunks, chunk_size, V, device=device, dtype=inter_dtype)
        k_cumdecay = torch.empty(B, H, num_chunks, chunk_size, K, device=device, dtype=inter_dtype)
        g_cum = torch.empty(B, H, num_chunks, chunk_size, device=device, dtype=torch.float32)
        output_buf = torch.empty(B, H, num_chunks, chunk_size, V, device=device, dtype=torch.float32)

        grid_intra = (B * H, num_chunks, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid_intra,
            _intra_chunk_prepare_kernel,
            (
                query,
                key,
                value,
                beta,
                g,
                q_chunked,
                k_chunked,
                v_corrected,
                k_cumdecay,
                g_cum,
                T,
                K,
                V,
                scale,
                H,
                use_qk_l2norm_in_kernel,
                chunk_size,
                BLOCK_K,
                solve_steps,
                inter_bf16,
            ),
        )

        has_initial_state = initial_state is not None
        if has_initial_state:
            init_state = initial_state.contiguous().float()
        else:
            init_state = torch.empty(1, 1, 1, 1, device=device, dtype=torch.float32)

        final_state = None
        if output_final_state:
            final_state = torch.empty(B, H, K, V, device=device, dtype=torch.float32)

        BLOCK_V = min(128, 1 << (V - 1).bit_length())
        dummy = torch.empty(1, 1, 1, 1, device=device, dtype=torch.float32)

        grid_inter = (B * H, (V + BLOCK_V - 1) // BLOCK_V, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid_inter,
            _chunk_inter_recurrence_kernel,
            (
                q_chunked,
                k_chunked,
                v_corrected,
                k_cumdecay,
                g_cum,
                output_buf,
                init_state if has_initial_state else dummy,
                final_state if output_final_state else dummy,
                num_chunks,
                K,
                V,
                H,
                has_initial_state,
                output_final_state,
                chunk_size,
                BLOCK_K,
                BLOCK_V,
            ),
        )

        output = output_buf.reshape(B, H, num_chunks * chunk_size, V)[:, :, :T, :]
        output = output.transpose(1, 2).contiguous().to(initial_dtype)
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        raise NotImplementedError("backward not implemented")


@register_impl("chunk_gated_delta_rule", backend="cutile")
def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Drop-in cuTile replacement for torch_chunk_gated_delta_rule."""
    return _ChunkGatedDeltaRuleCuTile.apply(
        query,
        key,
        value,
        g,
        beta,
        chunk_size,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )
