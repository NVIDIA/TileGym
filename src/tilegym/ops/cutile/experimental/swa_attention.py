# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Sliding window attention (SWA) prefill kernel.
#
# Each query attends to at most W keys behind it (plus causal mask).
# Online softmax with exp2 + FTZ for SFU utilization.
# 2D grid: bid(0) = Q tile block, bid(1) = batch*head.
#
# Autotuned on B300: M=64, N=128, occ=2, fast precision.
# Validated with in-house autotuner + NVBench cold measurements.

import math

import cuda.tile as ct
import torch
import torch.nn.functional as F

from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
INV_LOG_2 = 1.0 / math.log(2)  # pre-computed for exp2-based softmax
_NEG_INF = -1e30  # sentinel for masked-out attention positions


def _cdiv(a, b):
    return (a + b - 1) // b


# occupancy=2 keeps two CTAs resident per SM for latency hiding.
# the @experimental_kernel tag prints a one-time notice on first launch.
@experimental_kernel
@ct.kernel(occupancy=2)
def swa_fwd_kernel(
    Q,
    K,
    V,
    Out,
    qk_scale: float,
    seq_k: int,
    window_size: int,
    stride_q: int,  # number of Q tiles per head (for flat-buffer indexing)
    stride_kv: int,  # number of KV tiles per head
    TILE_M: ConstInt,  # tile rows (query)
    TILE_N: ConstInt,  # tile cols (key/value)
    TILE_D: ConstInt,  # head dimension
    CAUSAL: ConstBool,
):
    # 2D grid: bid(0) iterates over Q-tile blocks, bid(1) over batch*heads
    q_block = ct.bid(0)
    off_hz = ct.bid(1)

    # map block IDs to flat-buffer offsets so each head's tiles
    # don't bleed into the next head's memory region
    q_start = q_block * TILE_M
    q_offset = off_hz * stride_q + q_block
    kv_base = off_hz * stride_kv

    q_tile = ct.load(Q, index=(q_offset, 0), shape=(TILE_M, TILE_D), padding_mode=ct.PaddingMode.ZERO)

    # online softmax running state: max, log-sum-exp, accumulator
    m_i = ct.full((TILE_M,), _NEG_INF, dtype=ct.float32)
    l_i = ct.zeros((TILE_M,), dtype=ct.float32)
    acc = ct.zeros((TILE_M, TILE_D), dtype=ct.float32)

    # convert scale to log2 space so we can use exp2 (maps to SFU hw)
    scale_log2 = qk_scale * INV_LOG_2
    offs_m = ct.arange(TILE_M, dtype=ct.int32) + q_start

    # compute the KV block range that intersects with the sliding window.
    # this is where the O(S*W) complexity comes from -- we skip blocks
    # that are entirely outside the window instead of iterating over all S.
    kv_lo = max(0, q_start - window_size + 1) // TILE_N
    kv_hi = _cdiv(seq_k, TILE_N)
    if CAUSAL:
        kv_hi = min(kv_hi, (q_start + TILE_M - 1) // TILE_N + 1)

    for kv_block in range(kv_lo, kv_hi):
        kv_start = kv_block * TILE_N

        k_tile = ct.load(K, index=(kv_base + kv_block, 0), shape=(TILE_N, TILE_D), padding_mode=ct.PaddingMode.ZERO)
        v_tile = ct.load(V, index=(kv_base + kv_block, 0), shape=(TILE_N, TILE_D), padding_mode=ct.PaddingMode.ZERO)

        # QK^T matmul for this tile pair
        qk = ct.mma(q_tile, ct.transpose(k_tile), ct.zeros((TILE_M, TILE_N), dtype=ct.float32))

        # three-part mask: trailing window, causal upper-triangle, seq bounds
        offs_n = ct.arange(TILE_N, dtype=ct.int32) + kv_start
        offs_n_2d = ct.expand_dims(offs_n, axis=0)
        offs_m_2d = ct.expand_dims(offs_m, axis=1)

        mask = offs_n_2d > (offs_m_2d - window_size)  # trailing window
        if CAUSAL:
            mask = mask & (offs_n_2d <= offs_m_2d)  # causal (no future keys)
        mask = mask & (offs_n_2d < seq_k)  # don't read past actual seq len

        qk = ct.where(mask, qk, ct.full((TILE_M, TILE_N), _NEG_INF, dtype=ct.float32))

        # online softmax: rescale running state by exp2(old_max - new_max).
        # exp2 + flush_to_zero maps directly to the GPU SFU hardware.
        m_new = ct.maximum(m_i, ct.max(qk * scale_log2, axis=1))
        alpha = ct.exp2(m_i - m_new, flush_to_zero=True)
        p = ct.exp2(qk * scale_log2 - ct.expand_dims(m_new, axis=1), flush_to_zero=True)

        # update running sum and weighted accumulator
        l_i = alpha * l_i + ct.sum(p, axis=1)
        p_fp16 = ct.astype(p, ct.float16)  # downcast for the PV matmul
        acc = ct.expand_dims(alpha, axis=1) * acc + ct.mma(p_fp16, v_tile, ct.zeros((TILE_M, TILE_D), dtype=ct.float32))
        m_i = m_new

    # final normalization: divide accumulated values by softmax denominator.
    # clamp l_i away from zero so a fully-masked row (all keys outside the
    # window) produces zeros rather than NaN.
    l_i = ct.maximum(l_i, ct.full((TILE_M,), 1e-6, dtype=ct.float32))
    out = acc / ct.expand_dims(l_i, axis=1)
    ct.store(Out, index=(q_offset, 0), tile=ct.astype(out, ct.float16))


# -- host launcher --

_DEFAULT_TILE_M = 64
_DEFAULT_TILE_N = 128  # autotuned: 1.9x faster than N=64 on B300


def tile_swa_attention(q, k, v, window_size, scaling=None, is_causal=True, **kwargs):
    # q: (B, H, S_Q, D), k/v: (B, H_K, S_K, D) -- fp16
    if q.dtype not in (torch.float16,):
        raise ValueError(f"SWA kernel requires fp16 input, got {q.dtype}")

    B, H, S_Q, D = q.shape
    _, H_K, S_K, _ = k.shape

    if scaling is None:
        scaling = 1.0 / math.sqrt(D)
    if window_size <= 0:
        window_size = S_K  # non-positive W means full causal

    # expand KV heads for GQA (Mistral uses 8 KV heads for 32 Q heads)
    if H_K != H:
        if H_K > H or H % H_K != 0:
            raise ValueError(
                f"Invalid GQA head configuration: query heads H={H} must be an integer multiple of KV heads H_K={H_K}."
            )
        kv_repeat = H // H_K
        k = k.repeat_interleave(kv_repeat, dim=1)
        v = v.repeat_interleave(kv_repeat, dim=1)

    TILE_M = _DEFAULT_TILE_M
    TILE_N = _DEFAULT_TILE_N

    # compute strides: how many tiles each head occupies in the flat buffer
    stride_q = _cdiv(S_Q, TILE_M)
    stride_kv = _cdiv(S_K, TILE_N)
    S_Q_padded = stride_q * TILE_M
    S_K_padded = stride_kv * TILE_N

    # flatten (B, H, S, D) -> (B*H, S, D) for contiguous tile indexing
    q_3d = q.reshape(B * H, S_Q, D)
    k_3d = k.reshape(B * H, S_K, D)
    v_3d = v.reshape(B * H, S_K, D)

    # pad seq dim to tile boundary so tile loads don't cross head boundaries
    if S_Q_padded != S_Q:
        q_3d = F.pad(q_3d, (0, 0, 0, S_Q_padded - S_Q))
    if S_K_padded != S_K:
        k_3d = F.pad(k_3d, (0, 0, 0, S_K_padded - S_K))
        v_3d = F.pad(v_3d, (0, 0, 0, S_K_padded - S_K))

    # reshape to (B*H*S_padded, D) -- the kernel indexes this as a 2D tile grid
    q_flat = q_3d.reshape(-1, D).contiguous()
    k_flat = k_3d.reshape(-1, D).contiguous()
    v_flat = v_3d.reshape(-1, D).contiguous()
    out_flat = torch.empty_like(q_flat)

    ct.launch(
        torch.cuda.current_stream(),
        (stride_q, B * H, 1),
        swa_fwd_kernel,
        (
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            scaling,
            S_K,
            window_size,
            stride_q,
            stride_kv,
            TILE_M,
            TILE_N,
            D,
            is_causal,
        ),
    )

    # strip padding and reshape back to (B, H, S_Q, D)
    out_3d = out_flat.reshape(B * H, S_Q_padded, D)[:, :S_Q, :]
    return out_3d.reshape(B, H, S_Q, D).contiguous()


# register as the cutile backend for the "swa_attention" dispatch key
register_impl("swa_attention", backend="cutile")(tile_swa_attention)


# -- HuggingFace model integration --


def get_swa_fmha_interface(window_size=4096, backend=None):
    """Returns a drop-in replacement for ALL_ATTENTION_FUNCTIONS["sdpa"].

    Prefill uses the cuTile SWA kernel; decode falls back to SDPA since
    our kernel doesn't track absolute position for KV-cache scenarios.
    """

    def swa_fmha_wrapper(module, q, k, v, attention_mask=None, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1))
        if is_causal is None:
            is_causal = True

        # decode (single token) -- our kernel is a prefill kernel, so we
        # fall back to PyTorch SDPA for autoregressive decode steps.
        # also need to expand KV heads for GQA since SDPA expects matched dims.
        if q.size(-2) == 1:
            if k.size(1) != q.size(1):
                q_heads = q.size(1)
                kv_heads = k.size(1)
                if kv_heads > q_heads or q_heads % kv_heads != 0:
                    raise ValueError(
                        f"decode path requires q head count to be a multiple of k/v head count, "
                        f"got q_heads={q_heads}, kv_heads={kv_heads}"
                    )
                n_rep = q_heads // kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            # cuDNN backend can fail on some GPUs (e.g. B300), try flash then math
            for be in [torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.MATH]:
                try:
                    with torch.nn.attention.sdpa_kernel(be):
                        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                    return o.transpose(1, 2).contiguous(), None
                except RuntimeError:
                    continue
            raise RuntimeError("no working SDPA backend for decode")

        # prefill -- fall back to SDPA for unsupported cases (padded batches
        # or training with dropout) since our kernel doesn't handle them.
        if attention_mask is not None or dropout != 0.0:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal
            ).transpose(1, 2).contiguous(), None

        # try to read window size from the model's config (e.g.
        # MistralConfig.sliding_window), fall back to the user-supplied default
        w = getattr(getattr(module, "config", None), "sliding_window", None)
        if w is None or w is False:
            w = window_size
        if w is None:
            w = k.size(-2)  # no window at all, full causal

        from tilegym.ops import swa_attention as _swa

        o = _swa(q.half(), k.half(), v.half(), window_size=w, scaling=scaling, is_causal=is_causal, backend=backend)
        return o.transpose(1, 2).contiguous(), None

    return swa_fmha_wrapper


def apply_tilegym_swa_to_mistral(window_size=4096, use_cutile=True):
    """Monkey-patch Mistral to route attention through the SWA kernel.

    Call before model creation. Same pattern as TileGym's existing
    apply_tilegym_kernel_to_llama / apply_tilegym_kernel_to_mistral.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.mistral import modeling_mistral

    if use_cutile:
        from tilegym.backend import set_backend

        set_backend("cutile")

    # replace the default SDPA attention function with our SWA wrapper
    ALL_ATTENTION_FUNCTIONS["sdpa"] = get_swa_fmha_interface(window_size=window_size)

    # also patch RoPE, RMSNorm, and SwiGLU if the tilegym kernels are available.
    # these are optional -- attention is the main target.
    try:
        from tilegym.ops import get_apply_rope_func
        from tilegym.ops import get_rms_norm_module
        from tilegym.ops import get_swiglu_module

        modeling_mistral.apply_rotary_pos_emb = get_apply_rope_func(model="llama")
        modeling_mistral.MistralRMSNorm = get_rms_norm_module()
        modeling_mistral.MistralMLP = get_swiglu_module()
    except ImportError:
        pass
    except Exception:
        raise
