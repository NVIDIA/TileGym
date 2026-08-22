# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from typing import Optional
from typing import Tuple

import pytest
import torch
import triton
import triton.language as tl

from tilegym.backend import is_backend_available
from tilegym.backend import set_backend
from tilegym.ops import fmha_variant

from .. import common

USE_FULL_CONFIG = os.environ.get("USE_FULL_CONFIG", "0") == "1"

# Backends exercised by the tests.
_backends = ["triton", "cutile"]

_test_backends = ["triton", "cutile"]


def get_configs_full():
    return [
        (
            batch,
            num_heads,
            num_head_groups,
            max_q_seq_len,
            max_kv_seq_len,
            head_dim,
            is_causal,
            dtype,
            backend,
            window_size,
            use_random_mask,
            bias_type,
            dropout,
            layout,
            varlen,
        )
        for batch in [1, 10, 30]
        for num_heads in [4, 32]
        for num_head_groups in [1, 4]
        for max_q_seq_len in [1, 128, 257]
        for max_kv_seq_len in [256, 1024, 2048]
        for head_dim in [128]
        for is_causal in [True, False]
        for dtype in [torch.bfloat16]
        for backend in _backends
        for window_size in [0, 512]
        for use_random_mask in [True, False]
        for bias_type in [None, "vector", "matrix", "alibi"]
        for dropout in [0.0, 0.1]
        for layout in ["bnsd", "nsbd"]
        for varlen in [True, False]
    ]


def get_configs_test():
    return [
        (
            batch,
            num_heads,
            num_head_groups,
            max_q_seq_len,
            max_kv_seq_len,
            head_dim,
            is_causal,
            dtype,
            backend,
            window_size,
            use_random_mask,
            bias_type,
            dropout,
            layout,
            varlen,
        )
        for batch in [4]
        for num_heads in [32]
        for num_head_groups in [1]
        for max_q_seq_len in [128, 257]
        for max_kv_seq_len in [496, 2064]
        for head_dim in [64, 128]
        for is_causal in [True, False]
        for dtype in [torch.bfloat16]
        for backend in _test_backends
        for window_size, use_random_mask, bias_type, dropout, layout, varlen in [
            (512, False, None, 0.0, "bnsd", False),
            (0, True, None, 0.0, "bnsd", False),
            (0, False, None, 0.0, "bnsd", False),
            (0, False, "vector", 0.0, "bnsd", False),
            (0, False, "matrix", 0.0, "bnsd", False),
            (0, False, "alibi", 0.0, "bnsd", False),
            (0, False, None, 0.1, "bnsd", False),
            (0, False, None, 0.0, "nsbd", False),
            (0, False, None, 0.0, "bnsd", True),
            (512, True, None, 0.0, "nsbd", True),
            (0, False, "alibi", 0.1, "nsbd", True),
            (512, True, "matrix", 0.1, "nsbd", True),
        ]
    ]


@triton.jit
def _generate_dropout_mask_kernel_2d(
    mask_ptr,
    batch,
    num_heads,
    max_q_seq_len,
    max_kv_seq_len,
    dropout_p,
    seed,
    stride_mask_b,
    stride_mask_h,
    stride_mask_q,
    stride_mask_kv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m.to(tl.int64) * BLOCK_M
    n_start = pid_n.to(tl.int64) * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    total_m = batch * num_heads * max_q_seq_len

    m_mask = m_offsets < total_m
    n_mask = n_offsets < max_kv_seq_len

    m_flat = m_offsets
    q_idx = m_flat % max_q_seq_len
    m_flat = m_flat // max_q_seq_len
    head_idx = m_flat % num_heads
    batch_idx = m_flat // num_heads

    batch_idx_2d = batch_idx[:, None]
    head_idx_2d = head_idx[:, None]
    q_idx_2d = q_idx[:, None]
    kv_idx_2d = n_offsets[None, :]

    random_offset = (
        batch_idx_2d * stride_mask_b
        + head_idx_2d * stride_mask_h
        + q_idx_2d * stride_mask_q
        + kv_idx_2d * stride_mask_kv
    )

    random_vals = tl.rand(seed, random_offset)
    keep_mask = random_vals > dropout_p

    memory_offsets = (
        batch_idx_2d * stride_mask_b
        + head_idx_2d * stride_mask_h
        + q_idx_2d * stride_mask_q
        + kv_idx_2d * stride_mask_kv
    )

    valid_mask = m_mask[:, None] & n_mask[None, :]

    tl.store(mask_ptr + memory_offsets, keep_mask.to(tl.uint8), mask=valid_mask)


def _process_attention_output(output, q_lens=None, ref_mask=None, layout="bnsd", dropout_mask=None):
    """
    Process attention output, set invalid positions to 0 to skip comparison

    Args:
        output: Attention output tensor, shape (batch, num_heads, seq_len, head_dim)
        q_lens: Query sequence actual length, shape (batch)
        ref_mask: Reference mask, shape (batch, num_heads, seq_len, kv_seq_len)

    Returns:
        Processed output tensor
    """
    dtype = output.dtype
    if dtype == torch.float8_e5m2:
        output = output.float()

    processed_output = output.clone()

    if q_lens is not None:
        s_dim = layout.find("s")
        output_s_broadcast_shape = [1] * 4
        output_s_broadcast_shape[2] = output.shape[s_dim]  # bnsd
        q_len_mask = (
            torch.arange(output.shape[s_dim], device=output.device).view(output_s_broadcast_shape)
            >= q_lens[:, None, None, None]
        )
        q_len_mask = torch.einsum(f"bnsd->{layout}", q_len_mask).contiguous()
        processed_output.masked_fill_(q_len_mask, 0)

    # Combine ref_mask and dropout_mask with OR logic
    combined_mask = None
    if ref_mask is not None:
        combined_mask = ref_mask
    if dropout_mask is not None:
        # Convert dropout_mask: True(keep) -> False(mask), False(drop) -> True(mask)
        dropout_mask_inverted = ~dropout_mask
        if combined_mask is not None:
            # OR logic: if either mask is True, the position is masked
            combined_mask = torch.logical_or(combined_mask, dropout_mask_inverted)
        else:
            combined_mask = dropout_mask_inverted

    if combined_mask is not None:
        # Check if entire rows are masked (all positions in the last dimension are True)
        fully_masked_rows = combined_mask.all(dim=-1).unsqueeze(-1)  # (batch, num_heads, q_seq_len)

        fully_masked_expanded = (
            torch.einsum(f"bnsd->{layout}", fully_masked_rows).contiguous().expand_as(processed_output)
        )
        processed_output.masked_fill_(fully_masked_expanded, 0)
    return processed_output.to(dtype)


def _generate_dropout_mask(
    batch: int,
    num_heads: int,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    dropout_p: float,
    seed: int,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = torch.cuda.current_device()

    mask = torch.empty((batch, num_heads, max_q_seq_len, max_kv_seq_len), device=device, dtype=torch.uint8)

    BLOCK_M = 64
    BLOCK_N = 64

    total_m = batch * num_heads * max_q_seq_len
    grid_m = triton.cdiv(total_m, BLOCK_M)
    grid_n = triton.cdiv(max_kv_seq_len, BLOCK_N)
    grid = (grid_m, grid_n)

    _generate_dropout_mask_kernel_2d[grid](
        mask,
        batch,
        num_heads,
        max_q_seq_len,
        max_kv_seq_len,
        dropout_p,
        seed,
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return mask.bool()


def _get_shape_by_layout(layout, b, n, s, d):
    block_size_map = {"b": b, "n": n, "s": s, "d": d}
    return [block_size_map[dim] for dim in layout]


def _get_data(*shape, dtype, device, mean=0.1, normal_std=0.2):
    if dtype == torch.float8_e5m2:
        return torch.empty(*shape, dtype=torch.float16, device=device).normal_(mean, normal_std).to(dtype)
    return torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)


def _get_qkv(
    batch,
    q_heads,
    kv_heads,
    q_seq_len,
    kv_seq_len,
    head_dim,
    device,
    dtype,
    mean=0.1,
    normal_std=0.2,
    layout="bnsd",
):
    q = _get_data(
        *_get_shape_by_layout(layout, batch, q_heads, q_seq_len, head_dim),
        dtype=dtype,
        device=device,
        mean=mean,
        normal_std=normal_std,
    )
    k = _get_data(
        *_get_shape_by_layout(layout, batch, kv_heads, kv_seq_len, head_dim),
        dtype=dtype,
        device=device,
        mean=mean,
        normal_std=normal_std,
    )
    v = _get_data(
        *_get_shape_by_layout(layout, batch, kv_heads, kv_seq_len, head_dim),
        dtype=dtype,
        device=device,
        mean=mean,
        normal_std=normal_std,
    )
    return q, k, v


def _generate_causal_mask(
    batch: int,
    num_heads: int,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    device: torch.device,
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate causal mask

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        max_q_seq_len: Maximum query sequence length
        max_kv_seq_len: Maximum key-value sequence length
        device: Device
        q_lens: Query sequence actual length
        kv_lens: Key-value sequence actual length

    Returns:
        Causal mask with shape (batch, num_heads, max_q_seq_len, max_kv_seq_len)
    """
    batch_masks = []
    for b in range(batch):
        q_len = q_lens[b].item() if q_lens is not None else max_q_seq_len
        kv_len = kv_lens[b].item() if kv_lens is not None else max_kv_seq_len
        causal_offset = kv_len - q_len
        assert kv_len >= q_len, f"kv_len ({kv_len}) should >= q_len ({q_len}) for batch {b}"

        # Generate causal mask for current batch
        batch_mask = (
            torch.triu(
                torch.ones((max_q_seq_len, max_kv_seq_len), device=device),
                diagonal=1 + causal_offset,
            )
            .expand(num_heads, max_q_seq_len, max_kv_seq_len)
            .contiguous()
            .bool()
        )
        batch_masks.append(batch_mask)
    return torch.stack(batch_masks, dim=0)


def _generate_window_mask(
    batch: int,
    num_heads: int,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    window_size: int,
    device: torch.device,
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate window mask

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        max_q_seq_len: Maximum query sequence length
        max_kv_seq_len: Maximum key-value sequence length
        window_size: Window size
        device: Device
        q_lens: Query sequence actual length
        kv_lens: Key-value sequence actual length

    Returns:
        Window mask with shape (batch, num_heads, max_q_seq_len, max_kv_seq_len)
    """
    batch_masks = []
    for b in range(batch):
        q_len = q_lens[b].item() if q_lens is not None else max_q_seq_len
        kv_len = kv_lens[b].item() if kv_lens is not None else max_kv_seq_len
        prefix_len = kv_len - q_len

        # Query and key real length indices
        q_indices = torch.arange(q_len, device=device).unsqueeze(1)  # (q_len, 1)
        k_indices = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, kv_len)

        # Query real absolute position
        query_pos = prefix_len + q_indices  # (q_len,1)
        key_pos = k_indices  # (1, kv_len)

        dist = (key_pos - query_pos).abs()  # (q_len, kv_len)

        window_mask = dist > window_size  # True means to mask

        # Pad to max length: initialize all True mask
        full_mask = torch.ones((max_q_seq_len, max_kv_seq_len), device=device, dtype=torch.bool)
        # Replace real window part with computed mask
        full_mask[:q_len, :kv_len] = window_mask

        # Expand to head dimension
        full_mask = full_mask.unsqueeze(0).expand(num_heads, max_q_seq_len, max_kv_seq_len).contiguous()

        batch_masks.append(full_mask)

    return torch.stack(batch_masks, dim=0)


def _get_mask(
    causal: bool = False,
    window_size: int = 0,
    use_random_mask: bool = False,
    max_q_seq_len: int = 0,
    max_kv_seq_len: int = 0,
    batch: int = 0,
    num_heads: int = 0,
    device: torch.device = torch.device("cuda"),
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Generate two masks: one for reference implementation and one for Triton kernel.

    The reference mask combines all requested types (causal, windowed, random).
    The Triton mask is used only when a random pattern is needed; Triton internally
    handles causal and windowed attention.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - ref_mask (torch.Tensor): Mask for the reference implementation.
            - triton_mask (torch.Tensor): Mask for the Triton kernel.
    """
    ref_mask = None
    triton_mask = None

    # Generate causal mask
    if causal is not False:
        ref_mask = _generate_causal_mask(batch, num_heads, max_q_seq_len, max_kv_seq_len, device, q_lens, kv_lens)

    # Generate window mask
    if window_size > 0:
        window_mask = _generate_window_mask(
            batch, num_heads, max_q_seq_len, max_kv_seq_len, window_size, device, q_lens, kv_lens
        )
        ref_mask = torch.logical_or(ref_mask, window_mask) if ref_mask is not None else window_mask

    # Generate random mask
    if use_random_mask:
        triton_mask = torch.randint(
            high=2,
            size=(batch, num_heads, max_q_seq_len, max_kv_seq_len),
            dtype=torch.bool,
            device=device,
        )
        ref_mask = torch.logical_or(ref_mask, triton_mask) if ref_mask is not None else triton_mask

    ref_mask = ref_mask.clone() if ref_mask is not None else None
    triton_mask = triton_mask.clone().to(torch.int8) if triton_mask is not None else None

    return ref_mask, triton_mask


def _get_bias(
    bias_type: Optional[str] = None,
    batch: int = 0,
    num_heads: int = 0,
    max_q_seq_len: int = 0,
    max_kv_seq_len: int = 0,
    out_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
    bias_bwd: bool = False,
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Return:
        triton_bias: bias for Triton kernel
        ref_bias: bias for reference implementation
    """
    dtype = torch.float16 if out_dtype == torch.float8_e5m2 else out_dtype
    if bias_type == "vector":
        bias = (
            torch.empty((batch, num_heads, 1, max_kv_seq_len), dtype=dtype, device=device)
            .normal_(mean=0.3, std=1.2)
            .requires_grad_(requires_grad=bias_bwd)
        )
        triton_bias = ref_bias = bias.to(out_dtype)
    elif bias_type == "matrix":
        bias = (
            torch.empty((batch, num_heads, max_q_seq_len, max_kv_seq_len), dtype=dtype, device=device)
            .normal_(mean=0.3, std=1.2)
            .requires_grad_(requires_grad=bias_bwd)
        )
        triton_bias = ref_bias = bias.to(out_dtype)
    elif bias_type == "alibi":
        alibi_scales = torch.rand(num_heads)
        triton_bias = alibi_scales.to(device).requires_grad_(requires_grad=bias_bwd)

        batch_bias = []
        for b in range(batch):
            q_seq_len = q_lens[b].item() if q_lens is not None else max_q_seq_len
            kv_seq_len = kv_lens[b].item() if kv_lens is not None else max_kv_seq_len
            q_i = torch.arange(1 - q_seq_len, 1).view(-1, 1).mul(-1)
            k_j = torch.arange(1 - kv_seq_len, 1).view(1, -1).mul(-1)
            neg_diag_dist = -(k_j - q_i).abs()
            bias = neg_diag_dist.unsqueeze(0) * alibi_scales.view(-1, 1, 1)
            padded_bias = torch.nn.functional.pad(
                bias,
                (0, max_kv_seq_len - kv_seq_len, 0, max_q_seq_len - q_seq_len),
                mode="constant",
                value=0,
            )
            batch_bias.append(padded_bias)
        ref_bias = torch.stack(batch_bias, dim=0).to(device).requires_grad_(requires_grad=bias_bwd)
    else:
        triton_bias = None
        ref_bias = None
    return triton_bias, ref_bias


def _get_varlen(
    varlen: bool,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    batch: int,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.int32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    q_lens = torch.randint(max_q_seq_len // 2, max_q_seq_len, (batch,), device=device, dtype=dtype) if varlen else None
    if varlen:
        kv_lens = torch.empty((batch,), device=device, dtype=dtype)
        for i in range(batch):
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_seq_len, (1,), device=device, dtype=dtype)
    else:
        kv_lens = None
    return q_lens, kv_lens


class Test_FMHA_variant(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, scaling=None, attention_mask=None, is_causal=False):
        dtype = q.dtype
        if dtype == torch.float8_e5m2:
            q = q.float()
            k = k.float()
            v = v.float()

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal, scale=scaling
        )
        return ref.to(dtype)

    @staticmethod
    def einsum_reference(
        q,
        k,
        v,
        scaling,
        q_lens=None,
        kv_lens=None,
        mask=None,
        bias=None,
        dropout=0.0,
        soft_cap=None,
        layout="bnsd",
        dropout_mask=None,
    ):
        dtype = q.dtype
        if dtype == torch.float8_e5m2:
            q = q.float()
            k = k.float()
            v = v.float()
            if bias is not None:
                bias = bias.float()

        q_layout = layout.replace("s", "i")
        k_layout = layout.replace("s", "j")
        inner_layout = "bnij"
        assert inner_layout[2:] == "ij"
        head_dim = layout.find("n")
        num_heads_q = q.shape[head_dim]
        num_heads_kv = k.shape[head_dim]
        if num_heads_q == num_heads_kv or num_heads_kv == 1:
            pass
        else:
            assert num_heads_q % num_heads_kv == 0
            num_head_groups = int(num_heads_q / num_heads_kv)
            k = torch.repeat_interleave(k, num_head_groups, dim=head_dim)
            v = torch.repeat_interleave(v, num_head_groups, dim=head_dim)

        p = torch.einsum(f"{q_layout},{k_layout}->{inner_layout}", q, k)
        p = p * scaling
        if bias is not None:
            p = p + bias

        # Apply soft cap before masking and softmax
        if soft_cap is not None:
            p = p / soft_cap
            p = torch.tanh(p)
            p = p * soft_cap

        if mask is not None:
            p.masked_fill_(mask[:, :, :, :], torch.finfo(p.dtype).min)

        if q_lens is not None:
            batch_dim = inner_layout.find("b")
            q_len_shape = 4 * [1]
            q_len_shape[batch_dim] = -1

            q_seq_dim = inner_layout.find("i")
            q_seq_idx_shape = 4 * [1]
            q_seq_idx_shape[q_seq_dim] = -1

            q_seq_idx = torch.arange(p.shape[q_seq_dim], device=p.device)
            q_len_mask = q_lens.view(q_len_shape) <= q_seq_idx.view(q_seq_idx_shape)
            p.masked_fill_(q_len_mask.expand_as(p), -float("inf"))

        if kv_lens is not None:
            batch_dim = inner_layout.find("b")
            kv_len_shape = 4 * [1]
            kv_len_shape[batch_dim] = -1

            kv_seq_dim = inner_layout.find("j")
            kv_seq_idx_shape = 4 * [1]
            kv_seq_idx_shape[kv_seq_dim] = -1

            kv_seq_idx = torch.arange(p.shape[kv_seq_dim], device=p.device)
            kv_len_mask = kv_lens.view(kv_len_shape) <= kv_seq_idx.view(kv_seq_idx_shape)
            p.masked_fill_(kv_len_mask.expand_as(p), -float("inf"))

        p = torch.softmax(p, dim=inner_layout.find("j"), dtype=torch.float32).to(v.dtype)

        if q_lens is not None:
            p = p.masked_fill(q_len_mask.expand_as(p), 0)
        if dropout > 0.0:
            p = torch.where(dropout_mask, p, 0.0) * float(1.0 / (1.0 - dropout))
        v_layout = layout.replace("s", "j")
        ref_out = torch.einsum(f"{inner_layout},{v_layout}->{q_layout}", p, v)
        return ref_out.to(dtype)

    @pytest.mark.parametrize(
        "batch, num_heads, num_head_groups, max_q_seq_len, max_kv_seq_len, "
        "head_dim, is_causal, dtype, backend, window_size, "
        "use_random_mask, bias_type, dropout, layout, varlen",
        get_configs_full() if USE_FULL_CONFIG else get_configs_test(),
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    def test_op(
        self,
        batch,
        num_heads,
        num_head_groups,
        max_q_seq_len,
        max_kv_seq_len,
        head_dim,
        is_causal,
        dtype,
        backend,
        arch,
        window_size,
        use_random_mask,
        bias_type,
        dropout,
        layout,
        varlen,
    ):
        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")
        if backend == "cutile":
            if not is_backend_available("cutile"):
                pytest.skip("Cutile backend not available")
            # Skip dropout tests for cutile - dropout requires deterministic random generation
            # which is complex to implement correctly in CuTile
            if dropout > 0:
                pytest.skip("Cutile does not support dropout with deterministic random generation")

        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        if max_q_seq_len > max_kv_seq_len:
            pytest.skip("max_q_seq_len should <= max_kv_seq_len")
        if num_heads % num_head_groups != 0:
            pytest.skip("num_heads should be divisible by num_head_groups")

        # Create random input tensors
        self.setUp()
        device = torch.device("cuda")
        seed = torch.random.initial_seed()

        kv_heads = num_head_groups
        q, k, v = _get_qkv(
            batch,
            num_heads,
            kv_heads,
            max_q_seq_len,
            max_kv_seq_len,
            head_dim,
            device,
            dtype,
            layout=layout,
        )
        q_lens, kv_lens = _get_varlen(varlen, max_q_seq_len, max_kv_seq_len, batch, device)
        ref_mask, triton_mask = _get_mask(
            causal=is_causal,
            window_size=window_size,
            use_random_mask=use_random_mask,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            batch=batch,
            num_heads=num_heads,
            device=device,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )
        triton_bias, ref_bias = _get_bias(
            bias_type,
            batch,
            num_heads,
            max_q_seq_len,
            max_kv_seq_len,
            dtype,
            device,
            False,
            q_lens,
            kv_lens,
        )
        dropout_mask = _generate_dropout_mask(batch, num_heads, max_q_seq_len, max_kv_seq_len, dropout, seed, device)

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        os.environ["TILEIR_ENABLE_FTZ"] = "1"
        os.environ["TILEIR_ENABLE_APPROX"] = "1"
        self.assertCorrectness(
            fmha_variant,
            self.einsum_reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "q_lens": q_lens,
                "kv_lens": kv_lens,
                "scaling": sm_scale,
                "dropout": dropout,
                "layout": layout,
            },
            extra_ref_kwargs={
                "mask": ref_mask,
                "bias": ref_bias,
                "dropout_mask": dropout_mask,
            },
            extra_test_kwargs={
                "random_mask": triton_mask,
                "is_causal": is_causal,
                "bias": triton_bias,
                "bias_type": bias_type,
                "seed": seed,
                "window_size": window_size,
            },
            atol=1e-1,
            rtol=1e-1,
            check_stride=False,
            output_processor=lambda ind, output, fn_kwargs, extra_test_kwargs, extra_ref_kwargs: (
                output
                if ind != 0
                else _process_attention_output(
                    output,
                    fn_kwargs["q_lens"],
                    extra_ref_kwargs["mask"],
                    fn_kwargs["layout"],
                    extra_ref_kwargs["dropout_mask"],
                )
            ),
        )

    @pytest.mark.parametrize(
        "batch,heads,seq_len,head_dim,dtype",
        [(4, 32, seq_len, 128, dtype) for dtype in [torch.float16, torch.float8_e5m2] for seq_len in [2**9, 2**13]],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("is_causal", [True])
    @pytest.mark.parametrize(
        "variant_config",
        [
            {},
            {"window_size": 512},
            {"use_random_mask": True},
            {"bias_type": "vector"},
            {"bias_type": "matrix"},
            {"bias_type": "alibi"},
            {"dropout": 0.1},
            {"layout": "nsbd"},
            {"window_size": 512, "bias_type": "matrix"},
            {"dropout": 0.1, "bias_type": "vector"},
            {"layout": "bnsd"},
        ],
        ids=lambda config: f"variant_{'_'.join(f'{k}_{v}' for k, v in sorted(config.items()))}" if config else "base",
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_perf_variants(
        self,
        batch,
        heads,
        seq_len,
        head_dim,
        dtype,
        is_causal,
        backend,
        variant_config,
        record_property,
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if torch.cuda.get_device_capability() in [(12, 0), (12, 1)] and seq_len == 2**13:
            pytest.skip("due to OOM")

        self.setUp()
        device = torch.device("cuda")
        seed = torch.random.initial_seed()
        window_size = variant_config.get("window_size", 0)
        use_random_mask = variant_config.get("use_random_mask", False)
        bias_type = variant_config.get("bias_type", None)
        dropout = variant_config.get("dropout", 0.0)
        layout = variant_config.get("layout", "bnsd")
        varlen = variant_config.get("varlen", False)

        if backend == "triton" and bias_type == "matrix" and seq_len == 2**13:
            pytest.xfail(
                "wrong result on the Triton-TileIR backend with the default BLOCK_M=256 BLOCK_N=128 "
                "(correct with BLOCK_M=64 BLOCK_N=64)"
            )
        if backend == "triton" and dropout > 0.0:
            pytest.xfail("dropout performance drop from tl.rand on the Triton-TileIR backend")

        q, k, v = _get_qkv(batch, heads, heads, seq_len, seq_len, head_dim, device, dtype, layout=layout)

        q_lens, kv_lens = _get_varlen(varlen, seq_len, seq_len, batch, device)

        ref_mask, triton_mask = _get_mask(
            causal=is_causal,
            window_size=window_size,
            use_random_mask=use_random_mask,
            max_q_seq_len=seq_len,
            max_kv_seq_len=seq_len,
            batch=batch,
            num_heads=heads,
            device=device,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )

        triton_bias, ref_bias = _get_bias(
            bias_type, batch, heads, seq_len, seq_len, dtype, device, False, q_lens, kv_lens
        )

        dropout_mask = _generate_dropout_mask(batch, heads, seq_len, seq_len, dropout, seed, device)

        sm_scale = 1.0 / math.sqrt(head_dim)
        os.environ["TILEIR_ENABLE_FTZ"] = "1"
        os.environ["TILEIR_ENABLE_APPROX"] = "1"

        if backend == "triton":
            set_backend("triton")
            backend_fn = lambda: fmha_variant(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                kv_lens=kv_lens,
                scaling=sm_scale,
                dropout=dropout,
                layout=layout,
                random_mask=triton_mask,
                is_causal=is_causal,
                bias=triton_bias,
                bias_type=bias_type,
                seed=seed,
                window_size=window_size,
            )
        elif backend == "pytorch":
            backend_fn = lambda: self.einsum_reference(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                kv_lens=kv_lens,
                scaling=sm_scale,
                dropout=dropout,
                layout=layout,
                mask=ref_mask,
                bias=ref_bias,
                dropout_mask=dropout_mask,
            )
        elif backend == "cutile":
            if not is_backend_available("cutile"):
                pytest.skip("Cutile backend not available")
            if dtype == torch.float8_e5m2:
                pytest.skip("Skip float8_e5m2 due to cutile not support float8")
            if dropout > 0:
                # _tile_fmha_variant raises NotImplementedError for dropout.
                pytest.skip("Cutile does not support dropout with deterministic random generation")
            if seq_len >= 2**13 and (bias_type == "matrix" or use_random_mask):
                # A (batch, heads, 8192, 8192) bias/mask overflows the stride of the
                # TILE_M=256/TILE_N=128 config, so autotune ends with "No valid config
                # found in search space".
                pytest.skip("Cutile: 8192-seqlen matrix bias / random mask overflows the tile stride")
            set_backend("cutile")
            backend_fn = lambda: fmha_variant(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                kv_lens=kv_lens,
                scaling=sm_scale,
                dropout=dropout,
                layout=layout,
                random_mask=triton_mask,
                is_causal=is_causal,
                bias=triton_bias,
                bias_type=bias_type,
                seed=seed,
                window_size=window_size,
            )
        elif backend == "tilecpp":
            if not is_backend_available("tilecpp"):
                pytest.skip("TileCpp backend not available")
            if dtype == torch.float8_e5m2:
                pytest.skip("Skip float8_e5m2 due to tilecpp not support float8")
            if dropout > 0:
                pytest.skip("TileCpp does not support dropout with deterministic random generation")
            if seq_len >= 2**13 and (bias_type == "matrix" or use_random_mask):
                # 8192-seqlen matrix bias and random-mask shapes are outside the
                # shape set this backend is validated on.
                pytest.skip("TileCpp: 8192-seqlen matrix bias / random mask is not covered")
            set_backend("tilecpp")
            backend_fn = lambda: fmha_variant(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                kv_lens=kv_lens,
                scaling=sm_scale,
                dropout=dropout,
                layout=layout,
                random_mask=triton_mask,
                is_causal=is_causal,
                bias=triton_bias,
                bias_type=bias_type,
                seed=seed,
                window_size=window_size,
            )
        else:
            pytest.skip(f"Backend {backend} not supported")

        if backend != "pytorch":
            if dtype == torch.float8_e5m2:
                atol = 3
                rtol = 0
            else:
                atol = 1e-1
                rtol = 1e-1
            self.assertCorrectness(
                backend_fn,
                lambda: self.einsum_reference(
                    q=q,
                    k=k,
                    v=v,
                    q_lens=q_lens,
                    kv_lens=kv_lens,
                    scaling=sm_scale,
                    dropout=dropout,
                    layout=layout,
                    mask=ref_mask,
                    bias=ref_bias,
                    dropout_mask=dropout_mask,
                ),
                kwargs={},
                atol=atol,
                rtol=rtol,
                check_stride=False,
                output_processor=lambda ind, output, fn_kwargs, extra_test_kwargs, extra_ref_kwargs: (
                    output if ind != 0 else _process_attention_output(output, q_lens, ref_mask, layout, dropout_mask)
                ),
            )

        result = common.benchmark_framework(backend, backend_fn, min_rep=50)
        record_property("benchmark", result)
