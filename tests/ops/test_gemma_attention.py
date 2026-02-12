# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for gemma-specific attention implementation"""

import math
import os
from typing import Optional
from typing import Tuple

import pytest
import torch

from tilegym.backend import set_backend
from tilegym.ops import gemma_attention

from .. import common


def _get_data(*shape, dtype, device, mean=0.1, normal_std=0.5):
    """Generate random test data"""
    return torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)


def _get_qkv(
    batch: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Q, K, V tensors for testing.

    Returns tensors in BNSD layout: [batch, num_heads, seq_len, head_dim]
    """
    q = _get_data(batch, num_heads_q, seq_len_q, head_dim, dtype=dtype, device=device)
    k = _get_data(batch, num_heads_kv, seq_len_kv, head_dim, dtype=dtype, device=device)
    v = _get_data(batch, num_heads_kv, seq_len_kv, head_dim, dtype=dtype, device=device)
    return q, k, v


def _generate_causal_mask(
    batch: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate causal mask for self-attention.

    Returns:
        Boolean mask [batch, num_heads, seq_len, seq_len] where True = mask out
    """
    # Upper triangular mask (diagonal=1 means exclude diagonal itself)
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )
    # Expand to [batch, num_heads, seq_len, seq_len]
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1).contiguous()


def _generate_window_mask(
    batch: int,
    num_heads: int,
    seq_len: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate sliding window mask.

    Args:
        window_size: Tokens within [-window_size, +window_size] can attend to each other

    Returns:
        Boolean mask [batch, num_heads, seq_len, seq_len] where True = mask out
    """
    # Create position indices
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

    # Calculate absolute distance
    dist = (k_pos - q_pos).abs()  # [seq_len, seq_len]

    # Mask positions outside the window
    window_mask = dist > window_size  # True = mask out

    # Expand to [batch, num_heads, seq_len, seq_len]
    return window_mask.unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1).contiguous()


def _get_mask(
    batch: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
    is_causal: bool = False,
    window_size: int = 0,
) -> Optional[torch.Tensor]:
    """
    Generate combined attention mask.

    Args:
        batch: Batch size
        num_heads: Number of heads
        seq_len: Sequence length
        device: Target device
        is_causal: Whether to apply causal masking
        window_size: Sliding window size (0 = no window)

    Returns:
        Boolean mask [batch, num_heads, seq_len, seq_len] where True = mask out
        Returns None if no masking is needed
    """
    mask = None

    # Causal mask
    if is_causal:
        mask = _generate_causal_mask(batch, num_heads, seq_len, device)

    # Sliding window mask
    if window_size > 0:
        window_mask = _generate_window_mask(batch, num_heads, seq_len, window_size, device)
        if mask is not None:
            mask = torch.logical_or(mask, window_mask)
        else:
            mask = window_mask

    return mask


def einsum_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    mask: Optional[torch.Tensor] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for gemma attention.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        scaling: Attention scaling factor (typically 1/sqrt(head_dim))
        mask: Boolean mask [batch, num_heads, seq_len, seq_len] where True = mask out
        soft_cap: Soft cap value for logits (None for no soft cap)

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    dtype = q.dtype
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]

    # Handle GQA: repeat k/v if needed
    if num_heads_q != num_heads_kv and num_heads_kv != 1:
        assert num_heads_q % num_heads_kv == 0
        num_head_groups = num_heads_q // num_heads_kv
        k = torch.repeat_interleave(k, num_head_groups, dim=1)
        v = torch.repeat_interleave(v, num_head_groups, dim=1)

    # Compute attention scores: [batch, num_heads, seq_len, seq_len]
    # Using einsum format: "bnid,bnjd->bnij"
    p = torch.einsum("bnid,bnjd->bnij", q, k)
    p = p * scaling

    # Apply soft cap BEFORE masking (critical order!)
    if soft_cap is not None:
        p = p / soft_cap
        p = torch.tanh(p)
        p = p * soft_cap

    # Apply mask using -inf for numerical stability
    if mask is not None:
        p = p.masked_fill(mask, torch.finfo(p.dtype).min)

    # Softmax with float32 for numerical stability, then cast back
    p = torch.softmax(p, dim=-1, dtype=torch.float32).to(v.dtype)

    # Compute output: [batch, num_heads, seq_len, head_dim]
    # Using einsum format: "bnij,bnjd->bnid"
    output = torch.einsum("bnij,bnjd->bnid", p, v)

    return output.to(dtype)


class TestGemmaAttention(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "batch_size, num_heads, num_kv_heads, seq_len, head_dim, window_size, soft_cap, is_causal, dtype",
        [
            # Basic tests
            # (1, 8, 8, 128, 64, 0, None, True, torch.float16),  # No window, no soft cap
            # (1, 8, 8, 128, 64, 64, None, True, torch.float16),  # Window only
            (1, 8, 8, 128, 64, 64, 10.0, True, torch.float16),  # Both soft cap and window
            # GQA tests
            # (2, 8, 4, 256, 128, 128, 10.0, True, torch.float16),  # 2:1 GQA ratio
            (1, 12, 4, 512, 64, 256, 30.0, True, torch.bfloat16),  # 3:1 GQA ratio
            # Non-causal
            (1, 8, 8, 128, 64, 0, None, False, torch.float16),  # Non-causal, no window
            (1, 8, 8, 128, 64, 64, 10.0, False, torch.float32),  # Non-causal with features (critical test case)
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        window_size,
        soft_cap,
        is_causal,
        dtype,
        backend,
        arch,
    ):
        """Test gemma attention correctness against pure PyTorch reference"""
        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")

        self.setUp()
        set_backend(backend)
        device = torch.device("cuda")

        # Create test data
        q, k, v = _get_qkv(batch_size, num_heads, num_kv_heads, seq_len, seq_len, head_dim, device, dtype)
        scaling = 1.0 / math.sqrt(head_dim)

        # Generate mask
        ref_mask = _get_mask(batch_size, num_heads, seq_len, device, is_causal, window_size)

        # Run test
        self.assertCorrectness(
            lambda: gemma_attention(
                q=q,
                k=k,
                v=v,
                scaling=scaling,
                is_causal=is_causal,
                soft_cap=soft_cap,
                window_size=window_size,
                backend=backend,
            ),
            lambda: einsum_reference(q=q, k=k, v=v, scaling=scaling, mask=ref_mask, soft_cap=soft_cap),
            kwargs={},
            atol=1e-2,
            rtol=1e-2,
            check_stride=False,
        )
