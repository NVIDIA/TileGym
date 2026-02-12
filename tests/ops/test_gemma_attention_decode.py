# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Gemma attention decode implementation with soft cap and sliding window support"""

import math
import os
from typing import Optional
from typing import Tuple

import pytest
import torch

import tilegym
from tilegym.backend import set_backend
from tilegym.ops import gemma_attention_decode

from .. import common


def _get_data(
    *shape,
    dtype=torch.float32,
    device="cuda",
    mean=0.0,
    normal_std=0.5,
):
    """Generate random test data"""
    return torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)


def _get_qkv_decode(
    batch: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len_kv: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Q, K, V tensors for decode testing.

    For decode phase:
    - Q has seq_len = 1 (single token generation)
    - K, V have seq_len = seq_len_kv (KV cache)

    Returns tensors in BNSD layout: [batch, num_heads, seq_len, head_dim]
    """
    # Q: [batch, num_heads_q, 1, head_dim]
    q = _get_data(batch, num_heads_q, 1, head_dim, dtype=dtype, device=device)
    # K: [batch, num_heads_kv, seq_len_kv, head_dim]
    k = _get_data(batch, num_heads_kv, seq_len_kv, head_dim, dtype=dtype, device=device)
    # V: [batch, num_heads_kv, seq_len_kv, head_dim]
    v = _get_data(batch, num_heads_kv, seq_len_kv, head_dim, dtype=dtype, device=device)
    return q, k, v


def einsum_reference_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    window_size: int = 0,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for Gemma attention decode.

    For decode phase:
    - Q: [batch, num_heads, 1, head_dim]
    - K: [batch, num_kv_heads, seq_len, head_dim]
    - V: [batch, num_kv_heads, seq_len, head_dim]
    - Output: [batch, num_heads, 1, head_dim]

    Args:
        q: Query tensor [batch, num_heads, 1, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        scaling: Attention scaling factor (typically 1/sqrt(head_dim))
        window_size: Sliding window size (0 for global attention)
        soft_cap: Soft cap value for logits (None for no soft cap)

    Returns:
        Output tensor [batch, num_heads, 1, head_dim]
    """
    dtype = q.dtype
    batch, num_heads_q, _, head_dim = q.shape
    num_heads_kv = k.shape[1]
    seq_len = k.shape[2]

    # Handle GQA: repeat k/v if needed
    if num_heads_q != num_heads_kv and num_heads_kv != 1:
        assert num_heads_q % num_heads_kv == 0
        num_head_groups = num_heads_q // num_heads_kv
        k = torch.repeat_interleave(k, num_head_groups, dim=1)
        v = torch.repeat_interleave(v, num_head_groups, dim=1)

    # Compute attention scores: [batch, num_heads, 1, seq_len]
    # Using einsum format: "bnid,bnjd->bnij" where i=1 for decode
    p = torch.einsum("bnid,bnjd->bnij", q, k)
    p = p * scaling

    # Apply soft cap BEFORE masking (critical order!)
    if soft_cap is not None:
        p = p / soft_cap
        p = torch.tanh(p)
        p = p * soft_cap

    # Apply sliding window mask for decode
    # For decode, query position is at seq_len (the new token being generated)
    # We want KV positions in range [seq_len - window_size, seq_len - 1]
    if window_size > 0:
        # Create mask: True for positions to mask out
        kv_positions = torch.arange(seq_len, device=q.device)
        query_pos = seq_len - 1  # Query is attending from the last position
        # Mask out positions outside the window
        mask = kv_positions < (query_pos - window_size)
        mask = mask.view(1, 1, 1, seq_len).expand(batch, num_heads_q, 1, -1)
        p = p.masked_fill(mask, torch.finfo(p.dtype).min)

    # Softmax with float32 for numerical stability, then cast back
    p = torch.softmax(p, dim=-1, dtype=torch.float32).to(v.dtype)

    # Compute output: [batch, num_heads, 1, head_dim]
    # Using einsum format: "bnij,bnjd->bnid" where i=1 for decode
    output = torch.einsum("bnij,bnjd->bnid", p, v)

    return output.to(dtype)


class TestGemmaAttentionDecode(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "batch_size, num_heads, num_kv_heads, seq_len_kv, head_dim, window_size, soft_cap, dtype",
        [
            # Basic tests - no window, no soft cap
            (1, 8, 8, 128, 64, 0, None, torch.float16),
            # (1, 8, 8, 256, 64, 0, None, torch.float16),
            # (2, 8, 8, 512, 128, 0, None, torch.float16),
            # # Soft cap only
            (1, 8, 8, 128, 64, 0, 10.0, torch.float16),
            # (1, 8, 8, 256, 64, 0, 50.0, torch.float16),
            # (2, 8, 8, 512, 128, 0, 30.0, torch.bfloat16),
            # # Window only
            # (1, 8, 8, 256, 64, 64, None, torch.float16),
            # (1, 8, 8, 512, 64, 128, None, torch.float16),
            # # Both soft cap and window
            (1, 8, 8, 256, 64, 64, 10.0, torch.float16),
            # (2, 8, 8, 512, 128, 128, 50.0, torch.bfloat16),
            # # GQA tests (multiple query heads per KV head)
            # (1, 8, 4, 256, 64, 0, None, torch.float16),  # 2:1 GQA ratio
            # (1, 8, 4, 256, 64, 0, 10.0, torch.float16),  # 2:1 GQA with soft cap
            (1, 8, 4, 256, 64, 64, 10.0, torch.float16),  # 2:1 GQA with both
            # (2, 12, 4, 512, 128, 128, 30.0, torch.bfloat16),  # 3:1 GQA ratio
            # (1, 16, 4, 256, 64, 64, 50.0, torch.float16),  # 4:1 GQA ratio
            # # Longer sequences
            (1, 8, 8, 10024, 64, 0, None, torch.float16),
            (1, 8, 8, 10019, 64, 256, 50.0, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op(
        self,
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len_kv,
        head_dim,
        window_size,
        soft_cap,
        dtype,
        framework: str,
        arch,
    ):
        """Test Gemma attention decode correctness against pure PyTorch reference"""
        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")
        self.setUp()
        set_backend(framework)
        device = torch.device("cuda")

        # Create test data
        q, k, v = _get_qkv_decode(batch_size, num_heads, num_kv_heads, seq_len_kv, head_dim, device, dtype)
        scaling = 1.0 / math.sqrt(head_dim)

        # Run test
        self.assertCorrectness(
            lambda: gemma_attention_decode(
                q=q,
                k=k,
                v=v,
                scaling=scaling,
                window_size=window_size,
                soft_cap=soft_cap,
            ),
            lambda: einsum_reference_decode(
                q=q,
                k=k,
                v=v,
                scaling=scaling,
                window_size=window_size,
                soft_cap=soft_cap,
            ),
            kwargs={},
            atol=1e-2,
            rtol=1e-2,
            check_stride=False,
        )
