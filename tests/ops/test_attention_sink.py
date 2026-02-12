# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os

import pytest
import torch

import tilegym
import tilegym.ops
from tilegym.backend import get_current_backend
from tilegym.backend import register_impl
from tilegym.backend import set_backend

from .. import common

_backends = ["cutile"]


def get_data(
    *shape,
    dtype,
    device,
    mean=0.0,
    normal_std=1.0,
):
    """Generate random tensor data for testing."""
    out = torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)
    return out


class Test_AttentionSink(common.PyTestCase):
    @staticmethod
    def reference(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sinks: torch.Tensor,
        sm_scale: float = 0.125,
        sliding_window: int | None = None,
        start_q: torch.LongTensor = 0,
    ):
        """Reference implementation for attention with sinks using PyTorch."""
        batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
        batch_size, num_keys, num_key_value_heads, head_dim = key.shape

        sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
        key = key.unsqueeze(3)
        value = value.unsqueeze(3)

        pos_keys = torch.arange(num_keys, device=query.device)
        pos_queries = torch.arange(num_queries, device=query.device) + start_q
        mask = pos_keys[None, :] > pos_queries[:, None]
        mask = mask.float().masked_fill(mask, float("-inf"))

        if sliding_window:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
        logits = logits + mask[None, None, None, :, :]

        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(sinks, logits_max)
        sinks_exp = torch.exp(sinks - logits_or_sinks_max)
        unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnormalized_scores / normalizer

        output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

        output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).to(
            query.dtype
        )
        return output

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_queries", [1, 128])
    @pytest.mark.parametrize("num_keys", [128, 32])
    @pytest.mark.parametrize("num_key_value_heads", [8])
    @pytest.mark.parametrize("num_key_value_groups", [8])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("sm_scale", [0.125])
    @pytest.mark.parametrize("sliding_window", [None, 128])
    @pytest.mark.parametrize("start_q", [0, 5])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch_size,
        num_queries,
        num_keys,
        num_key_value_heads,
        num_key_value_groups,
        head_dim,
        sm_scale,
        sliding_window,
        start_q,
        backend: str,
    ):
        """Test correctness of attention_sink implementation against reference."""
        if num_queries > num_keys:
            pytest.skip("Number of queries cannot exceed number of keys")

        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend {backend} is not supported: {e}")

        self.setUp()

        # Create random input tensors
        q = get_data(
            batch_size,
            num_queries,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = get_data(batch_size, num_keys, num_key_value_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v = get_data(batch_size, num_keys, num_key_value_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        sinks = get_data(num_key_value_heads * num_key_value_groups, device="cuda", dtype=torch.bfloat16)

        start_q_tensor = torch.tensor([start_q], dtype=torch.int32).cuda()

        # Test implementation
        test_fn = lambda: tilegym.ops.attention_sink(q, k, v, sinks, sm_scale, sliding_window, start_q_tensor)
        # Reference implementation
        ref_fn = lambda: self.reference(q, k, v, sinks, sm_scale, sliding_window, start_q_tensor)

        self.assertCorrectness(
            test_fn,
            ref_fn,
            kwargs={},
            atol=5e-2,
            rtol=1e-2,
            check_stride=False,
        )
