# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

from tilegym.backend import set_backend
from tilegym.ops import mla_interface

from .. import common


class Test_MLA(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, qpe, kpe, is_causal, scaling=None):
        qkv_dtype = v.dtype
        q = q.half()
        qpe = qpe.half()
        k = k.half()
        kpe = kpe.half()
        v = v.half()
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

        # Get dimensions
        batch_size, num_head_q, seq_len, head_dim = q.shape
        _, num_head_kv, _, _ = k.shape

        # Handle multi-query attention (when num_head_q != num_head_kv)
        if num_head_q != num_head_kv:
            # Make sure num_head_q is divisible by num_head_kv
            assert num_head_q % num_head_kv == 0, "Query heads must be divisible by KV heads"

            # Calculate how many query heads are served by each kv head
            query_group_size = num_head_q // num_head_kv

            # Expand k and v to match the query head dimension
            # Shape: [batch, num_head_kv, seq_len, head_dim] -> [batch, num_head_q, seq_len, head_dim]
            k_expanded = k.unsqueeze(2).expand(batch_size, num_head_kv, query_group_size, seq_len, head_dim)
            k_expanded = k_expanded.reshape(batch_size, num_head_q, seq_len, head_dim)

            v_expanded = v.unsqueeze(2).expand(batch_size, num_head_kv, query_group_size, seq_len, head_dim)
            v_expanded = v_expanded.reshape(batch_size, num_head_q, seq_len, head_dim)

            # Use expanded tensors
            k = k_expanded
            v = v_expanded

        # Calculate attention scores
        qk = torch.matmul(q, k.transpose(2, 3))
        if qpe is not None and kpe is not None:
            # Handle kpe for multi-query attention if needed
            if kpe.shape[1] == 1 and num_head_q > 1:
                kpe = kpe.expand(-1, num_head_q, -1, -1)
            qk = qk + torch.matmul(qpe, kpe.transpose(2, 3))
        qk = qk.float()
        qk *= scaling

        # Apply causal mask if needed
        if is_causal:
            if q.size(-2) > 1:
                rows, cols = torch.triu_indices(qk.shape[-2], qk.shape[-1], offset=1, device=qk.device)
                qk[..., rows, cols] = float("-inf")

        # Calculate attention weights
        m = torch.max(qk, dim=-1)[0]
        qk -= m.unsqueeze(-1)
        p = qk.exp_()
        l = torch.sum(p, dim=-1)
        p /= l.unsqueeze(-1)
        p = p.to(qkv_dtype)
        # Calculate output
        o = torch.matmul(p.half(), v).to(qkv_dtype)
        return o

    _backends = ["cutile"]

    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize(
        "BLOCK_M", [64] if torch.cuda.get_device_capability() in [(12, 0), (12, 1)] else [128, 256]
    )
    @pytest.mark.parametrize("BLOCK_N", [64] if torch.cuda.get_device_capability() in [(12, 0), (12, 1)] else [128])
    @pytest.mark.parametrize("num_group_size", [1, 4])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        is_causal,
        num_group_size,
        dtype,
        BLOCK_M,
        BLOCK_N,
        backend,
        arch,
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")

        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: timeout.")

        if backend == "cutile" and is_causal == False:
            pytest.skip("Skip non-causal due to cutile not support")

        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        self.setUp()

        # Create test data
        num_batch = 1
        num_head_q = 16  # Query heads
        num_head_kv = num_head_q // num_group_size  # Key/Value heads - should divide num_head_q evenly
        S_qkv = 9
        BLOCK_D = 128
        BLOCK_KPE = 64

        device = torch.device("cuda")

        # Create random tensors with appropriate head dimensions
        q = torch.empty(num_batch, num_head_q, S_qkv, BLOCK_D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

        qpe = torch.empty(num_batch, num_head_q, S_qkv, BLOCK_KPE, device=device, dtype=dtype).normal_(
            mean=0.0, std=0.3
        )

        # Key and value tensors use num_head_kv
        k = torch.empty(num_batch, num_head_kv, S_qkv, BLOCK_D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

        kpe = torch.empty(num_batch, 1, S_qkv, BLOCK_KPE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

        v = torch.empty(num_batch, num_head_kv, S_qkv, BLOCK_D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

        # Calculate scaling
        scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

        # Configure kernel parameters
        if backend == "cutile":
            kernel_configs = {
                "TILE_M": BLOCK_M,
                "TILE_N": BLOCK_N,
            }
        else:
            kernel_configs = {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
            }

        # Define a wrapper to match the interface expected by assertCorrectness
        def mla_wrapper(q, k, v, qpe, kpe, is_causal, scaling, kernel_configs):
            return mla_interface(q, k, v, qpe, kpe, is_causal, scaling, kernel_configs=kernel_configs)

        # Use assertCorrectness to compare the implementations
        self.assertCorrectness(
            mla_wrapper,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "qpe": qpe,
                "kpe": kpe,
                "is_causal": is_causal,
                "scaling": scaling,
            },
            extra_test_kwargs={
                "kernel_configs": kernel_configs,
            },
            rtol=1e-2,
            atol=1e-2,
        )
