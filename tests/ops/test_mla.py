# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym
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
    _perf_frameworks = _backends + ["pytorch"]

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

    @pytest.mark.parametrize(
        "batch,heads,seq_len,d_model,d_pe,dtype",
        [
            (4, 32, seq_len, 128, 64, dtype)
            for seq_len in [2**9, 2**10, 2**11, 2**12, 2**13]
            for dtype in [torch.bfloat16, torch.float8_e5m2]
        ],
        ids=lambda x: (str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x)),
    )
    @pytest.mark.parametrize("is_causal", [True])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(self, batch, heads, seq_len, d_model, d_pe, dtype, is_causal, framework, record_property):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if framework == "cutile":
            if is_causal == False:
                pytest.skip("Skip non-causal case for cutile")
        if dtype == torch.float8_e5m2 and torch.cuda.get_device_capability()[0] == 8:
            pytest.skip("Skip case due to sm80 not support float8")
        if seq_len == 8192 and torch.cuda.get_device_capability()[0] == 8:
            pytest.skip("Skip case on ampere due to OOM")
        if torch.cuda.get_device_capability() == (12, 0) and seq_len == 8192:
            pytest.skip("Skip OOM on B20X (sm120): MLA with seqlen=8192 requires 16 GiB, exceeds 32 GiB total VRAM")
        self.setUp()
        device = torch.device("cuda")
        # Create random tensors
        q = (
            torch.empty(batch, heads, seq_len, d_model, device=device, dtype=torch.half)
            .normal_(mean=0.0, std=0.3)
            .to(dtype)
        )

        qpe = (
            torch.empty(batch, heads, seq_len, d_pe, device=device, dtype=torch.half)
            .normal_(mean=0.0, std=0.3)
            .to(dtype)
        )

        k = (
            torch.empty(batch, heads, seq_len, d_model, device=device, dtype=torch.half)
            .normal_(mean=0.0, std=0.3)
            .to(dtype)
        )

        kpe = torch.empty(batch, 1, seq_len, d_pe, device=device, dtype=torch.half).normal_(mean=0.0, std=0.3).to(dtype)

        v = (
            torch.empty(batch, heads, seq_len, d_model, device=device, dtype=torch.half)
            .normal_(mean=0.0, std=0.3)
            .to(dtype)
        )

        # Calculate scaling
        scaling = 1.0 / math.sqrt(d_model + d_pe)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(q, k, v, qpe, kpe, is_causal, scaling)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: mla_interface(q, k, v, qpe, kpe, is_causal, scaling)
        else:
            pytest.skip(f"Framework {framework} is not available")

        # pytorch reference uses dynamic tensor creation (torch.triu_indices) which is
        # incompatible with CUDA graph capture — disabling cudagraph for that framework
        # to prevent capture_epilogue() being skipped on failure, which would leave the
        # default CUDA generator in capturing_=True state and corrupt subsequent tests.
        use_cudagraph = framework != "pytorch"
        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=use_cudagraph)
        record_property("benchmark", result)

        if dtype == torch.bfloat16:
            tols = {"rtol": 1e-2, "atol": 1e-2}
        else:
            tols = {"rtol": 1e-1, "atol": 1e-1}

        torch.cuda.empty_cache()
        # check after benchmark
        self.assertCorrectness(
            framework_fn,
            lambda: self.reference(q, k, v, qpe, kpe, is_causal, scaling),
            kwargs={},
            **tols,
        )

        # Explicit cleanup to prevent OOM
        del q, k, v, qpe, kpe, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
