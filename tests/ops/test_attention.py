# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc
import math

import pytest
import torch

import tilegym
from tilegym.backend import set_backend
from tilegym.ops import fmha_interface

from .. import common


def get_data(
    *shape,
    dtype,
    device,
    mean=0.0,
    normal_std=1.0,
):
    if dtype == torch.float8_e5m2:
        out = torch.empty(*shape, dtype=torch.float16, device=device).normal_(mean, normal_std).to(dtype)
    else:
        out = torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)
    return out


class Test_FMHA(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, scaling=None, attention_mask=None, is_causal=False):
        if q.dtype == torch.float8_e5m2:
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=scaling,
            )
            return ref.to(q.dtype)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal, scale=scaling
        )
        return ref

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "batch_size, num_heads, seq_len, head_dim, is_causal, dtype",
        [
            (1, 1, 9, 128, False, torch.bfloat16),
            (1, 32, 2047, 128, True, torch.float16),
            (2, 32, 4095, 128, True, torch.bfloat16),
            (2, 32, 4095, 128, True, torch.float8_e5m2),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        is_causal,
        dtype,
        backend,
        arch,
    ):
        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")
        if arch in ["sm80"] and dtype == torch.float8_e5m2:
            pytest.skip("Skip on sm80: float8_e5m2 is not supported")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        # Create random input tensors
        q = get_data(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        k = get_data(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        v = get_data(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        if dtype == torch.float8_e5m2:
            atol = 3
            rtol = 0
        else:
            atol = 5e-2
            rtol = 1e-2
        self.assertCorrectness(
            fmha_interface,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "scaling": sm_scale,
                "is_causal": is_causal,
            },
            atol=atol,
            rtol=rtol,
            check_stride=False,
        )

    @pytest.mark.parametrize(
        "batch,heads,seq_len,head_dim,dtype",
        [
            (4, 32, seq_len, head_dim, dtype)
            for dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2]
            for seq_len in (
                [
                    2**9,
                    2**10,
                    2**11,
                    2**12,
                    2**13,
                ]  # can be divided by BLOCK
                + [
                    2**10 + 1,
                    2**11 + 1,
                    2**12 + 1,
                ]  # can not be divided by BLOCK
            )
            for head_dim in ([64] if torch.cuda.get_device_capability()[0] == 8 else [128])
        ],
        ids=lambda x: (str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x)),
    )
    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(self, batch, heads, seq_len, head_dim, dtype, is_causal, framework, record_property):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if torch.cuda.get_device_capability()[0] == 8:
            if dtype == torch.float8_e5m2 or dtype == torch.bfloat16:
                pytest.skip(f"Skip {dtype} type now")

        self.setUp()
        device = torch.device("cuda")
        has_backward = False
        mean, normal_std = 0.0, 1.0
        q = get_data(
            batch,
            heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            mean=mean,
            normal_std=normal_std,
        )
        k = get_data(
            batch,
            heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            mean=mean,
            normal_std=normal_std,
        )
        v = get_data(
            batch,
            heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            mean=mean,
            normal_std=normal_std,
        )

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(q, k, v, scaling=sm_scale, is_causal=is_causal)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: fmha_interface(
                q, k, v, scaling=sm_scale, is_causal=is_causal, has_backward=has_backward
            )
        else:
            pytest.skip(f"Framework {framework} is not available")
        if framework != "pytorch":
            if dtype == torch.float8_e5m2:
                atol = 3
                rtol = 0
            else:
                atol = 5e-2
                rtol = 2e-2
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(q, k, v, scaling=sm_scale, is_causal=is_causal),
                kwargs={},
                atol=atol,
                rtol=rtol,
                check_stride=False,
            )
        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del q, k, v, framework_fn
        torch.cuda.empty_cache()
        gc.collect()

    @pytest.mark.parametrize(
        "model, batch_size, num_heads, seq_len, head_dim",
        [
            (
                "llama",
                1,
                32,
                9,
                128,
            ),  # BLOCK_M: 64, BLOCK_N: 128, num_warps: 4, num_ctas: 1, num_stages: 2,
            (
                "llama",
                1,
                32,
                31072,
                128,
            ),  # BLOCK_M: 128, BLOCK_N: 128, num_warps: 8, num_ctas: 1, num_stages: 2
        ],
        ids=lambda x: (str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x)),
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e5m2])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf_llm(self, model, batch_size, num_heads, seq_len, head_dim, dtype, framework, record_property):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if torch.cuda.get_device_capability()[0] == 8:
            if dtype == torch.float8_e5m2:
                pytest.skip("Skip due to sm80 not support float8 type")
        self.setUp()
        device = torch.device("cuda")
        q = get_data(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = get_data(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = get_data(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        if framework == "pytorch":
            framework_fn = lambda: self.reference(q, k, v, scaling=sm_scale, is_causal=True)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: fmha_interface(q, k, v, scaling=sm_scale, is_causal=True, has_backward=False)
        else:
            pytest.skip(f"Framework {framework} is not available")
        if framework != "pytorch":
            if dtype == torch.float8_e5m2:
                atol = 3
                rtol = 0
            else:
                atol = 1e-2
                rtol = 1e-2
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(q, k, v, scaling=sm_scale, is_causal=True),
                kwargs={},
                rtol=rtol,
                atol=atol,
                check_stride=False,
            )
        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del q, k, v, framework_fn
        torch.cuda.empty_cache()
        gc.collect()
