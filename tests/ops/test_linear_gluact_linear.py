# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common


class Test_LinearGluactLinear(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    def _prepare_data(
        self,
        m,
        n,
        k,
        dtype,
    ):
        """Create test tensors for the linear_gluact_linear tests."""
        self.setUp()
        device = torch.device("cuda")
        a = torch.rand(m, k, device=device, dtype=dtype, requires_grad=True)
        c = torch.rand(k, n, device=device, dtype=dtype, requires_grad=True)
        # Separate weights for the activation and non-activation branches
        b_act = torch.rand(n, k, device=device, dtype=dtype, requires_grad=True)
        b_noact = torch.rand(n, k, device=device, dtype=dtype, requires_grad=True)
        return a, b_act, b_noact, c, None, None

    @staticmethod
    def reference(
        input,
        weight_act,
        weight_noact,
        weight2,
        bias_act,
        bias_noact,
        act_type,
        dropout_prob=0.0,
    ):
        """Reference implementation (separate weights)"""
        noact_out = torch.nn.functional.linear(input, weight_noact, bias=bias_noact)
        act_in = torch.nn.functional.linear(input, weight_act, bias=bias_act)

        if act_type == "relu":
            act_out = torch.nn.functional.relu(act_in)
        elif act_type == "gelu":
            act_out = torch.nn.functional.gelu(act_in, approximate="none")
        elif act_type == "gelu-tanh":
            act_out = torch.nn.functional.gelu(act_in, approximate="tanh")
        elif act_type == "silu":
            act_out = torch.nn.functional.silu(act_in)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

        act_out = act_out * noact_out
        if dropout_prob > 0.0:
            act_out = torch.nn.functional.dropout(act_out, p=dropout_prob)
        return torch.nn.functional.linear(act_out, weight2)

    @pytest.mark.parametrize(
        "m, n, k, act_type, dtype",
        [
            (1024, 1024, 512, "gelu-tanh", torch.float32),
            (1024, 1024, 1023, "silu", torch.float32),
            (1024, 1024, 1024, "silu", torch.float32),
            (1024, 1024, 1024, "relu", torch.float32),
            (512, 512, 256, "silu", torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, k, act_type, dtype, backend, arch):
        """Test dispatched linear_gluact_linear implementation with different backends"""
        if k == 1023:
            pytest.skip("Skip 1023 because strides must be 16-byte aligned when creating tensor descriptors")

        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")

        # Set backend
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        a, b_act, b_noact, c, _, _ = self._prepare_data(m, n, k, dtype)

        # Dispatched implementation doesn't support bias and dropout yet
        bias_act = None
        bias_noact = None

        def dispatch_adapter(
            input,
            weight_act,
            weight_noact,
            weight2,
            bias_act,
            bias_noact,
            act_type,
            dropout_prob,
        ):
            # Implementation doesn't support bias and dropout yet
            if bias_act is not None or bias_noact is not None:
                pytest.skip("Implementation doesn't support bias yet")
            if dropout_prob > 0.0:
                pytest.skip("Implementation doesn't support dropout yet")
            # Use the dispatch interface
            return tilegym.ops.linear_gluact_linear(input, weight_act, weight_noact, weight2, act_type)

        self.assertCorrectness(
            dispatch_adapter,
            self.reference,
            {
                "input": a,
                "weight_act": b_act,
                "weight_noact": b_noact,
                "weight2": c,
                "bias_act": bias_act,
                "bias_noact": bias_noact,
                "act_type": act_type,
                "dropout_prob": 0.0,
            },
            rtol=5e-3 if dtype == torch.float16 else 3e-3,
            atol=1e-3 if dtype == torch.float16 else 1e-5,
        )

    @pytest.mark.parametrize(
        "m, n, k, act_type, dtype",
        [
            (1024, 1024, 1024, "silu", torch.float16),
            (2048, 2048, 2048, "silu", torch.float16),
            (4096, 4096, 4096, "silu", torch.float16),
        ],
        ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, m, n, k, act_type, dtype, backend, arch, record_property):
        """Performance test with backend comparison"""
        self.setUp()
        device = torch.device("cuda")

        # Create tensors with proper scaling to avoid fp16 overflow
        # Use randn (normal distribution) with small scale like real neural networks
        scale = 0.02  # Similar to typical weight initialization
        a = torch.randn(m, k, device=device, dtype=dtype) * scale
        c = torch.randn(k, n, device=device, dtype=dtype) * scale
        b_act = torch.randn(n, k, device=device, dtype=dtype) * scale
        b_noact = torch.randn(n, k, device=device, dtype=dtype) * scale

        if backend != "pytorch":
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")

        with torch.no_grad():
            if backend == "pytorch":
                backend_fn = lambda: self.reference(a, b_act, b_noact, c, None, None, act_type)
            else:
                kernel_kwargs = {
                    "act_type": act_type,
                }

                backend_fn = lambda: tilegym.ops.linear_gluact_linear(a, b_act, b_noact, c, **kernel_kwargs)

            if backend != "pytorch":
                self.assertCorrectness(
                    backend_fn,
                    lambda: self.reference(a, b_act, b_noact, c, None, None, act_type),
                    kwargs={},
                    rtol=5e-3 if dtype == torch.float16 else 3e-3,
                    atol=1e-3 if dtype == torch.float16 else 1e-5,
                )

            res = common.benchmark_framework(backend, backend_fn, use_cudagraph=True)
            record_property("benchmark", res)

            # Explicit cleanup to prevent OOM
            del a, b_act, b_noact, c, backend_fn
            torch.cuda.empty_cache()
            import gc

            gc.collect()
