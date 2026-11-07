# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth full LayerNorm (with bias) with autograd."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import layernorm

DEVICE = "cuda"

_backends = ["cutile"]
_perf_backends = _backends + ["pytorch"]


class Test_Unsloth_Layernorm(common.PyTestCase):
    @staticmethod
    def reference(X, W, b, eps=1e-6):
        """PyTorch reference for full LayerNorm with bias."""
        x_f32 = X.float()
        shape = x_f32.shape
        dim = shape[-1]
        x_2d = x_f32.view(-1, dim)
        mean = x_2d.mean(dim=-1, keepdim=True)
        var = ((x_2d - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x_2d - mean) * torch.rsqrt(var + eps)
        output = normed * W.float() + b.float()
        return output.view(*shape).to(X.dtype)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 256),
            (8, 512),
            (16, 1024),
            (2, 4, 512),
            (4, 300),  # non-power-of-2 hidden dim
            (32, 4096),
            (21, 349, 1024),  # upstream unsloth: bsz=21, seqlen=349, dim=1024
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)
        b = torch.randn(dim, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            layernorm,
            self.reference,
            {"X": X, "W": W, "b": b, "eps": 1e-6},
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((4, 256), torch.float16),
            ((8, 512), torch.bfloat16),
            ((4, 256), torch.float32),
            ((2, 4, 512), torch.bfloat16),
            ((4, 2048), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend):
        """Test backward pass via autograd."""
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(42)
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)
        b = torch.randn(dim, dtype=dtype, device=DEVICE)
        dout = torch.ones(*shape, dtype=dtype, device=DEVICE)

        self.assertCorrectness(
            layernorm,
            self.reference,
            {"X": X, "W": W, "b": b, "eps": 1e-6},
            gradient=dout,
            rtol=1e-2,
            atol=1e-2,
        )


# =============================================================================
# Per-kernel perf classes
# =============================================================================


class Test_Unsloth_Layernorm_Forward(common.PyTestCase):
    """Perf for the LayerNorm forward."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 256),
            (32, 4096),
            (512, 4096),
            (1024, 4096),
            (2048, 4096),
        ],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, shape, backend, record_property):
        self.setUp()
        torch.manual_seed(42)
        dtype = torch.bfloat16
        dim = shape[-1]
        X = torch.randn(*shape, dtype=dtype, device=DEVICE)
        W = torch.randn(dim, dtype=dtype, device=DEVICE)
        b = torch.randn(dim, dtype=dtype, device=DEVICE)

        if backend == "pytorch":
            framework_fn = lambda: Test_Unsloth_Layernorm.reference(X, W, b, eps=1e-6)
        else:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")
            framework_fn = lambda: layernorm(X, W, b, eps=1e-6)

        result = common.benchmark_framework(backend, framework_fn)
        record_property("benchmark", result)


class Test_Unsloth_Layernorm_Backward(common.PyTestCase):
    """Perf for the LayerNorm backward."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 256),
            (32, 4096),
            (512, 4096),
            (1024, 4096),
        ],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, shape, backend, record_property):
        """Benchmark backward pass throughput for LayerNorm."""
        self.setUp()
        torch.manual_seed(42)
        dtype = torch.bfloat16
        dim = shape[-1]
        dout = torch.ones(*shape, dtype=dtype, device=DEVICE)

        if backend == "pytorch":
            X = torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)
            W = torch.randn(dim, dtype=dtype, device=DEVICE)
            b = torch.randn(dim, dtype=dtype, device=DEVICE)
            forward_fn = lambda: Test_Unsloth_Layernorm.reference(X, W, b, eps=1e-06)

        else:
            if tilegym.is_backend_available(backend):
                tilegym.set_backend(backend)
            else:
                pytest.skip(f"Backend {backend} is not available")
            X = torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)
            W = torch.randn(dim, dtype=dtype, device=DEVICE)
            b = torch.randn(dim, dtype=dtype, device=DEVICE)
            forward_fn = lambda: layernorm(X, W, b, eps=1e-06)

        result = common.benchmark_framework(backend, forward_fn, mode="backward", gradient=dout, grad_to_none=[X])
        record_property("benchmark", result)
