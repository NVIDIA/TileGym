# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch
import torch.nn as nn

import tilegym
from tests import common
from tilegym.suites.liger.ops import tiled_mlp


class SimpleMLP(nn.Module):
    """Simple MLP for testing: linear + relu."""

    def __init__(self, hidden_size, device="cuda", dtype=torch.float32):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        return torch.relu(self.fc(x))


class Test_Liger_TiledMLP(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "bsz, seq_len, hidden_size",
        [
            # Shapes from Liger test_tiled_mlp.py (using hidden_size only)
            (1, 1024, 128),  # num_shards=8 if auto
            (2, 1024, 64),  # num_shards=16 if auto
            (4, 127, 128),  # weird shape
            # Shapes from Liger test/transformers/test_tiled_mlp.py (SwiGLU variant)
            (2, 512, 512),
            (1, 1024, 256),
        ],
    )
    @pytest.mark.parametrize("num_shards", [None, 2, 4])
    @pytest.mark.parametrize("check_2d", [True, False])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, bsz, seq_len, hidden_size, num_shards, check_2d, backend, monkeypatch):
        """Test that tiled computation matches non-tiled computation."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        # Scale input down to reduce numerical sensitivity
        x_data = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=torch.float32) * 0.1

        if check_2d:
            x_data = x_data.view(-1, hidden_size)

        mlp = SimpleMLP(hidden_size, device=device, dtype=torch.float32)
        fn = lambda mod, x: mod(x)

        # Reference: non-tiled computation
        x_ref = x_data.detach().clone().requires_grad_(True)
        out_ref = fn(mlp, x_ref)

        # Tiled computation
        x_tiled = x_data.detach().clone().requires_grad_(True)
        out_tiled = tiled_mlp(fn, mlp, x_tiled, num_shards=num_shards)

        # atol=1e-3: float32 matmul with different chunk sizes may use different
        # cuBLAS algorithms, causing floating-point differences up to ~1e-4.
        assert torch.allclose(out_tiled, out_ref, atol=1e-3, rtol=1e-3), (
            f"Forward mismatch: max_diff={((out_tiled - out_ref).abs().max()).item():.8f}"
        )

    @pytest.mark.parametrize(
        "bsz, seq_len, hidden_size",
        [
            (1, 1024, 128),
            (4, 127, 128),
            (2, 512, 512),
        ],
    )
    @pytest.mark.parametrize("num_shards", [None, 2, 4])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, bsz, seq_len, hidden_size, num_shards, backend, monkeypatch):
        """Test that tiled backward matches non-tiled backward."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        x_data = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=torch.float32) * 0.1

        mlp = SimpleMLP(hidden_size, device=device, dtype=torch.float32)
        fn = lambda mod, x: mod(x)

        # Reference: non-tiled backward
        x_ref = x_data.detach().clone().requires_grad_(True)
        out_ref = fn(mlp, x_ref)
        out_ref.sum().backward()

        # Tiled backward
        x_tiled = x_data.detach().clone().requires_grad_(True)
        out_tiled = tiled_mlp(fn, mlp, x_tiled, num_shards=num_shards)
        out_tiled.sum().backward()

        assert x_tiled.grad is not None, "Tiled backward produced no gradient"
        assert torch.allclose(x_tiled.grad, x_ref.grad, atol=1e-5, rtol=1e-5), (
            f"Backward mismatch: max_diff={((x_tiled.grad - x_ref.grad).abs().max()).item():.8f}"
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 1024, 4096), torch.bfloat16),
            ((2, 2048, 4096), torch.bfloat16),
            ((2, 4096, 4096), torch.bfloat16),
            ((2, 8192, 4096), torch.bfloat16),
            ((2, 16384, 4096), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, shape, dtype, backend, record_property, monkeypatch):
        self.setUp()
        device = torch.device("cuda")
        bsz, seq_len, hidden_size = shape
        x_data = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1

        mlp = SimpleMLP(hidden_size, device=device, dtype=dtype)
        fn = lambda mod, x: mod(x)

        if backend in self._backends:
            if not tilegym.is_backend_available(backend):
                pytest.skip(f"Backend {backend} not available")
            tilegym.set_backend(backend)
            framework_fn = lambda: tiled_mlp(fn, mlp, x_data.detach().clone())
        elif backend == "pytorch":
            framework_fn = lambda: fn(mlp, x_data)
        else:
            pytest.skip(f"Backend {backend} not supported")

        if backend != "pytorch":
            out_test = framework_fn()
            out_ref = fn(mlp, x_data)
            # bf16 chunked GEMM (num_shards > 1) accumulates in a different order than
            # the unsharded reference, differing by ~1 bf16 ULP (~2e-3); ReLU-mask flips
            # at near-zero activations make rtol unusable, so bound on abs only.
            assert torch.allclose(out_test, out_ref, atol=2e-3, rtol=0)

        result = common.benchmark_framework(backend, framework_fn)
        record_property("benchmark", result)

        del x_data, framework_fn
        torch.cuda.empty_cache()
        gc.collect()
