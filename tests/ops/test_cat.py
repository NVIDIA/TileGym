# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from tilegym.backend import is_backend_available
from tilegym.backend import set_backend

from .. import common


def get_data(
    *shape,
    dtype,
    device,
    mean=0.0,
    normal_std=1.0,
    has_backward=False,
):
    """Generate test data for cat operations."""
    if dtype == torch.float8_e5m2:
        out = torch.empty(*shape, dtype=torch.float16, device=device).normal_(mean, normal_std).to(dtype)
    else:
        out = torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)
    out.requires_grad = has_backward
    return out


def get_cat_tensors(shapes, dtype, device):
    """Generate multiple tensors for concatenation."""
    tensors = []
    for shape in shapes:
        tensor = get_data(*shape, dtype=dtype, device=device)
        tensors.append(tensor)
    return tensors


class Test_CAT(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    @staticmethod
    def reference(tensors, dim=0):
        """Reference implementation using PyTorch's native cat."""
        return torch.cat(tensors, dim=dim)

    @pytest.mark.parametrize(
        "shapes, dtype, dim",
        [
            # Different concat along dim 0
            ([(64,), (32,), (96,)], torch.float16, 0),
            ([(128, 64), (64, 64), (32, 64)], torch.bfloat16, 0),
            ([(256, 128), (128, 128)], torch.float32, 0),
            # Different concat along dim 1
            ([(64, 32), (64, 48), (64, 16)], torch.float16, 1),
            ([(128, 64), (128, 96)], torch.bfloat16, 1),
            ([(256, 128), (256, 256), (256, 64)], torch.float32, 1),
            # 3D tensors
            ([(32, 64, 128), (16, 64, 128)], torch.float16, 0),
            ([(64, 32, 128), (64, 48, 128)], torch.bfloat16, 1),
            ([(64, 128, 32), (64, 128, 96)], torch.float32, 2),
            # Non-power-of-2 sizes
            ([(100, 200), (50, 200), (75, 200)], torch.bfloat16, 0),
            ([(513,), (256,)], torch.float16, 0),
            # Two tensors only
            ([(128, 256), (128, 512)], torch.float32, 1),
            # Interleaved mixed unit/non-unit inner sizes keep original output offsets
            ([(16, 1), (16, 3), (16, 1), (16, 2)], torch.float16, 1),
            # Four tensors (testing batch processing)
            ([(32, 64), (48, 64), (16, 64), (80, 64)], torch.float16, 0),
            # 4D tensors - KV cache: [batch, seq, heads, head_dim], cat on seq dim
            ([(2, 128, 8, 64), (2, 256, 8, 64)], torch.float16, 1),
            ([(1, 512, 16, 128), (1, 512, 16, 128), (1, 256, 16, 128)], torch.bfloat16, 1),
            # 4D tensors - KV cache: [batch, heads, seq, head_dim], cat on seq dim
            ([(2, 8, 128, 64), (2, 8, 64, 64)], torch.float16, 2),
            ([(1, 16, 256, 128), (1, 16, 256, 128)], torch.bfloat16, 2),
            # 4D tensors - cat on dim=0 (batch) and dim=3 (head_dim)
            ([(1, 8, 128, 64), (2, 8, 128, 64)], torch.float32, 0),
            ([(2, 8, 64, 64), (2, 8, 64, 128)], torch.float16, 3),
            # 8D tensors - verify arbitrary-rank support
            ([(2, 3, 4, 5, 6, 7, 8, 16), (2, 3, 4, 5, 6, 7, 8, 32)], torch.float16, 7),
            ([(2, 3, 4, 5, 6, 7, 8, 16), (2, 3, 4, 5, 6, 7, 4, 16)], torch.bfloat16, 6),
            ([(1, 2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 4, 5, 6, 7, 8)], torch.float32, 0),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        shapes,
        dtype,
        dim,
        backend,
        arch,
    ):
        """Test correctness of cat operations."""
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        self.setUp()
        from tilegym.ops import cat as tilegym_cat

        device = "cuda"

        # Generate test tensors
        tensors = get_cat_tensors(shapes, dtype, device)

        # Set appropriate tolerances based on dtype
        if dtype == torch.float8_e5m2:
            atol = 3
            rtol = 0
        elif dtype == torch.float16:
            atol = 1e-3
            rtol = 1e-3
        elif dtype == torch.bfloat16:
            atol = 1e-2
            rtol = 1e-2
        else:
            atol = 1e-5
            rtol = 1e-5

        self.assertCorrectness(
            tilegym_cat,
            self.reference,
            {
                "tensors": tensors,
                "dim": dim,
            },
            atol=atol,
            rtol=rtol,
            check_stride=False,
        )

    @pytest.mark.parametrize(
        "shapes, dtype, dim",
        [
            # Performance test cases - 2D
            ([(1024, 512), (2048, 512), (512, 512)], torch.float16, 0),
            ([(4096, 1024), (2048, 1024)], torch.float16, 0),
            ([(1024, 2048), (1024, 4096)], torch.float16, 1),
            ([(2048, 1024), (2048, 2048), (2048, 512)], torch.float16, 1),
            # Performance test cases - 4D KV cache realistic shapes
            ([(4, 128, 32, 128), (4, 128, 32, 128)], torch.bfloat16, 1),
            ([(4, 32, 128, 128), (4, 32, 128, 128)], torch.bfloat16, 2),
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(
        self,
        shapes,
        dtype,
        dim,
        backend,
        record_property,
    ):
        """Test performance of cat operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")

        self.setUp()
        device = torch.device("cuda")

        # Generate test data
        tensors = get_cat_tensors(shapes, dtype, device)

        if backend in self._backends:
            if not is_backend_available(backend):
                pytest.skip(f"Backend {backend} not available")
            set_backend(backend)
            from tilegym.ops import cat as tilegym_cat

            backend_fn = lambda: tilegym_cat(tensors, dim=dim)
        elif backend == "pytorch":
            backend_fn = lambda: self.reference(tensors, dim=dim)
        else:
            pytest.skip(f"Backend {backend} not supported")

        # Verify correctness before benchmarking
        if backend != "pytorch":
            self.assertCorrectness(
                backend_fn,
                lambda: self.reference(tensors, dim=dim),
                kwargs={},
                atol=1e-2,
                rtol=1e-2,
                check_stride=False,
            )

        # Benchmark performance
        result = common.benchmark_framework(backend, backend_fn)
        record_property("benchmark", result)
