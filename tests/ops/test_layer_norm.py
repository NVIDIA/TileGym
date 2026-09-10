# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common

_backends = ["cutile"]
_perf_backends = ["pytorch"] + _backends


class Test_LayerNormNCHW(common.PyTestCase):
    @staticmethod
    def reference(x, start_dim, end_dim, weight, bias, eps, weight_shift):
        normalized_shape = [x.shape[i] for i in range(start_dim, end_dim)]
        weight = weight + weight_shift
        y = (
            torch.nn.functional.layer_norm(x.transpose(1, 2), normalized_shape, weight, bias, eps)
            .transpose(1, 2)
            .contiguous()
        )

        return y

    @pytest.mark.parametrize(
        "m, n, k, weight_shift, dtype",
        [
            (2, 512, 256, 0.0, torch.float16),
            (512, 256, 128, 0.0, torch.float32),
            (256, 256, 128, 1.0, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, k, weight_shift, dtype, backend, arch):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        self.setUp()

        eps = 1e-5
        x_shape = (m, n, k)
        w_shape = (n,)
        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
        x = x.detach().requires_grad_(True)

        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

        dy = 0.1 * torch.randn_like(x)
        with torch.no_grad():
            self.assertCorrectness(
                tilegym.ops.layer_norm,
                self.reference,
                {
                    "x": x,
                    "start_dim": 1,
                    "end_dim": 2,
                    "weight": weight,
                    "bias": bias,
                    "eps": eps,
                    "weight_shift": weight_shift,
                },
                gradient=dy,
                rtol=1e-2,
                atol=1e-2,
            )

    @pytest.mark.parametrize(
        "m,n,k,weight_shift,dtype",
        [(1024, 32, 2**i, 0.0, torch.float16) for i in range(10, 15, 1)],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("framework", _perf_backends)
    def test_perf(self, m, n, k, dtype, framework, weight_shift, record_property):
        if framework in _backends:
            if tilegym.is_backend_available(framework):
                tilegym.set_backend(framework)
            else:
                pytest.skip(f"Backend {framework} is not available")
        self.setUp()
        device = torch.device("cuda")
        eps = 1e-5

        x_shape = (m, n, k)
        w_shape = (n,)
        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

        if framework in _backends:
            framework_fn = lambda: tilegym.ops.layer_norm(x, 1, 2, weight, bias, eps, weight_shift)
        elif framework == "pytorch":
            framework_fn = lambda: self.reference(x, 1, 2, weight, bias, eps, weight_shift)
        else:
            pytest.skip(f"Framework {framework} not supported")
        with torch.no_grad():
            result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del x, weight, framework_fn
        if "bias" in locals():
            del bias
        torch.cuda.empty_cache()
        import gc

        gc.collect()


class Test_LayerNorm2D(common.PyTestCase):
    @staticmethod
    def reference(x, normalized_shape, weight, bias, eps, weight_shift):
        weight = weight + weight_shift
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

    @pytest.mark.parametrize(
        "m, n, weight_shift, dtype",
        [
            (256, 256, 0.0, torch.float16),
            (512, 256, 0.0, torch.float32),
            (256, 256, 1.0, torch.float16),
            (9, 9, 0.0, torch.float16),
            (1004, 2456, 1.0, torch.float16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, n, weight_shift, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        self.setUp()

        eps = 1e-5

        x_shape = (m, n)
        w_shape = (n,)

        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
        x = x.detach().requires_grad_(True)

        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

        dy = 0.1 * torch.randn_like(x)
        with torch.no_grad():
            self.assertCorrectness(
                tilegym.ops.layer_norm,
                self.reference,
                {
                    "x": x,
                    "weight": weight,
                    "bias": bias,
                    "eps": eps,
                    "weight_shift": weight_shift,
                },
                extra_test_kwargs={
                    "start_dim": 1,
                    "end_dim": 2,
                },
                extra_ref_kwargs={
                    "normalized_shape": w_shape,
                },
                gradient=dy,
                rtol=0.0,
                atol=1e-2,
            )

    @pytest.mark.parametrize(
        "m,n,weight_shift,dtype",
        [(4096, 2**i, 0.0, torch.float16) for i in range(5, 15, 1)],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("framework", _perf_backends)
    def test_perf(self, m, n, dtype, framework, weight_shift, record_property):
        if framework in _backends:
            if tilegym.is_backend_available(framework):
                tilegym.set_backend(framework)
            else:
                pytest.skip(f"Backend {framework} is not available")
        self.setUp()
        device = torch.device("cuda")
        eps = 1e-5

        x_shape = (m, n)
        w_shape = (n,)
        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

        if framework in _backends:
            framework_fn = lambda: tilegym.ops.layer_norm(x, 1, 2, weight, bias, eps, weight_shift)
        elif framework == "pytorch":
            framework_fn = lambda: self.reference(x, w_shape, weight, bias, eps, weight_shift)
        else:
            pytest.skip(f"Framework {framework} not supported")
        with torch.no_grad():
            result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del x, weight, framework_fn
        if "bias" in locals():
            del bias
        torch.cuda.empty_cache()
        import gc

        gc.collect()
