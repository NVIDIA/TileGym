# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


import pytest
import torch

import tilegym
import tilegym.ops
from tilegym import set_backend

from ... import common

MAX_TEST_DIM = 5


class Test_GEGLU(common.PyTestCase):
    @staticmethod
    def reference(input, dim, approximate):
        dim_size = input.shape[dim]
        a, b = torch.split(input, dim_size // 2, dim)
        geglu = a * torch.nn.functional.gelu(b, approximate=approximate)
        return geglu

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "x_shape,dim,dtype,approximate",
        [
            # max_dim=1, test_dim=0
            ((4,), 0, torch.float32, "none"),
            ((4,), 0, torch.float32, "tanh"),
            ((4,), 0, torch.float16, "none"),
            ((4,), 0, torch.float16, "tanh"),
            # max_dim=2, test_dim=0,1
            ((16, 4), 0, torch.float32, "none"),
            ((16, 4), 0, torch.float32, "tanh"),
            ((16, 4), 0, torch.float16, "none"),
            ((16, 4), 0, torch.float16, "tanh"),
            ((16, 4), 1, torch.float32, "none"),
            ((16, 4), 1, torch.float32, "tanh"),
            ((16, 4), 1, torch.float16, "none"),
            ((16, 4), 1, torch.float16, "tanh"),
            # max_dim=3, test_dim=0,1,2
            ((64, 16, 4), 0, torch.float32, "none"),
            ((64, 16, 4), 0, torch.float32, "tanh"),
            ((64, 16, 4), 0, torch.float16, "none"),
            ((64, 16, 4), 0, torch.float16, "tanh"),
            ((64, 16, 4), 1, torch.float32, "none"),
            ((64, 16, 4), 1, torch.float32, "tanh"),
            ((64, 16, 4), 1, torch.float16, "none"),
            ((64, 16, 4), 1, torch.float16, "tanh"),
            ((64, 16, 4), 2, torch.float32, "none"),
            ((64, 16, 4), 2, torch.float32, "tanh"),
            ((64, 16, 4), 2, torch.float16, "none"),
            ((64, 16, 4), 2, torch.float16, "tanh"),
            # max_dim=4, test_dim=0,1,2,3
            ((256, 64, 16, 4), 0, torch.float32, "none"),
            ((256, 64, 16, 4), 0, torch.float32, "tanh"),
            ((256, 64, 16, 4), 0, torch.float16, "none"),
            ((256, 64, 16, 4), 0, torch.float16, "tanh"),
            ((256, 64, 16, 4), 1, torch.float32, "none"),
            ((256, 64, 16, 4), 1, torch.float32, "tanh"),
            ((256, 64, 16, 4), 1, torch.float16, "none"),
            ((256, 64, 16, 4), 1, torch.float16, "tanh"),
            ((256, 64, 16, 4), 2, torch.float32, "none"),
            ((256, 64, 16, 4), 2, torch.float32, "tanh"),
            ((256, 64, 16, 4), 2, torch.float16, "none"),
            ((256, 64, 16, 4), 2, torch.float16, "tanh"),
            ((256, 64, 16, 4), 3, torch.float32, "none"),
            ((256, 64, 16, 4), 3, torch.float32, "tanh"),
            ((256, 64, 16, 4), 3, torch.float16, "none"),
            ((256, 64, 16, 4), 3, torch.float16, "tanh"),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, x_shape, dim, dtype, approximate, backend):
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        self.setUp()
        device = torch.device("cuda")
        y_shape = list(x_shape)
        y_shape[dim] = y_shape[dim] // 2

        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(1.2).add_(0.6)
        x = x.detach().requires_grad_(True)

        dy = 0.1 * torch.randn(*y_shape, device=device)

        self.assertCorrectness(
            tilegym.ops.activation.geglu,
            self.reference,
            {"input": x, "dim": dim, "approximate": approximate},
            gradient=dy,
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "x_shape,dim,dtype,approximate",
        [
            ((256, 64, 16, 128), 3, torch.float16, "none"),
            ((256, 64, 16, 128), 3, torch.float16, "tanh"),
        ],
        ids=lambda x: str(x),
    )
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(self, x_shape, dim, dtype, approximate, framework, record_property):
        self.setUp()
        device = torch.device("cuda")
        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=True)
        # Calculate output shape for gradient
        y_shape = list(x_shape)
        y_shape[dim] = y_shape[dim] // 2
        dy = 0.1 * torch.randn(*y_shape, device=device)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(x, dim, approximate)
        elif tilegym.is_backend_available(framework):
            set_backend(framework)
            framework_fn = lambda: tilegym.ops.activation.geglu(x, dim, approximate)
        else:
            pytest.skip(f"Framework {framework} is not available")

        if framework != "pytorch":
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(x, dim, approximate),
                kwargs={},
                gradient=dy,
                rtol=1e-2,
                atol=1e-2,
            )

        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=False)
        record_property("benchmark", result)
