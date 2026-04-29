# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common


class Test_Dropout(common.PyTestCase):
    @staticmethod
    def reference(x, p, training=True, inplace=False):
        return torch.nn.functional.dropout(x, p, training, inplace)

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "m, p, training, inplace, dtype, eps",
        [
            (m, p, training, inplace, dtype, eps)
            for (m, p, dtype) in [
                (2**10, 0.2, torch.float32),
                (2**12, 0.4, torch.float32),
                (2**16, 0.8, torch.float32),
                (2**16, 0.8, torch.float16),
            ]
            for training in [True, False]
            for inplace in [True, False]
            for eps in [
                1e-05,
            ]
        ],
        ids=lambda x: (str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x)),
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, m, p, training, inplace, dtype, eps, backend, arch):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        if arch in ["sm120", "sm121"] and "-".join(
            map(
                str,
                (
                    m,
                    p,
                    training,
                    inplace,
                    dtype,
                    eps,
                ),
            )
        ) in [
            "4096-0.4-False-False-torch.float32-1e_05",
            "4096-0.4-False-True-torch.float32-1e_05",
            "65536-0.8-False-False-torch.float16-1e_05",
            "65536-0.8-False-False-torch.float32-1e_05",
            "65536-0.8-False-True-torch.float16-1e_05",
            "65536-0.8-False-True-torch.float32-1e_05",
            "65536-0.8-True-False-torch.float16-1e_05",
            "65536-0.8-True-False-torch.float32-1e_05",
            "65536-0.8-True-True-torch.float16-1e_05",
            "65536-0.8-True-True-torch.float32-1e_05",
        ]:
            pytest.skip("Skip due to global memory OOM")

        seed = torch.random.initial_seed()
        self.setUp()

        device = torch.device("cuda")

        requires_grad = False
        x = torch.rand(m, device=device, dtype=dtype, requires_grad=requires_grad) + eps
        x_clone = x.clone()

        res = tilegym.ops.dropout(x, seed, p, training, inplace)

        if (not training) or inplace:
            assert id(res) == id(x)

        if training:
            zero_ratio = 1 - torch.count_nonzero(res) / torch.numel(res)
            threshold = 0.04
            assert p - threshold < zero_ratio < p + threshold, zero_ratio
        else:
            self.assertAllClose(x, x_clone, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "m, p, inplace, dtype",
        [
            (m, p, inplace, dtype)
            for m in [2**i for i in range(20, 28, 2)]
            for p in [0.5]
            for inplace in [True, False]
            for dtype in [torch.float32, torch.float16]
        ],
        ids=lambda x: (str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x)),
    )
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(self, m, p, inplace, dtype, framework, record_property):
        self.setUp()
        seed = torch.random.initial_seed()
        device = torch.device("cuda")

        requires_grad = False

        x = torch.rand(m, device=device, dtype=dtype, requires_grad=requires_grad)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(x, p, True, inplace)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: tilegym.ops.dropout(x, seed, p, True, inplace)
        else:
            pytest.skip(f"Framework {framework} is not available")

        res = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del x, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
