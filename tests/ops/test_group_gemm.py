# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import set_backend

from .. import common


class Test_GroupGemm(common.PyTestCase):
    @staticmethod
    def reference(group_A, group_B, transpose_b=False):
        dtype = group_A[0].dtype
        return [
            torch.matmul(
                a.to(torch.half),
                b.to(torch.half).t() if transpose_b else b.to(torch.half),
            ).to(dtype)
            for a, b in zip(group_A, group_B)
        ]

    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "group_m, group_n, group_k, transpose_b, dtype",
        [
            (group_m, group_n, group_k, transpose_b, dtype)
            for group_m in [
                [1024, 512, 256, 128],
                [256, 256, 256, 256],
            ]
            for group_n in [
                [1024, 512, 256, 128],
                [128, 128, 128, 128],
            ]
            for group_k in [
                [1024, 512, 256, 128],
                [128, 128, 128, 128],
            ]
            for transpose_b in [True, False]
            for dtype in [
                torch.float16,
            ]
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        group_m,
        group_n,
        group_k,
        transpose_b,
        dtype,
        backend,
    ):
        if not is_backend_available(backend):
            pytest.skip("Cutile backend not available")

        device = torch.device("cuda")
        self.setUp()
        set_backend(backend)

        group_A = []
        group_B = []
        assert len(group_m) == len(group_n)
        assert len(group_n) == len(group_k)
        num_groups = len(group_m)
        for i in range(num_groups):
            M = group_m[i]
            N = group_n[i]
            K = group_k[i]
            A = torch.rand((M, K), device=device, dtype=torch.half).to(dtype)
            B = torch.rand(
                (N, K) if transpose_b else (K, N),
                device=device,
                dtype=torch.half,
            ).to(dtype)
            group_A.append(A)
            group_B.append(B)

        self.assertCorrectness(
            tilegym.ops.group_gemm,
            self.reference,
            {
                "group_A": group_A,
                "group_B": group_B,
                "transpose_b": transpose_b,
            },
            rtol=1e-3,
            atol=1e-8,
            multiple_outputs=True,
        )

    @pytest.mark.parametrize(
        "num_groups, group_m, group_n, group_k, transpose_b, dtype",
        [
            (num_groups, group_m, group_n, group_k, transpose_b, dtype)
            for num_groups in [1, 4, 16]
            for group_m in [2048, 8192]
            for group_n in [2048, 8192]
            for group_k in [2048, 8192]
            for transpose_b in [True, False]
            for dtype in [torch.float16, torch.float8_e5m2]
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(
        self,
        num_groups,
        group_m,
        group_n,
        group_k,
        transpose_b,
        dtype,
        backend,
        record_property,
    ):
        self.setUp()
        device = torch.device("cuda")
        group_A = []
        group_B = []

        for i in range(num_groups):
            A = torch.rand((group_m, group_k), device=device, dtype=torch.half).normal_(std=0.3).to(dtype)
            B = (
                torch.rand(
                    (group_n, group_k) if transpose_b else (group_k, group_n),
                    device=device,
                    dtype=torch.half,
                )
                .normal_(std=0.3)
                .to(dtype)
            )

            group_A.append(A)
            group_B.append(B)

        if backend == "pytorch":
            backend_fn = lambda: self.reference(group_A, group_B, transpose_b=transpose_b)
        elif is_backend_available(backend):
            if backend == "cutile" and dtype == torch.float8_e5m2:
                pytest.skip("Skip float8_e5m2 due to cutile not support float8")
            set_backend(backend)
            backend_fn = lambda: tilegym.ops.group_gemm(
                group_A,
                group_B,
                transpose_b=transpose_b,
            )
        else:
            pytest.skip(f"Backend {backend} is not available")

        if record_property is None:
            res = backend_fn()
            return

        res = common.benchmark_framework(backend, backend_fn, use_cudagraph=False)
        record_property("benchmark", res)
        if dtype == torch.float8_e5m2:
            atol = 1
            rtol = 1
        else:
            atol = 1e-2
            rtol = 1e-2
        # run after benchmark
        skip_correctness = backend == "pytorch"
        if not skip_correctness:
            self.assertCorrectness(
                backend_fn,
                lambda: self.reference(group_A, group_B, transpose_b=transpose_b),
                kwargs={},
                rtol=rtol,
                atol=atol,
                multiple_outputs=True,
            )

        # Explicit cleanup to prevent OOM
        del group_A, group_B, backend_fn
        if "kernel_configs" in locals():
            del kernel_configs
        torch.cuda.empty_cache()
        import gc

        gc.collect()
