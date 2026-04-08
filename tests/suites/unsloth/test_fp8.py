# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth FP8 quantization kernels (weight_dequant, act_quant, w8a8 matmul)."""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import act_quant
from tilegym.suites.unsloth.ops import w8a8_block_fp8_matmul
from tilegym.suites.unsloth.ops import weight_dequant

DEVICE = "cuda"

_backends = ["cutile"]


class Test_Unsloth_WeightDequant(common.PyTestCase):
    @staticmethod
    def reference(x, s, block_size=128, dtype=torch.bfloat16):
        """PyTorch reference for block-wise FP8 weight dequantization."""
        M, N = x.shape
        x_f32 = x.float()
        y = torch.empty(M, N, dtype=dtype, device=x.device)
        bm = (M + block_size - 1) // block_size
        bn = (N + block_size - 1) // block_size
        for i in range(bm):
            for j in range(bn):
                row_start = i * block_size
                row_end = min(row_start + block_size, M)
                col_start = j * block_size
                col_end = min(col_start + block_size, N)
                y[row_start:row_end, col_start:col_end] = (x_f32[row_start:row_end, col_start:col_end] * s[i, j]).to(
                    dtype
                )
        return y

    @pytest.mark.parametrize(
        "M, N, block_size",
        [
            (256, 256, 128),
            (512, 1024, 128),
            (384, 768, 128),
            # Larger shapes
            (2048, 4096, 128),
            (4096, 8192, 128),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, M, N, block_size, framework, arch):
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        if arch == "sm80":
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        torch.manual_seed(42)
        x_f32 = torch.randn(M, N, device=DEVICE)
        x_fp8 = x_f32.to(torch.float8_e4m3fn)
        bm = (M + block_size - 1) // block_size
        bn = (N + block_size - 1) // block_size
        s = torch.rand(bm, bn, dtype=torch.float32, device=DEVICE) + 0.1

        self.assertCorrectness(
            weight_dequant,
            self.reference,
            {"x": x_fp8, "s": s, "block_size": block_size},
            rtol=1e-3,
            atol=1e-3,
            check_stride=False,
        )


class Test_Unsloth_ActQuant(common.PyTestCase):
    @staticmethod
    def reference(x, block_size=128):
        """PyTorch reference for block-wise activation quantization."""
        x_flat = x.contiguous().view(-1)
        n_blocks = x_flat.numel() // block_size
        x_blocks = x_flat.view(n_blocks, block_size)
        x_f32 = x_blocks.float()

        s = x_f32.abs().max(dim=-1, keepdim=False).values / 448.0
        # Handle all-zero rows
        s = torch.where(s == 0, torch.ones_like(s), s)

        y_f32 = x_f32 / s.unsqueeze(-1)
        y = y_f32.to(torch.float8_e4m3fn)

        y = y.view(x.shape)
        s = s.view(*x.shape[:-1], x.shape[-1] // block_size)
        return y, s

    @pytest.mark.parametrize(
        "shape, block_size",
        [
            ((256, 256), 128),
            ((64, 1024), 128),
            ((128, 512), 128),
            # Larger shapes
            ((1024, 4096), 128),
            ((2048, 8192), 128),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, shape, block_size, framework, arch):
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        if arch == "sm80":
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        torch.manual_seed(42)
        x = torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE)

        y_result, s_result = act_quant(x, block_size=block_size)
        y_expected, s_expected = self.reference(x, block_size=block_size)

        torch.testing.assert_close(s_result, s_expected, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(y_result.float(), y_expected.float(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "shape, block_size",
        [
            ((256, 128), 128),
            ((128, 384), 128),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op_non_square(self, shape, block_size, framework, arch):
        """Test act_quant with non-square shapes for robustness."""
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        if arch == "sm80":
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        torch.manual_seed(42)
        x = torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE)

        y_result, s_result = act_quant(x, block_size=block_size)
        y_expected, s_expected = self.reference(x, block_size=block_size)

        torch.testing.assert_close(s_result, s_expected, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(y_result.float(), y_expected.float(), rtol=1e-2, atol=1e-2)


class Test_Unsloth_W8A8_FP8_Matmul(common.PyTestCase):
    @staticmethod
    def reference(A, B, As, Bs, block_size=None, output_dtype=torch.float32):
        """PyTorch reference for block-wise FP8 matmul."""
        if block_size is None:
            block_n, block_k = 128, 128
        else:
            block_n, block_k = block_size

        # Dequantize A and B, then matmul
        A_f32 = A.float()
        B_f32 = B.float()

        M = A_f32.shape[0]
        N, K = B_f32.shape

        # Block-wise dequantize A
        A_deq = torch.zeros(M, K, dtype=torch.float32, device=A.device)
        for m in range(M):
            for k_block in range(0, K, block_k):
                k_end = min(k_block + block_k, K)
                scale = As[m, k_block // block_k]
                A_deq[m, k_block:k_end] = A_f32[m, k_block:k_end] * scale

        # Block-wise dequantize B (N, K)
        B_deq = torch.zeros(N, K, dtype=torch.float32, device=B.device)
        for n_block in range(0, N, block_n):
            n_end = min(n_block + block_n, N)
            for k_block in range(0, K, block_k):
                k_end = min(k_block + block_k, K)
                scale = Bs[n_block // block_n, k_block // block_k]
                B_deq[n_block:n_end, k_block:k_end] = B_f32[n_block:n_end, k_block:k_end] * scale

        C = A_deq @ B_deq.T
        return C.to(output_dtype)

    @pytest.mark.parametrize(
        "M, N, K, block_size",
        [
            (128, 128, 128, [128, 128]),
            (256, 256, 256, [128, 128]),
            # Larger shapes
            (512, 1024, 512, [128, 128]),
            (1024, 2048, 1024, [128, 128]),
        ],
    )
    @pytest.mark.parametrize("framework", _backends)
    def test_op(self, M, N, K, block_size, framework, arch):
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        if arch == "sm80":
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        torch.manual_seed(42)
        block_n, block_k = block_size

        A_f32 = torch.randn(M, K, device=DEVICE) * 0.1
        A = A_f32.to(torch.float8_e4m3fn)
        B_f32 = torch.randn(N, K, device=DEVICE) * 0.1
        B = B_f32.to(torch.float8_e4m3fn)

        As = torch.rand(M, K // block_k, dtype=torch.float32, device=DEVICE) + 0.1
        Bs = torch.rand(N // block_n, K // block_k, dtype=torch.float32, device=DEVICE) + 0.1

        self.assertCorrectness(
            w8a8_block_fp8_matmul,
            self.reference,
            {"A": A, "B": B, "As": As, "Bs": Bs, "block_size": block_size},
            rtol=5e-2,
            atol=5e-2,
            check_stride=False,
        )
