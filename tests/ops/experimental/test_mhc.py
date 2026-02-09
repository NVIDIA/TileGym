# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common

X_DTYPE = torch.bfloat16
W_DTYPE = torch.float32
OTHER_DTYPE = torch.float32


class Test_MHC(common.PyTestCase):
    @staticmethod
    def reference_gemm_rmsnorm(x, w):
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)
        y = x_fp32 @ w_fp32
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=1, keepdim=True))
        return y, rms

    @staticmethod
    def reference_scale_bias_sigmoid(y, r, n, alpha_pre, alpha_post, alpha_res, bias):
        scale = torch.empty((y.shape[1],), dtype=torch.float32, device=y.device)
        scale[:n] = alpha_pre
        scale[n : 2 * n] = alpha_post
        scale[2 * n :] = alpha_res
        bias = bias.to(torch.float32)
        linear = y.to(torch.float32) * scale / r + bias
        out = linear.clone()
        out[:, :n] = torch.sigmoid(linear[:, :n])
        out[:, n : 2 * n] = 2.0 * torch.sigmoid(linear[:, n : 2 * n])
        return out.to(y.dtype)

    @staticmethod
    def reference_sinkhorn(y, n):
        y_out = y.clone()
        start = 2 * n
        end = start + n * n
        mat = y_out[:, start:end].to(torch.float32).reshape(-1, n, n)
        mat = torch.exp(mat)
        for _ in range(20):
            mat = mat / mat.sum(dim=2, keepdim=True)
            mat = mat / mat.sum(dim=1, keepdim=True)
        y_out[:, start:end] = mat.reshape(-1, n * n).to(y.dtype)
        return y_out

    @staticmethod
    def reference_apply_residual(x, f_out, y, n):
        B, nC = x.shape
        C = nC // n
        x_view = x.view(B, n, C).to(torch.float32)
        f_view = f_out
        h_post = y[:, n : 2 * n]
        h_res = y[:, 2 * n : 2 * n + n * n].view(B, n, n)
        x_res = torch.matmul(h_res, x_view)
        x_post = h_post.unsqueeze(-1) * f_view.unsqueeze(1)
        x_next = x_res + x_post
        return x_next.to(x.dtype).view(B, nC)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "m, k, n",
        [
            (128, 1024, 4),
            (128, 1024, 8),
            (128, 1000, 8),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_mhc_gemm_rms_scale_bf16_precision(self, m, k, n, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")
        out_n = n * n + 2 * n

        x = torch.randn((m, k), dtype=X_DTYPE, device=device, requires_grad=False)
        w = torch.randn((k, out_n), dtype=W_DTYPE, device=device, requires_grad=False)
        bias = torch.randn((out_n,), dtype=OTHER_DTYPE, device=device, requires_grad=False)
        alpha_pre = 0.8
        alpha_post = 1.1
        alpha_res = 0.9

        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        y_ref, r_ref = self.reference_gemm_rmsnorm(x, w)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        y_ref = self.reference_scale_bias_sigmoid(y_ref, r_ref, n, alpha_pre, alpha_post, alpha_res, bias)
        y_out, r_out = tilegym.ops.mhc_gemm_rms_scale(
            x,
            w,
            n,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
        )
        torch.testing.assert_close(y_out, y_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(r_out, r_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "m, n, dtype",
        [
            (256, 4, OTHER_DTYPE),
            (256, 8, OTHER_DTYPE),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_mhc_sinkhorn(self, m, n, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")
        out_n = n * n + 2 * n

        y = torch.randn((m, out_n), dtype=dtype, device=device, requires_grad=False)
        y_ref = self.reference_sinkhorn(y, n)
        y_test = y.clone()
        y_out = tilegym.ops.mhc_sinkhorn(y_test, n)
        tol = common.get_dtype_tolerances(dtype)
        out_close, msg = common.compare_tensors(y_out, y_ref, rtol=tol["rtol"], atol=tol["atol"])
        assert out_close, "\n".join(msg)

    @pytest.mark.parametrize(
        "m, n, c",
        [
            (128, 4, 1024),
            (64, 8, 2048),
            (128, 2, 1024),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_mhc_apply_residual(self, m, n, c, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        device = torch.device("cuda")
        out_n = n * n + 2 * n
        nC = n * c

        x = torch.randn((m, nC), dtype=X_DTYPE, device=device, requires_grad=False)
        f_out = torch.randn((m, c), dtype=X_DTYPE, device=device, requires_grad=False)
        y = torch.randn((m, out_n), dtype=OTHER_DTYPE, device=device, requires_grad=False)

        y_ref = self.reference_apply_residual(x, f_out, y, n)
        y_out = tilegym.ops.mhc_apply_residual(x, f_out, y, n)
        tol = common.get_dtype_tolerances(X_DTYPE)
        out_close, msg = common.compare_tensors(y_out, y_ref, rtol=tol["rtol"], atol=tol["atol"])
        assert out_close, "\n".join(msg)
