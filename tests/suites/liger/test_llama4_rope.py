# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import llama4_rope


def _make_freqs_cis(seq_len, head_dim, device, dtype=torch.float32):
    """Build a simple rotary frequency tensor of shape (seq_len, head_dim//2, 2)."""
    half = head_dim // 2
    # theta = 1 / (10000 ^ (2i / head_dim)) for i in [0, half)
    theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32, device=device) * 2.0 / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, theta)  # (seq_len, half)
    # return as (seq_len, half, 2) real tensor (cos, sin pairs)
    freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1).to(dtype)
    return freqs_cis


def _reference_rope(q, k, freqs_cis):
    """PyTorch reference for RoPE: complex multiplication in float32."""
    q_f = q.float()
    k_f = k.float()

    # freqs_cis: (seq_len, head_dim//2, 2) or (seq_len, head_dim)
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.view(freqs_cis.shape[0], -1, 2)
    freqs_cis_f = freqs_cis.float()  # (seq, half, 2)
    f_r = freqs_cis_f[..., 0]  # (seq, half)
    f_i = freqs_cis_f[..., 1]  # (seq, half)

    def apply_rope(x):
        # x: (B, S, H, D), treat as (B, S, H, half, 2)
        B, S, H, D = x.shape
        x_r = x[..., 0::2]  # (B, S, H, half) real
        x_i = x[..., 1::2]  # (B, S, H, half) imag
        # expand freqs to (1, S, 1, half)
        fr = f_r.unsqueeze(0).unsqueeze(2)  # (1, S, 1, half)
        fi = f_i.unsqueeze(0).unsqueeze(2)
        new_r = x_r * fr - x_i * fi
        new_i = x_r * fi + x_i * fr
        out = torch.stack([new_r, new_i], dim=-1).reshape(B, S, H, D)
        return out

    return apply_rope(q_f).to(q.dtype), apply_rope(k_f).to(k.dtype)


class Test_Liger_Llama4Rope(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 4, 64), torch.float32),
            ((1, 16, 8, 128), torch.float32),
            ((2, 8, 4, 64), torch.float16),
            ((2, 8, 4, 64), torch.bfloat16),
            ((1, 4, 2, 48), torch.float32),  # non-power-of-2 head_dim
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend, monkeypatch):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        B, S, H_q, D = shape
        H_k = H_q // 2 if H_q > 1 else H_q  # GQA-style: fewer key heads

        q = torch.randn(B, S, H_q, D, dtype=dtype, device=device)
        k = torch.randn(B, S, H_k, D, dtype=dtype, device=device)
        freqs_cis = _make_freqs_cis(S, D, device, dtype=torch.float32)

        def ref_fn(q=q, k=k, freqs_cis=freqs_cis):
            return _reference_rope(q, k, freqs_cis)

        def fw_fn(q=q, k=k, freqs_cis=freqs_cis):
            return llama4_rope(q.clone(), k.clone(), freqs_cis)

        q_ref, k_ref = ref_fn()
        q_out, k_out = fw_fn()

        assert torch.allclose(q_out.float(), q_ref.float(), atol=1e-2, rtol=1e-2), (
            f"q mismatch: max_diff={(q_out.float() - q_ref.float()).abs().max()}"
        )
        assert torch.allclose(k_out.float(), k_ref.float(), atol=1e-2, rtol=1e-2), (
            f"k mismatch: max_diff={(k_out.float() - k_ref.float()).abs().max()}"
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 4, 64), torch.float32),
            ((2, 8, 4, 64), torch.float16),
            ((2, 8, 4, 64), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass using conjugate rotation (imag_sign=-1)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        B, S, H_q, D = shape
        H_k = H_q

        q = torch.randn(B, S, H_q, D, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(B, S, H_k, D, dtype=dtype, device=device, requires_grad=True)
        freqs_cis = _make_freqs_cis(S, D, device, dtype=torch.float32)

        dout_q = torch.ones(B, S, H_q, D, dtype=dtype, device=device)
        dout_k = torch.ones(B, S, H_k, D, dtype=dtype, device=device)

        # Forward + backward
        q_out, k_out = llama4_rope(q.clone().requires_grad_(True), k.clone().requires_grad_(True), freqs_cis)
        # Backward should not raise
        (q_out.sum() + k_out.sum()).backward()
