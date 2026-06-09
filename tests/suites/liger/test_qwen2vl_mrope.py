# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import qwen2vl_mrope


def _make_cos_sin(bsz, seq_len, head_dim, device, dtype=torch.float32):
    """
    Build cos/sin tensors of shape (3, bsz, seq_len, head_dim).

    Temporal section: cos[0], Height section: cos[1], Width section: cos[2].
    """
    half = head_dim // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32, device=device) * 2.0 / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, theta)  # (seq_len, half)
    cos_1d = freqs.cos()  # (seq_len, half)
    sin_1d = freqs.sin()  # (seq_len, half)

    # Full head_dim: pad right half with zeros (only first half used for RoPE)
    cos_full = torch.cat([cos_1d, torch.zeros_like(cos_1d)], dim=-1)  # (seq_len, head_dim)
    sin_full = torch.cat([sin_1d, torch.zeros_like(sin_1d)], dim=-1)

    # Expand to (3, bsz, seq_len, head_dim) — same values across sections for simplicity
    cos_3d = cos_full.unsqueeze(0).unsqueeze(0).expand(3, bsz, seq_len, head_dim).contiguous().to(dtype)
    sin_3d = sin_full.unsqueeze(0).unsqueeze(0).expand(3, bsz, seq_len, head_dim).contiguous().to(dtype)
    return cos_3d, sin_3d


def _reference_mrope(q, k, cos, sin, mrope_section):
    """
    PyTorch float32 reference for Qwen2VL M-RoPE.

    q: (bsz, n_q_heads, seq_len, head_dim)
    k: (bsz, n_k_heads, seq_len, head_dim)
    cos/sin: (3, bsz, seq_len, head_dim)
    mrope_section: [t_section, h_section]
    """
    bsz, n_q_heads, seq_len, head_dim = q.shape
    hd_half = head_dim // 2

    t_end = mrope_section[0]
    h_end = t_end + mrope_section[1]

    # Build effective cos/sin: (bsz, seq_len, hd_half)
    cos_eff = torch.zeros(bsz, seq_len, hd_half, dtype=torch.float32, device=q.device)
    sin_eff = torch.zeros(bsz, seq_len, hd_half, dtype=torch.float32, device=q.device)

    cos_eff[:, :, :t_end] = cos[0, :, :, :t_end].float()
    sin_eff[:, :, :t_end] = sin[0, :, :, :t_end].float()
    cos_eff[:, :, t_end:h_end] = cos[1, :, :, t_end:h_end].float()
    sin_eff[:, :, t_end:h_end] = sin[1, :, :, t_end:h_end].float()
    cos_eff[:, :, h_end:] = cos[2, :, :, h_end:hd_half].float()
    sin_eff[:, :, h_end:] = sin[2, :, :, h_end:hd_half].float()

    def apply(x, c, s):
        # x: (bsz, n_heads, seq_len, head_dim)
        x_f = x.float()
        x_r = x_f[..., :hd_half]  # left half
        x_i = x_f[..., hd_half:]  # right half
        # c, s: (bsz, seq_len, hd_half) → (bsz, 1, seq_len, hd_half)
        c_exp = c.unsqueeze(1)
        s_exp = s.unsqueeze(1)
        new_r = x_r * c_exp - x_i * s_exp
        new_i = x_i * c_exp + x_r * s_exp
        return torch.cat([new_r, new_i], dim=-1).to(x.dtype)

    return apply(q, cos_eff, sin_eff), apply(k, cos_eff, sin_eff)


class Test_Liger_Qwen2VLMRope(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 16, 64), torch.float32),
            ((1, 4, 8, 128), torch.float32),
            ((2, 8, 16, 64), torch.float16),
            ((2, 8, 16, 64), torch.bfloat16),
            ((1, 2, 4, 48), torch.float32),  # non-power-of-2 head_dim
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, shape, dtype, backend, monkeypatch):
        """Test M-RoPE forward pass."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        bsz, n_q_heads, seq_len, head_dim = shape
        n_k_heads = max(1, n_q_heads // 2)
        hd_half = head_dim // 2
        # mrope_section: split head_dim_half into 3 roughly equal parts
        t_sec = hd_half // 3
        h_sec = hd_half // 3
        mrope_section = [t_sec, h_sec]

        q = torch.randn(bsz, n_q_heads, seq_len, head_dim, dtype=dtype, device=device)
        k = torch.randn(bsz, n_k_heads, seq_len, head_dim, dtype=dtype, device=device)
        cos, sin = _make_cos_sin(bsz, seq_len, head_dim, device)

        q_ref, k_ref = _reference_mrope(q, k, cos, sin, mrope_section)
        q_out, k_out = qwen2vl_mrope(q.clone(), k.clone(), cos, sin, mrope_section)

        assert torch.allclose(q_out.float(), q_ref.float(), atol=1e-2, rtol=1e-2), (
            f"q mismatch: max_diff={(q_out.float() - q_ref.float()).abs().max()}"
        )
        assert torch.allclose(k_out.float(), k_ref.float(), atol=1e-2, rtol=1e-2), (
            f"k mismatch: max_diff={(k_out.float() - k_ref.float()).abs().max()}"
        )

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 8, 16, 64), torch.float32),
            ((2, 8, 16, 64), torch.float16),
            ((2, 8, 16, 64), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, shape, dtype, backend, monkeypatch):
        """Test backward pass (gradient flows through M-RoPE)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        bsz, n_q_heads, seq_len, head_dim = shape
        n_k_heads = n_q_heads
        hd_half = head_dim // 2
        t_sec = hd_half // 3
        h_sec = hd_half // 3
        mrope_section = [t_sec, h_sec]

        q = torch.randn(bsz, n_q_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(bsz, n_k_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        cos, sin = _make_cos_sin(bsz, seq_len, head_dim, device)

        q_out, k_out = qwen2vl_mrope(q, k, cos, sin, mrope_section)
        (q_out.sum() + k_out.sum()).backward()

        # Gradient should be non-None
        assert q.grad is not None
        assert k.grad is not None
