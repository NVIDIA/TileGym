# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest

try:
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    pytest.skip("transformers not installed, skipping test_rope.py", allow_module_level=True)

import torch

import tilegym

if HAS_TRANSFORMERS:
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .. import common


class Test_RoPE(common.PyTestCase):
    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def reference_partial_rope(q, k, cos, sin):
        """Reference: rotate first rope_dim dims, passthrough the rest.
        Also works for full RoPE (passthrough slice is empty when rope_dim == head_dim)."""
        rope_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot = (q_rot * cos) + (Test_RoPE.rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (Test_RoPE.rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "bsz, seq_len, num_q_heads, num_kv_heads, head_dim, partial_rotary_factor",
        [
            # Full RoPE (partial_rotary_factor=1.0)
            (1, 128, 32, 32, 64, 1.0),
            (2, 128, 32, 32, 64, 1.0),
            # different q/k heads
            (1, 128, 32, 8, 64, 1.0),
            (2, 128, 32, 8, 64, 1.0),
            # Weird shapes
            pytest.param(3, 423, 73, 213, 92, 1.0, marks=pytest.mark.skip(reason="only support atol 1e-1")),
            pytest.param(3, 423, 73, 155, 92, 1.0, marks=pytest.mark.skip(reason="only support atol 1e-1")),
            # Partial RoPE: Qwen3.5 config (head_dim=256, partial_rotary_factor=0.25 → rope_dim=64)
            (1, 128, 32, 8, 256, 0.25),
            (2, 128, 32, 8, 256, 0.25),
            # Partial RoPE: 50% rotation
            (1, 128, 32, 32, 128, 0.5),
            (2, 128, 32, 8, 128, 0.5),
        ],
    )
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            pytest.param(torch.float32, 1e-5, 1e-5),
            pytest.param(torch.bfloat16, 1e-2, 1e-2),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        bsz,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        partial_rotary_factor,
        dtype,
        atol,
        rtol,
        backend,
    ):
        if dtype == torch.bfloat16:
            pytest.skip("random result mismatch on tilegym bfloat16 rope")

        self.setUp()
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Failed to set backend {backend}: {e}")

        device = torch.device("cuda")
        rope_dim = int(head_dim * partial_rotary_factor)

        _tensor_q = (
            torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device)
            .normal_(mean=0.0, std=1.0)
            .transpose(1, 2)
            .to(dtype)
        )
        _tensor_k = (
            torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device)
            .normal_(mean=0.0, std=1.0)
            .transpose(1, 2)
            .to(dtype)
        )

        q1 = _tensor_q.clone().requires_grad_(True)
        k1 = _tensor_k.clone().requires_grad_(True)

        q2 = _tensor_q.clone().requires_grad_(True)
        k2 = _tensor_k.clone().requires_grad_(True)

        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        rotary_emb = LlamaRotaryEmbedding(
            config=LlamaConfig(num_kv_heads=num_kv_heads, head_dim=rope_dim), device=device
        )
        cos, sin = rotary_emb(_tensor_k, pos_ids)
        # Validate forward pass
        dq, dk = (
            torch.randn_like(q1, device=device),
            torch.randn_like(k1, device=device).to(dtype),
        )

        if partial_rotary_factor < 1.0:
            # Partial RoPE: use our reference (slice + rotate + cat)
            hf_q, hf_k = self.reference_partial_rope(q1, k1, cos, sin)
        else:
            # Full RoPE: use HuggingFace reference as gold standard
            hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin)
        tt_q, tt_k = tilegym.ops.apply_rope_base(q2, k2, cos, sin, partial_rotary_factor=partial_rotary_factor)
        torch.testing.assert_close(hf_q, tt_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(hf_k, tt_k, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "bsz, seq_len, num_q_heads, num_kv_heads, head_dim, partial_rotary_factor",
        [
            (1, 128, 32, 32, 64, 1.0),
            (2, 128, 32, 8, 64, 1.0),
            (1, 512, 16, 16, 128, 1.0),
            # Partial RoPE
            (1, 128, 32, 8, 256, 0.25),
            (2, 128, 32, 8, 128, 0.5),
        ],
    )
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.float16, 2e-2, 2e-2),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(
        self,
        bsz,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        partial_rotary_factor,
        dtype,
        atol,
        rtol,
        backend,
    ):
        self.setUp()
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Failed to set backend {backend}: {e}")

        device = torch.device("cuda")
        rope_dim = int(head_dim * partial_rotary_factor)

        q_base = (
            torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device)
            .normal_(mean=0.0, std=1.0)
            .transpose(1, 2)
            .to(dtype)
        )
        k_base = (
            torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device)
            .normal_(mean=0.0, std=1.0)
            .transpose(1, 2)
            .to(dtype)
        )

        q_ref = q_base.clone().requires_grad_(True)
        k_ref = k_base.clone().requires_grad_(True)
        q_tt = q_base.clone().requires_grad_(True)
        k_tt = k_base.clone().requires_grad_(True)

        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        rotary_emb = LlamaRotaryEmbedding(
            config=LlamaConfig(num_kv_heads=num_kv_heads, head_dim=rope_dim), device=device
        )
        cos, sin = rotary_emb(k_ref, pos_ids)

        if partial_rotary_factor < 1.0:
            ref_q, ref_k = self.reference_partial_rope(q_ref, k_ref, cos, sin)
        else:
            ref_q, ref_k = apply_rotary_pos_emb(q_ref, k_ref, cos, sin)
        tt_q, tt_k = tilegym.ops.apply_rope_base(q_tt, k_tt, cos, sin, partial_rotary_factor=partial_rotary_factor)

        grad_q = torch.randn_like(ref_q)
        grad_k = torch.randn_like(ref_k)
        ((ref_q * grad_q).sum() + (ref_k * grad_k).sum()).backward()
        ((tt_q * grad_q).sum() + (tt_k * grad_k).sum()).backward()

        torch.testing.assert_close(q_ref.grad, q_tt.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_ref.grad, k_tt.grad, atol=atol, rtol=rtol)
