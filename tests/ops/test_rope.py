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
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
        [
            (1, 128, 32, 32, 64),
            (2, 128, 32, 32, 64),
            # different q/k heads
            (1, 128, 32, 8, 64),
            (2, 128, 32, 8, 64),
            # Weird shapes
            pytest.param(3, 423, 73, 213, 92, marks=pytest.mark.skip(reason="only support atol 1e-1")),
            pytest.param(3, 423, 73, 155, 92, marks=pytest.mark.skip(reason="only support atol 1e-1")),
        ],
    )
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            pytest.param(torch.float32, 1e-5, 1e-5),
            pytest.param(torch.bfloat16, 1e-2, 1e-2),
        ],
    )
    @pytest.mark.parametrize(
        "expand_position_ids",
        [True, False],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        bsz,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        dtype,
        expand_position_ids,
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

        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        if expand_position_ids:
            pos_ids = pos_ids.expand(bsz, -1)

        rotary_emb = LlamaRotaryEmbedding(
            config=LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), device=device
        )
        cos, sin = rotary_emb(k1, pos_ids)
        # Validate forward pass
        dq, dk = (
            torch.randn_like(q1, device=device),
            torch.randn_like(k1, device=device).to(dtype),
        )

        hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin, pos_ids)
        tt_q, tt_k = tilegym.ops.apply_rope_base(q2, k2, cos, sin, pos_ids)
        torch.testing.assert_close(hf_q, tt_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(hf_k, tt_k, atol=atol, rtol=rtol)
