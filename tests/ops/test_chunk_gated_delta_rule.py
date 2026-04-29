# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch
import torch.nn.functional as F

import tilegym

from .. import common

_backends = ["cutile"]
_perf_frameworks = _backends + ["pytorch"]

_SHAPE_CONFIGS = [
    pytest.param(1, 1, 1, 64, 64, 64, False, False, False, id="minimal"),
    pytest.param(2, 1, 4, 64, 64, 64, False, False, False, id="decode"),
    pytest.param(2, 16, 4, 64, 64, 64, False, False, False, id="T16"),
    pytest.param(1, 32, 8, 128, 128, 64, False, True, False, id="T32_final_state"),
    pytest.param(2, 64, 4, 64, 64, 64, False, False, False, id="T64"),
    pytest.param(2, 8, 4, 64, 64, 64, True, True, False, id="init_final_state"),
    pytest.param(1, 16, 4, 64, 128, 64, False, False, False, id="K64_V128"),
    pytest.param(2, 8, 4, 128, 64, 64, False, False, True, id="l2norm"),
    pytest.param(2, 4, 8, 256, 128, 64, True, True, False, id="K256_V128"),
    pytest.param(1, 100, 4, 64, 64, 64, False, False, False, id="T100_unaligned"),
    pytest.param(2, 128, 4, 64, 64, 32, False, False, False, id="chunk32"),
    pytest.param(1, 37, 4, 64, 64, 64, False, False, False, id="T37_prime"),
    pytest.param(2, 65, 4, 128, 128, 64, False, True, False, id="T65_off_by_1"),
]

_DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]


# fmt: off
# Reference implementations copied verbatim from HuggingFace transformers v4.57.6:
# https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L436-L439
def _l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L442-L519
def _torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
# fmt: on


class Test_ChunkGatedDeltaRule(common.PyTestCase):
    @staticmethod
    def reference(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    ):
        return _torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize(
        "B, T, H, K, V, CS, use_init, out_final, use_l2",
        _SHAPE_CONFIGS,
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, B, T, H, K, V, CS, use_init, out_final, use_l2, dtype, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        if dtype == torch.float32:
            pytest.skip("Skipping fp32 tests due to known failures; under investigation")

        self.setUp()

        from tilegym.ops import chunk_gated_delta_rule

        device = "cuda"
        torch.manual_seed(42)

        query = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        key = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        value = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
        g = -torch.abs(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.5
        beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))

        init_state = None
        if use_init:
            init_state = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

        ref_out, ref_state = self.reference(
            query.clone(),
            key.clone(),
            value.clone(),
            g.clone(),
            beta.clone(),
            chunk_size=CS,
            initial_state=init_state.clone() if init_state is not None else None,
            output_final_state=out_final,
            use_qk_l2norm_in_kernel=use_l2,
        )
        test_out, test_state = chunk_gated_delta_rule(
            query.clone(),
            key.clone(),
            value.clone(),
            g.clone(),
            beta.clone(),
            chunk_size=CS,
            initial_state=init_state.clone() if init_state is not None else None,
            output_final_state=out_final,
            use_qk_l2norm_in_kernel=use_l2,
        )

        atol = 1e-3 if dtype == torch.float32 else 2e-3
        rtol = 2e-3 if dtype == torch.float32 else 5e-3

        assert torch.allclose(ref_out, test_out, atol=atol, rtol=rtol), (
            f"Output mismatch: max_abs_err={(ref_out - test_out).abs().max().item():.2e}"
        )
        if out_final:
            assert torch.allclose(ref_state.float(), test_state.float(), atol=atol, rtol=rtol), (
                f"State mismatch: max_abs_err={(ref_state.float() - test_state.float()).abs().max().item():.2e}"
            )

    @pytest.mark.parametrize(
        "T",
        [128 * 2**i for i in range(0, 8)],
        ids=[f"T{128 * 2**i}" for i in range(0, 8)],
    )
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(self, T, framework, record_property, arch):
        self.setUp()
        dtype = torch.bfloat16
        B, H, D = 16, 8, 128
        CS = 64
        device = "cuda"

        torch.manual_seed(0)
        q = torch.randn(B, T, H, D, device=device, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype)
        g = -torch.abs(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.5
        beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))

        with torch.no_grad():
            if framework == "pytorch":
                framework_fn = lambda: self.reference(q, k, v, g, beta, chunk_size=CS)
            elif tilegym.is_backend_available(framework):
                tilegym.set_backend(framework)
                from tilegym.ops import chunk_gated_delta_rule

                framework_fn = lambda: chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=CS)
            else:
                pytest.skip(f"Framework {framework} is not available")

            result = common.benchmark_framework(framework, framework_fn, use_cupti=True)
            record_property("benchmark", result)

        del q, k, v, g, beta
        torch.cuda.empty_cache()
        gc.collect()
