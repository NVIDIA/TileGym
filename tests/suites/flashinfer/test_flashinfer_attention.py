# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import warnings

import pytest
import torch

import tilegym
from tests import common
from tests.test_utils import bsr_attention_sample
from tests.test_utils import cudnn_decode
from tests.test_utils import cudnn_prefill
from tilegym.suites.flashinfer import ops as flashinfer_ops

# Check if cuDNN is available for tests that use it as reference
CUDNN_AVAILABLE = cudnn_prefill.CUDNN_AVAILABLE


def _paged_attention_reference(
    q, k_cache, v_cache, actual_seq_lens, actual_seq_offset, block_tables, scale, causal=True
):
    """Pure-PyTorch fp32 reference for paged GQA attention. Used to validate fp8 paths
    where cuDNN reference is unavailable (cuDNN frontend lacks fp8 support).

    KV cache layout: (num_pages, num_kv_heads, page_size, head_dim) — HND.
    Q layout: (total_tokens, num_qo_heads, head_dim).
    """
    device = q.device
    total_tokens, num_qo_heads, head_dim_qk = q.shape
    _, num_kv_heads, page_size, head_dim_vo = v_cache.shape
    group_size = num_qo_heads // num_kv_heads
    batch_size = len(actual_seq_lens)

    q_f = q.float()
    k_f = k_cache.float()
    v_f = v_cache.float()

    out = torch.zeros((total_tokens, num_qo_heads, head_dim_vo), dtype=torch.float32, device=device)

    for b in range(batch_size):
        seq_len = int(actual_seq_lens[b].item())
        q_start = int(actual_seq_offset[b].item())
        q_end = q_start + seq_len  # prefill: q_len == kv_len

        pages_b = block_tables[b]
        num_pages_b = (seq_len + page_size - 1) // page_size

        K_b = k_f[pages_b[:num_pages_b]].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim_qk)[:, :seq_len]
        V_b = v_f[pages_b[:num_pages_b]].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim_vo)[:, :seq_len]

        Q_b = q_f[q_start:q_end].transpose(0, 1)  # (num_qo_heads, seq_len, head_dim_qk)
        K_exp = K_b.repeat_interleave(group_size, dim=0)
        V_exp = V_b.repeat_interleave(group_size, dim=0)

        scores = torch.bmm(Q_b, K_exp.transpose(1, 2)) * scale
        if causal:
            mask = torch.triu(
                torch.ones(scores.shape[-2], scores.shape[-1], dtype=torch.bool, device=device), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out[q_start:q_end] = torch.bmm(attn, V_exp).transpose(0, 1)

    return out


def _paged_decode_reference(q, k_cache, v_cache, actual_seq_lens, block_tables, qk_scale, v_scale=1.0):
    """Pure-PyTorch fp32 reference for paged GQA decode (q_len=1 per batch)."""
    device = q.device
    batch_size, num_qo_heads, head_dim_qk = q.shape
    _, num_kv_heads, page_size, head_dim_vo = v_cache.shape
    group_size = num_qo_heads // num_kv_heads

    q_f = q.float()
    k_f = k_cache.float()
    v_f = v_cache.float()

    out = torch.zeros((batch_size, num_qo_heads, head_dim_vo), dtype=torch.float32, device=device)

    for b in range(batch_size):
        kv_len = int(actual_seq_lens[b].item())
        pages_b = block_tables[b]
        num_pages_b = (kv_len + page_size - 1) // page_size

        K_b = k_f[pages_b[:num_pages_b]].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim_qk)[:, :kv_len]
        V_b = v_f[pages_b[:num_pages_b]].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim_vo)[:, :kv_len]
        K_exp = K_b.repeat_interleave(group_size, dim=0)
        V_exp = V_b.repeat_interleave(group_size, dim=0)

        scores = torch.einsum("hd,hkd->hk", q_f[b], K_exp) * qk_scale
        attn = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hk,hkd->hd", attn, V_exp) * v_scale

    return out


def _ragged_attention_reference(q, k_cache, v_cache, actual_seq_lens, actual_seq_offset, scale, causal=True):
    """Pure-PyTorch fp32 reference for ragged GQA attention (no paged KV)."""
    device = q.device
    total_tokens, num_qo_heads, head_dim_qk = q.shape
    _, num_kv_heads, head_dim_vo = v_cache.shape
    group_size = num_qo_heads // num_kv_heads
    batch_size = len(actual_seq_lens)

    q_f = q.float()
    k_f = k_cache.float()
    v_f = v_cache.float()

    out = torch.zeros((total_tokens, num_qo_heads, head_dim_vo), dtype=torch.float32, device=device)

    for b in range(batch_size):
        seq_len = int(actual_seq_lens[b].item())
        start = int(actual_seq_offset[b].item())
        end = start + seq_len

        Q_b = q_f[start:end].transpose(0, 1)
        K_b = k_f[start:end].transpose(0, 1)
        V_b = v_f[start:end].transpose(0, 1)

        K_exp = K_b.repeat_interleave(group_size, dim=0)
        V_exp = V_b.repeat_interleave(group_size, dim=0)

        scores = torch.bmm(Q_b, K_exp.transpose(1, 2)) * scale
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out[start:end] = torch.bmm(attn, V_exp).transpose(0, 1)

    return out


def _mla_decode_reference(q, kv_cache, actual_seq_lens, block_tables, qk_scale, v_scale=1.0):
    """Pure-PyTorch fp32 reference for MLA decode. KV cache is shared as both K and V
    (compressed representation). Assumes q_rope contribution is zero."""
    device = q.device
    batch_size, num_qo_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = kv_cache.shape  # MLA uses num_kv_heads=1

    q_f = q.float()
    kv_f = kv_cache.float()

    out = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)

    for b in range(batch_size):
        kv_len = int(actual_seq_lens[b].item())
        pages_b = block_tables[b]
        num_pages_b = (kv_len + page_size - 1) // page_size

        KV_b = kv_f[pages_b[:num_pages_b]].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :kv_len]
        KV_exp = KV_b.expand(num_qo_heads, kv_len, head_dim)

        scores = torch.einsum("hd,hkd->hk", q_f[b], KV_exp) * qk_scale
        attn = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hk,hkd->hd", attn, KV_exp) * v_scale

    return out


def get_prefill_problem_configs(quick_run=False, full_run=False):
    if quick_run:
        return [(num_batch, s_kv, head_dim_qk) for num_batch in [4] for s_kv in [1024] for head_dim_qk in [128, 192]]
    if full_run:
        return [
            (num_batch, s_kv, head_dim_qk)
            for num_batch in [1, 16, 32, 64, 100]
            for s_kv in [256, 1024, 2048, 4096, 8192]
            for head_dim_qk in [128, 192]
        ]
    else:
        return (
            [  # small problem sizes
                (1, s_kv, 128) for s_kv in [256, 1024]
            ]
            + [  # normal problem sizes
                (16, 1024, head_dim_qk) for head_dim_qk in [128]
            ]
            + [  # large problem sizes
                (100, 4096, head_dim_qk) for head_dim_qk in [128]
            ]
        )


class Test_FlashInfer_PrefillPaged(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize("dtype", ["float16"])
    @pytest.mark.parametrize("page_size", [128])
    @pytest.mark.parametrize("num_batch, s_kv, head_dim_qk", get_prefill_problem_configs(quick_run=True))
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        page_size,
        num_batch,
        s_kv,
        head_dim_qk,
        backend,
        dtype,
        monkeypatch,
    ):
        monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "1")
        # Convert string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        dtype = dtype_map[dtype]
        self.setUp()
        if backend != "pytorch" and tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")
        (
            q,
            k_cache,
            v_cache,
            scale,
            actual_seq_lens,
            block_tables,
            actual_seq_offset,
            q_indptr,
            k_indptr,
            v_indptr,
            o_indptr,
            lse_indptr,
        ) = bsr_attention_sample.generate_sample_data(
            batch_size=num_batch,
            max_seq_len=s_kv,
            head_dim_qk=head_dim_qk,
            page_size=page_size,
            is_decode=False,
            device=device,
            dtype=torch.float16,
        )

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        num_qo_heads = q.shape[1]
        head_dim_qk = q.shape[-1]
        lse = torch.zeros([q.shape[0], num_qo_heads], device=device, dtype=torch.float32)
        k_scale = 3.444
        v_scale = 2.444

        # Compute reference if cuDNN is available
        out_ref, lse_ref = None, None
        if CUDNN_AVAILABLE:
            out_ref, lse_ref = cudnn_prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache,
                v_cache * v_scale,
                scale * k_scale,
                workspace_buffer,
                max_token_per_sequence=s_kv,
                max_sequence_kv=s_kv,
                actual_seq_lens_q=actual_seq_lens,
                actual_seq_lens_kv=actual_seq_lens,
                block_tables=block_tables,
                causal=True,
                return_lse=True,
                lse=lse,
                is_cuda_graph_compatible=True,
                batch_offsets_q=q_indptr,
                batch_offsets_o=o_indptr,
                batch_offsets_stats=lse_indptr,
            )
        else:
            warnings.warn("cuDNN not available, skipping reference computation")

        try:
            out_impl, lse_impl = flashinfer_ops.prefill_attention_kv_paged(
                q,
                k_cache.transpose(1, 2),
                v_cache.transpose(1, 2),
                actual_seq_lens,
                actual_seq_lens,
                actual_seq_offset,
                block_tables,
                scale * k_scale,
                v_scale,
                num_batch,
                s_kv,
            )
        except Exception as e:
            raise e

        if out_ref is not None:
            torch.testing.assert_close(out_impl, out_ref, atol=1e-2, rtol=2e-1)
            torch.testing.assert_close(lse_impl, lse_ref, atol=1e-2, rtol=2e-1)


def get_prefill_ragged_problem_configs(quick_run=False, full_run=False):
    if quick_run:
        return [(num_batch, s_kv, head_dim_qk) for num_batch in [4] for s_kv in [1024] for head_dim_qk in [128, 192]]
    if full_run:
        return [
            (num_batch, s_kv, head_dim_qk)
            for num_batch in [1, 16, 32, 64, 100]
            for s_kv in [256, 1024, 2048, 4096, 8192]
            for head_dim_qk in [128, 192]
        ]
    else:
        return (
            [  # small problem sizes
                (1, s_kv, 192) for s_kv in [256, 1024]
            ]
            + [  # normal problem sizes
                (16, 1024, head_dim_qk) for head_dim_qk in [192]
            ]
            + [  # normal problem sizes
                (16, 8192, head_dim_qk) for head_dim_qk in [192]
            ]
            + [  # large problem sizes
                (32, 8192, head_dim_qk) for head_dim_qk in [192]
            ]
            + [  # large problem sizes
                (100, 4096, head_dim_qk) for head_dim_qk in [192]
            ]
        )


class Test_FlashInfer_PrefillRagged(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize("dtype", ["float16"])
    @pytest.mark.parametrize("num_batch, s_kv, head_dim_qk", get_prefill_ragged_problem_configs(quick_run=True))
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        num_batch,
        s_kv,
        head_dim_qk,
        dtype,
        backend,
        monkeypatch,
    ):
        monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "1")
        # Convert string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        dtype = dtype_map[dtype]
        self.setUp()
        if backend != "pytorch" and tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        device = torch.device("cuda")
        (
            q,
            k_cache,
            v_cache,
            scale,
            actual_seq_lens,
            block_tables,
            actual_seq_offset,
            q_indptr,
            k_indptr,
            v_indptr,
            o_indptr,
            lse_indptr,
        ) = bsr_attention_sample.generate_sample_data(
            batch_size=num_batch,
            max_seq_len=s_kv,
            head_dim_qk=head_dim_qk,
            is_decode=False,
            device=device,
            dtype=dtype,
        )

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        lse = torch.zeros([q.shape[0], q.shape[1]], device=device, dtype=torch.float32)

        k_scale = 2.414
        v_scale = 1.414

        # Compute reference if cuDNN is available
        out_ref, lse_ref = None, None
        if CUDNN_AVAILABLE:
            out_ref, lse_ref = cudnn_prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache,
                v_cache * v_scale,
                scale * k_scale,
                workspace_buffer,
                max_token_per_sequence=s_kv,
                max_sequence_kv=s_kv,
                actual_seq_lens_q=actual_seq_lens,
                actual_seq_lens_kv=actual_seq_lens,
                block_tables=None,
                causal=True,
                return_lse=True,
                lse=lse,
                is_cuda_graph_compatible=True,
                batch_offsets_q=q_indptr,
                batch_offsets_o=o_indptr,
                batch_offsets_k=k_indptr,
                batch_offsets_v=v_indptr,
                batch_offsets_stats=lse_indptr,
            )
        else:
            warnings.warn("cuDNN not available, skipping reference computation")

        try:
            out_impl, lse_impl = flashinfer_ops.prefill_attention_kv_ragged(
                q,
                k_cache,
                v_cache,
                actual_seq_lens,
                actual_seq_lens,
                actual_seq_offset,
                block_tables,
                scale * k_scale,
                v_scale,
                num_batch,
                s_kv,
            )
        except Exception as e:
            raise e

        if out_ref is not None:
            torch.testing.assert_close(out_impl, out_ref, atol=1e-2, rtol=2e-1)
            torch.testing.assert_close(lse_impl, lse_ref, atol=1e-2, rtol=2e-1)


def get_decoding_problem_configs(quick_run=False, full_run=False):
    if quick_run:
        return [(num_batch, s_kv, page_size) for num_batch in [4] for s_kv in [1024] for page_size in [128]]
    if full_run:
        return [
            (num_batch, s_kv, page_size)
            for num_batch in [1, 16, 32, 64, 200]
            for s_kv in [256, 1024, 2048, 4096, 8192]
            for page_size in [64, 128, 256]
        ]
    else:
        return (
            [  # small problem sizes
                (1, skv, ps) for skv in [256, 1024] for ps in [32, 64]
            ]
            + [  # normal problem sizes
                (16, skv, ps) for skv in [1024, 2048] for ps in [32, 64]
            ]
            + [  # large problem sizes
                (200, skv, ps) for skv in [8192] for ps in [32, 64]
            ]
        )


class Test_FlashInfer_DecodePaged(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize("dtype", ["float16"])
    @pytest.mark.parametrize("num_batch, s_kv, page_size", get_decoding_problem_configs(quick_run=True))
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        num_batch,
        s_kv,
        page_size,
        backend,
        dtype,
    ):
        # Convert string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        dtype = dtype_map[dtype]
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        (
            q,
            k_cache,
            v_cache,
            scale,
            actual_seq_lens,
            block_tables,
            actual_seq_offset,
            q_indptr,
            k_indptr,
            v_indptr,
            o_indptr,
            lse_indptr,
        ) = bsr_attention_sample.generate_sample_data(
            batch_size=num_batch,
            max_seq_len=s_kv,
            page_size=page_size,
            is_decode=True,
            device=device,
            dtype=dtype,
        )
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

        k_scale = 3.678
        v_scale = 2.678

        # Compute reference if cuDNN is available
        out_ref = None
        if CUDNN_AVAILABLE:
            out_ref = cudnn_decode.cudnn_batch_decode_with_kv_cache(
                q,
                k_cache,
                v_cache * v_scale,
                scale * k_scale,
                workspace_buffer,
                max_sequence_kv=s_kv,
                actual_seq_lens_kv=actual_seq_lens,
                block_tables=block_tables,
                is_cuda_graph_compatible=True,
                batch_offsets_q=q_indptr,
                batch_offsets_o=o_indptr,
            )
        else:
            warnings.warn("cuDNN not available, skipping reference computation")

        max_seq_len = actual_seq_lens.cpu().max().item()

        out_ref0 = flashinfer_ops.decode_attention_kv_paged(
            q,
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=False,
            force_persistent=False,
        )
        if out_ref is not None:
            torch.testing.assert_close(out_ref0, out_ref, atol=1e-2, rtol=2e-1)

        out_ref1 = flashinfer_ops.decode_attention_kv_paged(
            q,
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=False,
            force_persistent=True,
        )
        if out_ref is not None:
            torch.testing.assert_close(out_ref1, out_ref, atol=1e-2, rtol=2e-1)

        out_ref2 = flashinfer_ops.decode_attention_kv_paged(
            q,
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=True,
            force_persistent=False,
        )
        if out_ref is not None:
            torch.testing.assert_close(out_ref2, out_ref, atol=1e-2, rtol=2e-1)

        out_ref3 = torch.empty_like(out_ref0)
        out_ref3 = flashinfer_ops.decode_attention_kv_paged(
            q,
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=True,
            force_persistent=False,
            outputs=out_ref3,
        )
        torch.testing.assert_close(out_ref2, out_ref3)


class Test_FlashInfer_MLADecodePaged(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize("dtype", ["float16"])
    @pytest.mark.parametrize("num_batch, s_kv, page_size", get_decoding_problem_configs(quick_run=True))
    @pytest.mark.parametrize("num_heads", [32])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        num_batch,
        s_kv,
        page_size,
        num_heads,
        backend,
        dtype,
    ):
        if torch.cuda.get_device_capability()[0] == 12:
            pytest.xfail("Skip due to random result mismatch in sm120")

        # Convert string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        dtype = dtype_map[dtype]
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        head_dim_rope = 64
        head_dim_qk = 512
        num_qo_heads = num_heads
        (
            (q, q_rope),
            kv_cache,
            k_rope,
            scale,
            actual_seq_lens,
            block_tables,
            actual_seq_offset,
            q_indptr,
            _,
            _,
            _,
            _,
        ) = bsr_attention_sample.generate_sample_data(
            batch_size=num_batch,
            max_seq_len=s_kv,
            page_size=page_size,
            heads=num_qo_heads,
            group_size=num_qo_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_qk,
            head_dim_rope=head_dim_rope,
            is_decode=True,
            device=device,
            dtype=dtype,
        )

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

        k_scale = 1.678
        v_scale = 2.821

        # Compute reference if cuDNN is available
        out_ref = None
        if CUDNN_AVAILABLE:
            out_ref = cudnn_decode.cudnn_batch_decode_with_kv_cache(
                q,
                kv_cache,
                kv_cache * v_scale,
                scale * k_scale,
                workspace_buffer,
                max_sequence_kv=s_kv,
                actual_seq_lens_kv=actual_seq_lens,
                block_tables=block_tables,
                is_cuda_graph_compatible=True,
                batch_offsets_q=q_indptr,
                batch_offsets_o=q_indptr,
            )
        else:
            warnings.warn("cuDNN not available, skipping reference computation")

        max_seq_len = actual_seq_lens.cpu().max().item()
        out_ref0 = flashinfer_ops.decode_mla_kv_paged(
            q,
            q_rope.zero_(),
            kv_cache.reshape(-1, page_size, head_dim_qk),
            k_rope.reshape(-1, page_size, head_dim_rope),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=False,
            force_persistent=False,
        )
        if out_ref is not None:
            torch.testing.assert_close(out_ref0, out_ref, atol=1e-2, rtol=2e-1)

        out_ref1 = flashinfer_ops.decode_mla_kv_paged(
            q,
            q_rope.zero_(),
            kv_cache.reshape(-1, page_size, head_dim_qk),
            k_rope.reshape(-1, page_size, head_dim_rope),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=False,
            force_persistent=True,
        )
        if out_ref is not None:
            torch.testing.assert_close(out_ref1, out_ref, atol=1e-2, rtol=2e-1)

        out_ref2 = flashinfer_ops.decode_mla_kv_paged(
            q,
            q_rope.zero_(),
            kv_cache.reshape(-1, page_size, head_dim_qk),
            k_rope.reshape(-1, page_size, head_dim_rope),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=True,
            force_persistent=False,
        )
        if out_ref is not None:
            torch.testing.assert_close(out_ref2, out_ref, atol=1e-2, rtol=2e-1)

        out_ref3 = torch.empty_like(out_ref0)
        out_ref3 = flashinfer_ops.decode_mla_kv_paged(
            q,
            q_rope.zero_(),
            kv_cache.reshape(-1, page_size, head_dim_qk),
            k_rope.reshape(-1, page_size, head_dim_rope),
            actual_seq_lens,
            block_tables,
            scale * k_scale,
            v_scale,
            max_seq_len=max_seq_len,
            force_split_kv=True,
            force_persistent=False,
            outputs=out_ref3,
        )
        torch.testing.assert_close(out_ref2, out_ref3)
