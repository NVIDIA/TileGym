# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
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
        monkeypatch.setenv("DISABLE_AUTOTUNE", "1")
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
        total_num_pages = k_cache.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[1]
        head_dim_qk = q.shape[-1]
        head_dim_vo = v_cache.shape[-1]
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
        monkeypatch.setenv("DISABLE_AUTOTUNE", "1")
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
