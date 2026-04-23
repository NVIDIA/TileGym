# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym
from tests import common
from tilegym.backend import set_backend

_backends = ["cutile"]


def swa_reference(q, k, v, window_size, is_causal=True, scaling=None):
    """Pure PyTorch fp32 reference. Materializes the full SxS mask."""
    B, H, S_Q, D = q.shape
    S_K = k.shape[2]
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scaling
    i = torch.arange(S_Q, device=q.device).unsqueeze(1)
    j = torch.arange(S_K, device=q.device).unsqueeze(0)
    mask = j > (i - window_size)
    if is_causal:
        mask = mask & (j <= i)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


class TestSWAAttention(common.PyTestCase):
    def _run_test(self, B, H, S, D, W, dtype, backend):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()

        device = torch.device("cuda")
        q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
        k = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
        v = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.5)

        def test_fn(q, k, v, window_size, is_causal):
            return tilegym.ops.swa_attention(q, k, v, window_size=window_size, is_causal=is_causal)

        self.assertCorrectness(
            test_fn,
            swa_reference,
            kwargs={"q": q, "k": k, "v": v, "window_size": W, "is_causal": True},
            rtol=1e-2,
            atol=5e-2,
        )

    # -- basic correctness --

    @pytest.mark.parametrize("backend", _backends)
    def test_op_window_equals_seq(self, backend):
        self._run_test(B=1, H=1, S=128, D=128, W=128, dtype=torch.float16, backend=backend)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_small_window(self, backend):
        self._run_test(B=1, H=1, S=256, D=128, W=128, dtype=torch.float16, backend=backend)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_window_of_one(self, backend):
        self._run_test(B=1, H=1, S=128, D=128, W=1, dtype=torch.float16, backend=backend)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_multi_head(self, backend):
        self._run_test(B=2, H=8, S=256, D=128, W=128, dtype=torch.float16, backend=backend)

    # -- edge cases --

    @pytest.mark.parametrize("backend", _backends)
    def test_op_seq_not_divisible_by_tile(self, backend):
        self._run_test(B=1, H=1, S=100, D=128, W=64, dtype=torch.float16, backend=backend)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_window_equals_tile(self, backend):
        self._run_test(B=1, H=1, S=256, D=128, W=64, dtype=torch.float16, backend=backend)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_first_tokens(self, backend):
        self._run_test(B=1, H=1, S=64, D=128, W=128, dtype=torch.float16, backend=backend)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_very_small_seq(self, backend):
        self._run_test(B=1, H=1, S=16, D=128, W=8, dtype=torch.float16, backend=backend)

    # -- GQA (Grouped-Query Attention) --

    @pytest.mark.parametrize("backend", _backends)
    def test_op_gqa(self, backend):
        """Mistral-style GQA: 32 Q heads, 8 KV heads (4:1 ratio)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()

        B, H, H_K, S, D, W = 1, 32, 8, 256, 128, 128
        dtype = torch.float16
        device = torch.device("cuda")

        q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
        k = torch.empty(B, H_K, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
        v = torch.empty(B, H_K, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.5)

        # the kernel handles GQA expansion internally
        out = tilegym.ops.swa_attention(q, k, v, window_size=W, is_causal=True, backend=backend)

        # reference uses pre-expanded KV so it operates on matched head counts
        k_exp = k.repeat_interleave(H // H_K, dim=1)
        v_exp = v.repeat_interleave(H // H_K, dim=1)
        ref = swa_reference(q, k_exp, v_exp, window_size=W, is_causal=True)

        self.assertAllClose(out, ref, rtol=1e-2, atol=5e-2)

    # -- various shapes --

    @pytest.mark.parametrize(
        "backend,S,W",
        [(b, s, w) for b in _backends for s, w in [(512, 256), (1024, 512), (2048, 1024), (4096, 2048), (4096, 4096)]],
    )
    def test_op_various_configs(self, backend, S, W):
        self._run_test(B=1, H=1, S=S, D=128, W=W, dtype=torch.float16, backend=backend)

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _backends)
    def test_op_long_context_mistral(self, backend):
        # Mistral-style: 8K context, 4K window.
        # Marked slow: materializes an 8192x8192 fp32 reference matrix.
        self._run_test(B=1, H=1, S=8192, D=128, W=4096, dtype=torch.float16, backend=backend)
