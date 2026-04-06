# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for Unsloth MoE Grouped GEMM (forward + backward).

Covers:
  - Forward correctness: small, medium, production-like shapes
  - Backward correctness: gradient VALUE verification for both X and W
  - Edge cases: unequal experts, zero-token experts, single expert, non-power-of-2 dims
  - Deterministic behavior: repeated runs produce identical results
  - Performance: production-representative shapes
"""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.unsloth.ops import grouped_gemm

DEVICE = "cuda"

_backends = ["cutile"]


# Tolerances per dtype (atol, rtol) — matching unsloth conventions
TOLERANCE = {
    torch.bfloat16: (5e-2, 5e-2),
    torch.float16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}


def _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend, **kwargs):
    """Run grouped_gemm with the correct backend-specific kwargs."""
    return grouped_gemm(X, W, m_sizes, topk, gather_indices=gather_indices, **kwargs)


def _skip_if_unavailable(backend):
    """Skip test if backend is not available."""
    if tilegym.is_backend_available(backend):
        tilegym.set_backend(backend)
    else:
        pytest.skip(f"Backend {backend} is not available")


class Test_Unsloth_GroupedGemm(common.PyTestCase):
    @staticmethod
    def reference_forward(X, W, m_sizes, topk):
        """PyTorch reference for grouped GEMM forward: Y = X @ W^T per expert."""
        num_experts = W.shape[0]
        N = W.shape[1]
        total_tokens = X.shape[0]
        Y = torch.zeros(total_tokens, N, device=X.device, dtype=X.dtype)

        offset = 0
        for e in range(num_experts):
            m = m_sizes[e].item()
            if m > 0:
                X_e = X[offset : offset + m]  # (m, K)
                W_e = W[e]  # (N, K)
                Y[offset : offset + m] = X_e @ W_e.T  # (m, N)
            offset += m
        return Y

    # ------------------------------------------------------------------
    # Forward correctness
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            # Small shapes (fast, for basic correctness)
            (4, 16, 64, 64, 1),
            (4, 32, 128, 128, 1),
            (8, 8, 64, 128, 1),
            # Medium shapes
            (8, 64, 512, 256, 1),
            # Production-like shapes (scaled down for CI)
            (16, 32, 2048, 1024, 1),  # Llama4-like: E=16, large N/K
            (16, 16, 768, 512, 1),  # Qwen3-like: E=16, medium N/K
            # topk > 1 (MoE standard: each token routed to multiple experts)
            (8, 16, 128, 128, 2),  # topk=2, DeepSeek-V3 style
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test grouped GEMM forward correctness."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        test_fn = lambda **kw: _run_grouped_gemm(
            kw["X"], kw["W"], kw["m_sizes"], kw["topk"], kw["gather_indices"], backend
        )
        ref_fn = lambda **kw: self.reference_forward(kw["X"], kw["W"], kw["m_sizes"], kw["topk"])

        atol, rtol = TOLERANCE[dtype]
        self.assertCorrectness(
            test_fn,
            ref_fn,
            {"X": X, "W": W, "m_sizes": m_sizes, "topk": topk, "gather_indices": gather_indices},
            rtol=rtol,
            atol=atol,
            check_stride=False,
        )

    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            (128, 8, 768, 512, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward_large(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test forward with very large expert count (Qwen3 E=128). Skipped in CI."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        result = _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend)
        expected = self.reference_forward(X, W, m_sizes, topk)
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "m_sizes_list, N, K",
        [
            # Unequal token distribution
            ([32, 16, 8, 8], 64, 64),
            ([64, 0, 32, 0], 128, 128),  # Some experts get 0 tokens
            ([0, 0, 0, 64], 64, 64),  # Only last expert has tokens
            ([128], 256, 128),  # Single expert
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_edge_cases(self, m_sizes_list, N, K, dtype, backend):
        """Test edge cases: unequal experts, zero-token experts, single expert."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        num_experts = len(m_sizes_list)
        m_sizes = torch.tensor(m_sizes_list, dtype=torch.int32, device=DEVICE)
        total_tokens = sum(m_sizes_list)
        if total_tokens == 0:
            pytest.skip("All-zero m_sizes not meaningful")

        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        result = _run_grouped_gemm(X, W, m_sizes, 1, gather_indices, backend)
        expected = self.reference_forward(X, W, m_sizes, 1)
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    # Non-power-of-2 dims:
    #   CuTile: gather handles partial tiles
    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K",
        [
            (4, 20, 300, 192),
            (8, 13, 100, 96),  # Odd token count + non-power-of-2
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_non_power_of_2(self, num_experts, tokens_per_expert, N, K, dtype, backend):
        """Test with non-power-of-2 dimensions."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert
        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        m_sizes = torch.full((num_experts,), tokens_per_expert, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        result = _run_grouped_gemm(X, W, m_sizes, 1, gather_indices, backend)
        expected = self.reference_forward(X, W, m_sizes, 1)
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    # ------------------------------------------------------------------
    # Backward correctness — gradient VALUE verification
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            (4, 16, 64, 64, 1),
            (4, 32, 128, 128, 1),
            (8, 64, 512, 256, 1),  # Medium shape
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_X_grad(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test backward: verify X gradient VALUES against PyTorch reference."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        # Upstream gradient for reproducible backward
        dY = torch.randn(total_tokens, N, dtype=dtype, device=DEVICE) * 0.1

        # --- Reference backward (PyTorch) ---
        # NOTE: scale BEFORE requires_grad_ so X_ref remains a leaf tensor
        X_ref = (torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1).requires_grad_(True)
        W_ref = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        Y_ref = self.reference_forward(X_ref, W_ref, m_sizes, topk)
        Y_ref.backward(dY)
        X_grad_ref = X_ref.grad.clone()

        # --- Kernel backward ---
        X_test = X_ref.detach().clone().requires_grad_(True)
        W_test = W_ref.detach().clone()
        Y_test = _run_grouped_gemm(X_test, W_test, m_sizes, topk, gather_indices, backend)
        Y_test.backward(dY)

        assert X_test.grad is not None, "Gradient should flow back to X"
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(X_test.grad, X_grad_ref, rtol=rtol, atol=atol, msg="X gradient mismatch")

    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            (4, 16, 64, 64, 1),
            (4, 32, 128, 128, 1),
            (8, 64, 512, 256, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_W_grad(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend, request):
        """Test backward: verify W gradient VALUES against PyTorch reference."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        dY = torch.randn(total_tokens, N, dtype=dtype, device=DEVICE) * 0.1

        # --- Reference backward (PyTorch) ---
        X_ref = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        # NOTE: scale BEFORE requires_grad_ so W_ref remains a leaf tensor
        W_ref = (torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1).requires_grad_(True)
        Y_ref = self.reference_forward(X_ref, W_ref, m_sizes, topk)
        Y_ref.backward(dY)
        W_grad_ref = W_ref.grad.clone()

        # --- Kernel backward ---
        X_test = X_ref.detach().clone()
        W_test = W_ref.detach().clone().requires_grad_(True)
        Y_test = _run_grouped_gemm(X_test, W_test, m_sizes, topk, gather_indices, backend)
        Y_test.backward(dY)

        assert W_test.grad is not None, "Gradient should flow back to W"
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(W_test.grad, W_grad_ref, rtol=rtol, atol=atol, msg="W gradient mismatch")

    # ------------------------------------------------------------------
    # Deterministic behavior
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K",
        [
            (8, 32, 256, 128),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_deterministic(self, num_experts, tokens_per_expert, N, K, backend):
        """Verify repeated runs produce identical results (no non-determinism)."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        dtype = torch.bfloat16
        total_tokens = num_experts * tokens_per_expert
        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        m_sizes = torch.full((num_experts,), tokens_per_expert, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        result1 = _run_grouped_gemm(X, W, m_sizes, 1, gather_indices, backend)
        result2 = _run_grouped_gemm(X, W, m_sizes, 1, gather_indices, backend)
        torch.testing.assert_close(result1, result2, rtol=0, atol=0, msg="Non-deterministic forward: two runs differ")

    # ------------------------------------------------------------------
    # Forward: upstream model configs (LLAMA4, QWEN3)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            # Llama4 MoE: 16 experts, hidden=5120, intermediate=8192 (~1.3GB W)
            (16, 64, 8192, 5120, 1),
            # Qwen3 MoE: 128 experts, hidden=2048, intermediate=768, topk=8
            (128, 8, 768, 2048, 8),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward_upstream_model(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test forward with shapes matching upstream LLAMA4/QWEN3 model configs."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        result = _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend)
        expected = self.reference_forward(X, W, m_sizes, topk)
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    # ------------------------------------------------------------------
    # Forward + backward correctness: topk=4
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            (4, 16, 128, 128, 4),
            (8, 8, 256, 128, 4),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward_topk4(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test grouped GEMM forward with topk=4 (unsloth coverage)."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        X = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        result = _run_grouped_gemm(X, W, m_sizes, topk, gather_indices, backend)
        expected = self.reference_forward(X, W, m_sizes, topk)
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            (4, 16, 128, 128, 4),
            (8, 8, 256, 128, 4),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_X_grad_topk4(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test backward X gradient with topk=4."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
        dY = torch.randn(total_tokens, N, dtype=dtype, device=DEVICE) * 0.1

        X_ref = (torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1).requires_grad_(True)
        W_ref = torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1
        Y_ref = self.reference_forward(X_ref, W_ref, m_sizes, topk)
        Y_ref.backward(dY)
        X_grad_ref = X_ref.grad.clone()

        X_test = X_ref.detach().clone().requires_grad_(True)
        W_test = W_ref.detach().clone()
        Y_test = _run_grouped_gemm(X_test, W_test, m_sizes, topk, gather_indices, backend)
        Y_test.backward(dY)

        assert X_test.grad is not None, "Gradient should flow back to X"
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(X_test.grad, X_grad_ref, rtol=rtol, atol=atol, msg="X gradient mismatch (topk=4)")

    @pytest.mark.parametrize(
        "num_experts, tokens_per_expert, N, K, topk",
        [
            (4, 16, 128, 128, 4),
            (8, 8, 256, 128, 4),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_W_grad_topk4(self, num_experts, tokens_per_expert, N, K, topk, dtype, backend):
        """Test backward W gradient with topk=4."""
        _skip_if_unavailable(backend)

        torch.manual_seed(42)
        total_tokens = num_experts * tokens_per_expert * topk
        m_sizes = torch.full((num_experts,), tokens_per_expert * topk, dtype=torch.int32, device=DEVICE)
        gather_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
        dY = torch.randn(total_tokens, N, dtype=dtype, device=DEVICE) * 0.1

        X_ref = torch.randn(total_tokens, K, dtype=dtype, device=DEVICE) * 0.1
        W_ref = (torch.randn(num_experts, N, K, dtype=dtype, device=DEVICE) * 0.1).requires_grad_(True)
        Y_ref = self.reference_forward(X_ref, W_ref, m_sizes, topk)
        Y_ref.backward(dY)
        W_grad_ref = W_ref.grad.clone()

        X_test = X_ref.detach().clone()
        W_test = W_ref.detach().clone().requires_grad_(True)
        Y_test = _run_grouped_gemm(X_test, W_test, m_sizes, topk, gather_indices, backend)
        Y_test.backward(dY)

        assert W_test.grad is not None, "Gradient should flow back to W"
        atol, rtol = TOLERANCE[dtype]
        torch.testing.assert_close(W_test.grad, W_grad_ref, rtol=rtol, atol=atol, msg="W gradient mismatch (topk=4)")
