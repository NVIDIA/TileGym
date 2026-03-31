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


def _generate_indices(B, S, H_kv, topk, S_kv, device):
    """Generate valid top-k indices for testing.

    All topk slots filled with valid indices in [0, S_kv).
    Uses unique causal past indices (0..s) first; if s + 1 < topk,
    remaining slots are filled with future indices (s+1..S_kv-1).
    Future fillers are masked out by causal masking and don't affect output.
    """
    assert topk <= S_kv
    indices = torch.zeros(B, S, H_kv, topk, dtype=torch.int32, device=device)
    for b in range(B):
        for s in range(S):
            for g in range(H_kv):
                past_n = min(topk, s + 1)
                past = torch.randperm(s + 1, device=device)[:past_n].to(torch.int32)
                if past_n < topk:
                    future_n = topk - past_n
                    future_base = s + 1
                    future = (torch.randperm(S_kv - future_base, device=device)[:future_n] + future_base).to(
                        torch.int32
                    )
                    idx = torch.cat([past, future])
                else:
                    idx = past
                # Shuffle so future fillers aren't always at the end
                indices[b, s, g, :] = idx[torch.randperm(topk, device=device)]
    return indices


class Test_SparseMLA(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, indices, qpe, kpe, is_causal=True, scaling=None):
        """Pure PyTorch reference for sparse MLA forward."""
        qkv_dtype = v.dtype
        B, H, S, D = q.shape
        _, H_kv, S_kv, _ = k.shape
        _, _, _, topk = indices.shape
        D_PE = qpe.shape[-1]

        if scaling is None:
            scaling = 1.0 / math.sqrt(D + D_PE)

        # Upcast to float for reference accuracy
        q = q.float()
        k = k.float()
        v = v.float()
        qpe = qpe.float()
        kpe = kpe.float()

        # Handle GQA: expand K/V/KPE to match query head count
        if H != H_kv:
            assert H % H_kv == 0
            group_size = H // H_kv
            k = k.unsqueeze(2).expand(B, H_kv, group_size, S_kv, D).reshape(B, H, S_kv, D)
            v = v.unsqueeze(2).expand(B, H_kv, group_size, S_kv, D).reshape(B, H, S_kv, D)
            kpe_expanded = kpe.expand(B, H, S_kv, D_PE)
            # Expand indices: [B, S, H_kv, topk] -> [B, S, H, topk]
            indices = indices.unsqueeze(3).expand(B, S, H_kv, group_size, topk).reshape(B, S, H, topk)
        else:
            kpe_expanded = kpe.expand(B, H, S_kv, D_PE)

        # Gather K, V, KPE at indexed positions
        idx = indices.long().permute(0, 2, 1, 3)  # [B, H, S, topk]

        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B, H, S, topk, D]
        gathered_k = torch.gather(k, 2, idx_expanded.reshape(B, H, -1, D)).reshape(B, H, S, topk, D)
        gathered_v = torch.gather(v, 2, idx_expanded.reshape(B, H, -1, D)).reshape(B, H, S, topk, D)

        idx_pe = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D_PE)
        gathered_kpe = torch.gather(kpe_expanded, 2, idx_pe.reshape(B, H, -1, D_PE)).reshape(B, H, S, topk, D_PE)

        # score = q @ gathered_k^T + qpe @ gathered_kpe^T
        score = torch.einsum("bhsd,bhstd->bhst", q, gathered_k)
        score += torch.einsum("bhsd,bhstd->bhst", qpe, gathered_kpe)
        score *= scaling

        # Causal mask: mask out indices where idx > query position s
        if is_causal:
            s_positions = torch.arange(S, device=q.device).view(1, 1, S, 1)
            idx_for_mask = idx  # [B, H, S, topk]
            causal_mask = idx_for_mask <= s_positions
            score = score.masked_fill(~causal_mask, float("-inf"))

        # Softmax over topk dimension
        attn = torch.softmax(score, dim=-1)

        # Output: attn @ gathered_v (both float32)
        out = torch.einsum("bhst,bhstd->bhsd", attn, gathered_v)
        return out.to(qkv_dtype)

    def _run_test(self, B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()

        device = torch.device("cuda")
        q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        k = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        v = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        qpe = torch.empty(B, H, S, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        kpe = torch.empty(B, 1, S_kv, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        indices = _generate_indices(B, S, H_kv, topk, S_kv, device)

        scaling = 1.0 / math.sqrt(D + D_PE)

        def test_fn(q, k, v, indices, qpe, kpe, is_causal, scaling):
            return tilegym.ops.sparse_mla(
                q,
                k,
                v,
                indices,
                qpe,
                kpe,
                is_causal=is_causal,
                scaling=scaling,
            )

        self.assertCorrectness(
            test_fn,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "indices": indices,
                "qpe": qpe,
                "kpe": kpe,
                "is_causal": True,
                "scaling": scaling,
            },
            rtol=1e-2,
            atol=1e-2,
        )

    # ---- Core test 1: basic shapes, kv_group=1 ----
    @pytest.mark.parametrize(
        "B, H, S, S_kv, D, D_PE, H_kv, topk",
        [
            (1, 16, 64, 128, 128, 64, 1, 64),  # small basic
            (2, 32, 128, 256, 128, 64, 1, 128),  # B > 1
            (1, 64, 128, 256, 128, 64, 1, 256),  # larger H, larger topk
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_basic(self, B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch):
        self._run_test(B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch)

    # ---- Core test 2: GQA with kv_group > 1 ----
    @pytest.mark.parametrize(
        "B, H, S, S_kv, D, D_PE, H_kv, topk",
        [
            (1, 16, 64, 128, 128, 64, 4, 64),  # group_size=4
            (1, 32, 64, 128, 128, 64, 8, 64),  # group_size=4
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_gqa(self, B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch):
        """GQA: multiple query heads share one KV head."""
        self._run_test(B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch)

    # ---- Core test 3: topk == S_kv (should approximate dense MLA) ----
    @pytest.mark.parametrize(
        "B, H, S_kv, D, D_PE, H_kv",
        [
            (1, 16, 64, 128, 64, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_topk_equals_skv(self, B, H, S_kv, D, D_PE, H_kv, dtype, backend, arch):
        """When topk == S_kv and all indices present, should match dense MLA."""
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()

        device = torch.device("cuda")
        S = S_kv
        topk = S_kv  # 64 — satisfies topk % TILE_N == 0

        q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        k = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        v = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        qpe = torch.empty(B, H, S, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        kpe = torch.empty(B, 1, S_kv, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

        # All positions selected
        indices = (
            torch.arange(S_kv, device=device).view(1, 1, 1, S_kv).expand(B, S, H_kv, S_kv).to(torch.int32).contiguous()
        )

        scaling = 1.0 / math.sqrt(D + D_PE)

        def test_fn(q, k, v, indices, qpe, kpe, is_causal, scaling):
            return tilegym.ops.sparse_mla(
                q,
                k,
                v,
                indices,
                qpe,
                kpe,
                is_causal=is_causal,
                scaling=scaling,
            )

        self.assertCorrectness(
            test_fn,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "indices": indices,
                "qpe": qpe,
                "kpe": kpe,
                "is_causal": True,
                "scaling": scaling,
            },
            rtol=1e-2,
            atol=1e-2,
        )

    # ---- Core test 4: irregular shapes ----
    @pytest.mark.parametrize(
        "B, H, S, S_kv, D, D_PE, H_kv, topk",
        [
            (1, 16, 33, 77, 128, 64, 1, 64),  # non-power-of-2 S, S_kv
            (1, 16, 1, 128, 128, 64, 1, 64),  # single query position
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_irregular_shapes(self, B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch):
        """Non-power-of-2 and edge case shapes."""
        self._run_test(B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, backend, arch)

    # ---- Core test 5: forced (TILE_H, TILE_N) configs via kernel_configs ----
    def _run_test_with_config(self, B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, tile_h, tile_n, backend, arch):
        """Correctness test with a forced complete (TILE_H, TILE_N) config."""
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()

        device = torch.device("cuda")
        q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        k = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        v = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        qpe = torch.empty(B, H, S, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        kpe = torch.empty(B, 1, S_kv, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
        indices = _generate_indices(B, S, H_kv, topk, S_kv, device)

        scaling = 1.0 / math.sqrt(D + D_PE)

        def test_fn(q, k, v, indices, qpe, kpe, is_causal, scaling):
            return tilegym.ops.sparse_mla(
                q,
                k,
                v,
                indices,
                qpe,
                kpe,
                is_causal=is_causal,
                scaling=scaling,
                kernel_configs={"TILE_H": tile_h, "TILE_N": tile_n},
            )

        self.assertCorrectness(
            test_fn,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "indices": indices,
                "qpe": qpe,
                "kpe": kpe,
                "is_causal": True,
                "scaling": scaling,
            },
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "H, H_kv, tile_h, tile_n, desc",
        [
            (16, 16, 1, 64, "1:1 mapping fallback"),
            (16, 1, 4, 64, "small TILE_H"),
            (16, 4, 4, 64, "GQA group_size=4"),
            (64, 1, 16, 64, "large TILE_H"),
            (16, 1, 16, 32, "TILE_H with small TILE_N"),
            (24, 8, 1, 64, "non-pow2 group_size=3 fallback"),
            (24, 4, 2, 64, "non-pow2 group_size=6, TILE_H=2"),
            (18, 6, 1, 64, "odd prime group_size=3 fallback"),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_tile_h_configs(self, H, H_kv, tile_h, tile_n, desc, dtype, backend, arch):
        """Forced-config correctness tests for TILE_H multi-head-per-block."""
        B, S, S_kv, D, D_PE, topk = 1, 16, 128, 128, 64, 64
        self._run_test_with_config(B, H, S, S_kv, D, D_PE, H_kv, topk, dtype, tile_h, tile_n, backend, arch)

    # ---- Core test 6: invalid kernel_configs rejection ----
    @pytest.mark.parametrize("backend", _backends)
    def test_invalid_kernel_configs(self, backend, arch):
        """Partial or invalid kernel_configs must raise immediately."""
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        device = torch.device("cuda")
        dtype = torch.bfloat16
        B, H, S, S_kv, D, D_PE, H_kv, topk = 1, 16, 8, 64, 128, 64, 4, 64

        q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(std=0.3)
        k = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(std=0.3)
        v = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(std=0.3)
        qpe = torch.empty(B, H, S, D_PE, device=device, dtype=dtype).normal_(std=0.3)
        kpe = torch.empty(B, 1, S_kv, D_PE, device=device, dtype=dtype).normal_(std=0.3)
        indices = _generate_indices(B, S, H_kv, topk, S_kv, device)
        scaling = 1.0 / math.sqrt(D + D_PE)

        def call_with_config(cfg):
            return tilegym.ops.sparse_mla(
                q,
                k,
                v,
                indices,
                qpe,
                kpe,
                is_causal=True,
                scaling=scaling,
                kernel_configs=cfg,
            )

        # Partial configs — missing required keys
        with pytest.raises((ValueError, KeyError)):
            call_with_config({"TILE_H": 4})
        with pytest.raises((ValueError, KeyError)):
            call_with_config({"TILE_N": 64})

        # Invalid values — not power of 2
        with pytest.raises(AssertionError):
            call_with_config({"TILE_H": 3, "TILE_N": 64})
        with pytest.raises(AssertionError):
            call_with_config({"TILE_H": 4, "TILE_N": 48})

        # Constraint violation — TILE_H > query_group_size
        # H=16, H_kv=4 → query_group_size=4, so TILE_H=8 is too large
        with pytest.raises(AssertionError):
            call_with_config({"TILE_H": 8, "TILE_N": 64})

        # Constraint violation — group-straddling
        # Use H=24, H_kv=8 → query_group_size=3, TILE_H=2 doesn't divide 3
        q24 = torch.empty(B, 24, S, D, device=device, dtype=dtype).normal_(std=0.3)
        qpe24 = torch.empty(B, 24, S, D_PE, device=device, dtype=dtype).normal_(std=0.3)
        k8 = torch.empty(B, 8, S_kv, D, device=device, dtype=dtype).normal_(std=0.3)
        v8 = torch.empty(B, 8, S_kv, D, device=device, dtype=dtype).normal_(std=0.3)
        indices8 = _generate_indices(B, S, 8, topk, S_kv, device)
        with pytest.raises(AssertionError):
            tilegym.ops.sparse_mla(
                q24,
                k8,
                v8,
                indices8,
                qpe24,
                kpe,
                is_causal=True,
                scaling=scaling,
                kernel_configs={"TILE_H": 2, "TILE_N": 64},
            )
