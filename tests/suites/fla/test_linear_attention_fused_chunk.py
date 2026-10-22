# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test for fused_chunk_linear_attn - ported (self-contained) from
flash-linear-attention (fla.ops.linear_attn.fused_chunk).

Reference is a torch-only naive recurrence (no fla, no einops), based on
fla.ops.linear_attn.naive.naive_recurrent_linear_attn.
"""

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_fused_chunk as fused_chunk_linear_attn

_BACKENDS = ["cutile"]


def _feature_map(x: torch.Tensor) -> torch.Tensor:
    """elu(x) + 1: a standard positive feature map for linear attention.

    Used for the normalize=True path so the running denominator <q, cumsum(k)>
    stays bounded away from zero (random signed q/k can produce a near-zero
    denominator, which amplifies tiny matmul differences arbitrarily).
    """
    return F.elu(x) + 1.0


def _assert_close_rms(test: torch.Tensor, ref: torch.Tensor, ratio: float, name: str = ""):
    """Compare tensors by RMS-error ratio for recurrence gradients."""
    test32, ref32 = test.float(), ref.float()
    rms_err = (test32 - ref32).square().mean().sqrt()
    rms_ref = ref32.square().mean().sqrt()
    err_ratio = (rms_err / (rms_ref + 1e-8)).item()
    assert err_ratio < ratio, f"{name} RMS err ratio {err_ratio:.3e} exceeds {ratio:.3e}"


def naive_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Torch-only reference. Inputs [B, T, H, K], [B, T, H, K], [B, T, H, V]."""
    dtype = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    q, k, v = (x.to(torch.float32) for x in (q, k, v))
    o = torch.empty_like(v)

    S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
    for t in range(T):
        S = S + torch.einsum("b h k, b h v -> b h k v", k[:, t], v[:, t])
        o[:, t] = torch.einsum("b h k v, b h k -> b h v", S, q[:, t] * scale)
    if normalize:
        k_cum = k.cumsum(1)
        o = o / ((q * scale * k_cum).sum(-1, keepdim=True) + 1e-10)
    return o.to(dtype)


def naive_recurrent_linear_attn_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    normalize: bool = False,
    initial_state: torch.Tensor | tuple | None = None,
    g: torch.Tensor | None = None,
):
    """Torch-only reference exposing the recurrent state, mirroring the kernel.

    Returns ``(o, final_kv_state, final_z_state)`` where ``final_kv_state`` is the
    KV recurrence ``S = sum_t k_t (x) v_t`` (shape ``[B, H, K, V]``) and, when
    ``normalize`` is True, ``final_z_state`` is the running key-cumsum at the last
    timestep (shape ``[B, 1, H, K]``). ``initial_state`` follows the public API:
    a plain KV tensor ``[B, H, K, V]`` when ``normalize=False`` or the tuple
    ``(kv_state, z_state)`` when ``normalize=True`` (z_state shape ``[B, 1, H, K]``).
    An optional per-token log-decay ``g`` (``[B, T, H]``) turns the recurrence into
    the gated (simple-GLA) form ``S_t = exp(g_t) * S_{t-1} + k_t (x) v_t``.

    This independent reference is needed because the in-file
    ``naive_recurrent_linear_attn`` has no initial_state / final_state support, so
    the USE_INITIAL_STATE / STORE_FINAL_STATE kernel branches cannot otherwise be
    validated at arbitrary shapes.
    """
    dtype = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    q, k, v = (x.to(torch.float32) for x in (q, k, v))
    o = torch.empty_like(v)

    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None

    S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
    if kv_init is not None:
        S = S + kv_init.to(torch.float32)
    for t in range(T):
        if g is not None:
            S = S * g[:, t].float().exp()[..., None, None]
        S = S + torch.einsum("b h k, b h v -> b h k v", k[:, t], v[:, t])
        o[:, t] = torch.einsum("b h k v, b h k -> b h v", S, q[:, t] * scale)

    final_kv = S
    final_z = None
    if normalize:
        k_cum = k.cumsum(1)
        if z_init is not None:
            # z_init has shape [B, 1, H, K]; broadcast-add along time.
            k_cum = k_cum + z_init.to(torch.float32)
        o = o / ((q * scale * k_cum).sum(-1, keepdim=True) + 1e-10)
        final_z = k_cum[:, -1:].contiguous()
    return o.to(dtype), final_kv, final_z


# T must be a multiple of the internal chunk size (min(64, next_pow2(T))).
# For T in {64,128,256} chunk size is 64, and T is a multiple of 64.
SHAPES = [
    # (B, T, H, K, V)
    (1, 64, 1, 32, 32),
    (2, 128, 2, 48, 48),
    (1, 256, 2, 64, 64),
    (2, 128, 1, 32, 64),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionFusedChunk(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable). The module-level
        ``fused_chunk_linear_attn`` dispatch op then routes to the active backend."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    # bfloat16 only exercises normalize=True: the unnormalized recurrence
    # accumulates large bf16 rounding error (intrinsic to bf16, not the kernel).
    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize(
        "dtype, normalize",
        [
            (torch.float32, True),
            (torch.float32, False),
            (torch.bfloat16, True),
        ],
    )
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op(self, B, T, H, K, V, dtype, normalize, backend):
        self._prep(backend)

        if dtype == torch.float32:
            atol, rtol = 2e-2, 2e-2
        else:
            atol, rtol = 2e-2, 2e-2

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        if normalize:
            # Positive feature map keeps the normalization denominator stable.
            q = _feature_map(q)
            k = _feature_map(k)

        def test_fn():
            o, _ = fused_chunk_linear_attn(q, k, v, normalize=normalize)
            return o

        def ref_fn():
            return naive_recurrent_linear_attn(q, k, v, normalize=normalize)

        self.assertCorrectness(
            test_fn,
            ref_fn,
            kwargs={},
            atol=atol,
            rtol=rtol,
            check_stride=False,
        )

    # ------------------------------------------------------------------
    # Coverage-extension tests (added to fill branch gaps in test_op).
    # ------------------------------------------------------------------

    # T values whose internal chunk size = min(64, max(16, next_pow2(T))) is
    # SMALLER than 64 and/or for which T is NOT a multiple of the chunk size.
    #   T=16 -> BT=16 (exact), T=24 -> BT=32 (24 % 32 != 0, partial last chunk),
    #   T=48 -> BT=64 (48 % 64 != 0, single partial chunk),
    #   T=80 -> BT=64 (80 % 64 != 0, full + partial chunk, NT=2).
    @pytest.mark.parametrize("T", [16, 24, 48, 80])
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_boundary(self, T, normalize, backend):
        """Check causal output and recurrent state for a partial final chunk."""
        self._prep(backend)
        atol, rtol = 2e-2, 2e-2
        dtype = torch.float32
        B, H, K, V = 2, 2, 32, 48
        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        if normalize:
            q = _feature_map(q)
            k = _feature_map(k)

        def test_fn():
            o, _ = fused_chunk_linear_attn(q, k, v, normalize=normalize)
            return o

        def ref_fn():
            return naive_recurrent_linear_attn(q, k, v, normalize=normalize)

        self.assertCorrectness(test_fn, ref_fn, kwargs={}, atol=atol, rtol=rtol, check_stride=False)

    # K/V drive BK = min(max(next_pow2(K),16),64) and NK = cdiv(K, BK)
    # (and BV/NV symmetrically in the backward). K or V > 64 forces NK/NV > 1,
    # which switches fused_chunk_fwd to the multi-tile accumulator path
    #   o = v.new_empty(NK, *v.shape, fp32) ... o = o.sum(0)
    # K not a power of two (e.g. 96) additionally exercises the K-dim
    # padding on the final partial feature tile.
    @pytest.mark.parametrize(
        "K, V",
        [
            (128, 64),  # NK = 2 (K split into two 64-blocks), V single tile
            (64, 128),  # NV-relevant; V split path on the value dim
            (96, 96),  # NK = 2 with a PARTIAL last K-block (96 not power of 2)
        ],
    )
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_kv_multi_tile(self, K, V, normalize, backend):
        """Check feature-tile reduction and padding when K or V spans multiple tiles."""
        self._prep(backend)
        atol, rtol = 2e-2, 2e-2
        dtype = torch.float32
        B, T, H = 2, 128, 2
        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        if normalize:
            q = _feature_map(q)
            k = _feature_map(k)

        def test_fn():
            o, _ = fused_chunk_linear_attn(q, k, v, normalize=normalize)
            return o

        def ref_fn():
            return naive_recurrent_linear_attn(q, k, v, normalize=normalize)

        self.assertCorrectness(test_fn, ref_fn, kwargs={}, atol=atol, rtol=rtol, check_stride=False)

    @pytest.mark.parametrize("B, T, H, K, V", [(1, 64, 1, 32, 32), (2, 128, 2, 48, 64)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_and_final_state_unnormalized(self, B, T, H, K, V, backend):
        """USE_INITIAL_STATE + STORE_FINAL_STATE (normalize=False) paths."""
        self._prep(backend)
        atol, rtol = 2e-2, 2e-2
        dtype = torch.float32
        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        h0 = torch.randn(B, H, K, V, dtype=dtype, device="cuda")

        o, final_state = fused_chunk_linear_attn(
            q,
            k,
            v,
            normalize=False,
            initial_state=h0,
            output_final_state=True,
        )
        ref_o, ref_kv, _ = naive_recurrent_linear_attn_state(
            q,
            k,
            v,
            normalize=False,
            initial_state=h0,
        )

        assert o.shape == (B, T, H, V)
        assert final_state is not None and final_state.shape == (B, H, K, V)
        torch.testing.assert_close(o, ref_o, atol=atol, rtol=rtol)
        torch.testing.assert_close(final_state.to(torch.float32), ref_kv, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("B, T, H, K, V", [(1, 64, 1, 32, 32), (2, 128, 2, 48, 64)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_and_final_state_normalized(self, B, T, H, K, V, backend):
        """USE_INITIAL_STATE + final state on the normalize=True tuple path."""
        self._prep(backend)
        atol, rtol = 3e-2, 3e-2
        dtype = torch.float32
        q = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        k = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        kv0 = torch.randn(B, H, K, V, dtype=dtype, device="cuda")
        # z_state has shape [B, 1, H, K] (one running key-cumsum per (b,h)).
        z0 = _feature_map(torch.randn(B, 1, H, K, dtype=dtype, device="cuda"))

        o, final_state = fused_chunk_linear_attn(
            q,
            k,
            v,
            normalize=True,
            initial_state=(kv0, z0),
            output_final_state=True,
        )
        ref_o, ref_kv, ref_z = naive_recurrent_linear_attn_state(
            q,
            k,
            v,
            normalize=True,
            initial_state=(kv0, z0),
        )

        assert o.shape == (B, T, H, V)
        assert isinstance(final_state, tuple) and len(final_state) == 2
        kv_final, z_final = final_state
        assert kv_final.shape == (B, H, K, V)
        assert z_final.shape == (B, 1, H, K)
        torch.testing.assert_close(o, ref_o, atol=atol, rtol=rtol)
        torch.testing.assert_close(kv_final.to(torch.float32), ref_kv, atol=atol, rtol=rtol)
        torch.testing.assert_close(z_final.to(torch.float32), ref_z, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("scale", [1.0, 0.1, 4.0])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_explicit_scale(self, scale, backend):
        """Non-default ``scale`` argument propagated through fwd + normalize."""
        self._prep(backend)
        atol, rtol = 2e-2, 2e-2
        dtype = torch.float32
        B, T, H, K, V = 2, 128, 2, 32, 48
        q = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        k = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")

        def test_fn():
            o, _ = fused_chunk_linear_attn(q, k, v, scale=scale, normalize=True)
            return o

        def ref_fn():
            return naive_recurrent_linear_attn(q, k, v, scale=scale, normalize=True)

        self.assertCorrectness(test_fn, ref_fn, kwargs={}, atol=atol, rtol=rtol, check_stride=False)

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_identical_rows(self, normalize, backend):
        """Degenerate input: every timestep identical across T."""
        self._prep(backend)
        atol, rtol = 2e-2, 2e-2
        dtype = torch.float32
        B, T, H, K, V = 1, 128, 1, 32, 32
        row_q = _feature_map(torch.randn(B, 1, H, K, dtype=dtype, device="cuda"))
        row_k = _feature_map(torch.randn(B, 1, H, K, dtype=dtype, device="cuda"))
        row_v = torch.randn(B, 1, H, V, dtype=dtype, device="cuda")
        q = row_q.expand(B, T, H, K).contiguous()
        k = row_k.expand(B, T, H, K).contiguous()
        v = row_v.expand(B, T, H, V).contiguous()

        def test_fn():
            o, _ = fused_chunk_linear_attn(q, k, v, normalize=normalize)
            return o

        def ref_fn():
            return naive_recurrent_linear_attn(q, k, v, normalize=normalize)

        self.assertCorrectness(test_fn, ref_fn, kwargs={}, atol=atol, rtol=rtol, check_stride=False)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward(self, backend):
        self._prep(backend)
        out_atol, out_rtol = 2e-2, 2e-2
        grad_ratio = 5e-2
        dtype = torch.float32
        B, T, H, K, V = 2, 64, 2, 32, 32

        base_q = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        base_k = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        base_v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")

        q1 = base_q.clone().requires_grad_(True)
        k1 = base_k.clone().requires_grad_(True)
        v1 = base_v.clone().requires_grad_(True)
        o1, _ = fused_chunk_linear_attn(q1, k1, v1, normalize=True)
        go = torch.randn_like(o1)
        o1.backward(go)

        q2 = base_q.clone().requires_grad_(True)
        k2 = base_k.clone().requires_grad_(True)
        v2 = base_v.clone().requires_grad_(True)
        o2 = naive_recurrent_linear_attn(q2, k2, v2, normalize=True)
        o2.backward(go)

        torch.testing.assert_close(o1, o2, atol=out_atol, rtol=out_rtol)
        for name, test_grad, ref_grad in (
            ("dq", q1.grad, q2.grad),
            ("dk", k1.grad, k2.grad),
            ("dv", v1.grad, v2.grad),
        ):
            _assert_close_rms(test_grad, ref_grad, grad_ratio, name=name)

    @pytest.mark.parametrize("B, T, H, K, V", [(2, 80, 2, 32, 32), (1, 256, 2, 32, 32)])
    @pytest.mark.parametrize("varlen", [False, True])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_per_token_gate(self, B, T, H, K, V, varlen, backend):
        """Per-token gates preserve output, final state and input/state gradients."""
        self._prep(backend)
        if backend == "cutile":
            from tilegym.suites.fla.cutile import linear_attention_fused_chunk as _mod
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        torch.manual_seed(0)
        if varlen:
            # Pack B * T tokens into one row as three sequences of unequal length.
            total = B * T
            seq_lens = [total // 3 - 5, total // 3 + 9]
            seq_lens.append(total - sum(seq_lens))
            cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], device="cuda")
            B, T, N = 1, total, len(seq_lens)
        else:
            cu_seqlens, N = None, B
        dtype = torch.float32

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device="cuda")) / 8
        h0 = torch.randn(N, H, K, V, dtype=dtype, device="cuda")

        def make_leaves():
            return tuple(x.clone().detach().requires_grad_(True) for x in (q, k, v, g, h0))

        test_leaves = make_leaves()
        o, ht = _mod.fused_chunk_simple_gla(
            *test_leaves[:4],
            initial_state=test_leaves[4],
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        do, dht = torch.randn_like(o), torch.randn_like(ht)
        ((o * do).sum() + (ht * dht).sum()).backward()

        ref_leaves = make_leaves()
        rq, rk, rv, rg, rh0 = ref_leaves
        if varlen:
            bounds = cu_seqlens.tolist()
            outs = [
                naive_recurrent_linear_attn_state(
                    rq[:, bos:eos], rk[:, bos:eos], rv[:, bos:eos], initial_state=rh0[i : i + 1], g=rg[:, bos:eos]
                )
                for i, (bos, eos) in enumerate(zip(bounds[:-1], bounds[1:]))
            ]
            ref_o, ref_ht = torch.cat([x[0] for x in outs], 1), torch.cat([x[1] for x in outs], 0)
        else:
            ref_o, ref_ht, _ = naive_recurrent_linear_attn_state(rq, rk, rv, initial_state=rh0, g=rg)
        ((ref_o * do).sum() + (ref_ht * dht).sum()).backward()

        _assert_close_rms(o, ref_o, 1e-2, name="o")
        _assert_close_rms(ht, ref_ht, 1e-2, name="final_state")
        for name, test_leaf, ref_leaf in zip(("dq", "dk", "dv", "dg", "dh0"), test_leaves, ref_leaves):
            _assert_close_rms(test_leaf.grad, ref_leaf.grad, 5e-2, name=name)

    # (B, T, H, K, V) -- representative perf shapes (T multiple of chunk size 64).
    PERF_SHAPES = [
        # (B, T, H, K, V) -- aligned with fla upstream benchmark SHAPE_CONFIGS (K = V = D).
        (1, 8192, 96, 128, 128),  # B1_T8192_H96_D128
        (2, 16384, 16, 128, 128),  # B2_T16384_H16_D128
        (4, 2048, 16, 128, 128),  # B4_T2048_H16_D128
        (4, 4096, 64, 128, 128),  # B4_T4096_H64_D128
        (8, 2048, 32, 256, 256),  # B8_T2048_H32_D256
        (8, 1024, 8, 64, 64),  # B8_T1024_H8_D64
    ]

    @pytest.mark.parametrize("B, T, H, K, V", PERF_SHAPES)
    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf(self, B, T, H, K, V, framework, record_property):
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

        dtype = torch.bfloat16
        torch.manual_seed(0)
        q = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        k = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda"))
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the active backend matches the naive reference on
        # this exact perf shape (bf16 + normalize=True with the positive feature
        # map) BEFORE timing it. test_op validates exactly this combination at
        # atol/rtol 2e-2, so we mirror it with the file's RMS-ratio helper at 2e-2.
        # A hard assert turns a silently-broken kernel (e.g. NaN output) into a
        # loud failure instead of a fast-but-garbage benchmark number.
        o_chk, _ = fused_chunk_linear_attn(q, k, v, normalize=True)
        o_ref = naive_recurrent_linear_attn(q, k, v, normalize=True)
        ratio = (
            (o_chk.float() - o_ref.float()).square().mean().sqrt() / (o_ref.float().square().mean().sqrt() + 1e-8)
        ).item()
        assert ratio < 2e-2, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del o_chk, o_ref

        framework_fn = lambda k=k, q=q, v=v: fused_chunk_linear_attn(q, k, v, normalize=True)[0]

        result = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {H}, {K}, {V})")
        record_property("op", "fused_chunk_linear_attn")

        del q, k, v, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    @pytest.mark.parametrize("B, T, H, K, V", PERF_SHAPES)
    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf_bwd(self, B, T, H, K, V, framework, record_property):
        # Backward perf for fused_chunk_linear_attn, exercising fused_chunk_bwd /
        # fused_chunk_bwd_kernel via FusedChunkFunction.backward. The forward
        # test_perf above never triggers the backward, and no other module
        # imports this file's fused_chunk_bwd, so the bwd kernel is otherwise
        # uncovered by any perf path. Mark q/k/v grad-bearing and pass
        # grad_to_none so benchmark_framework records the backward under
        # MODE=auto (mirrors the chunk ops' perf tests). normalize=True also
        # drives the StatefulNormalize backward.
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

        dtype = torch.bfloat16
        torch.manual_seed(0)
        q = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda")).detach()
        k = _feature_map(torch.randn(B, T, H, K, dtype=dtype, device="cuda")).detach()
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the forward output matches the naive reference on
        # this exact perf shape (bf16 + normalize=True, positive feature map)
        # BEFORE timing the fwd+bwd. test_op validates this combination at 2e-2,
        # so we mirror it with the file's RMS-ratio helper at 2e-2. Run on the
        # grad-free tensors first, then enable autograd for the benchmark. A hard
        # assert turns a silently-broken kernel (e.g. NaN) into a loud failure.
        o_chk, _ = fused_chunk_linear_attn(q, k, v, normalize=True)
        o_ref = naive_recurrent_linear_attn(q, k, v, normalize=True)
        ratio = (
            (o_chk.float() - o_ref.float()).square().mean().sqrt() / (o_ref.float().square().mean().sqrt() + 1e-8)
        ).item()
        assert ratio < 2e-2, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del o_chk, o_ref

        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)

        framework_fn = lambda k=k, q=q, v=v: fused_chunk_linear_attn(q, k, v, normalize=True)[0]

        # Backward is also timed here (grad-enabled inputs + MODE=auto), so a
        # silently-broken backward would produce a meaningless number too. Run one
        # fwd+bwd on this exact perf shape and assert the grads are finite -- a
        # NaN/Inf gradient becomes a loud failure instead of a fast-but-garbage
        # backward timing. (Backward *closeness* is covered by test_backward at
        # functional shapes; autograd through the naive reference at the perf T is
        # prohibitively memory-heavy.)
        _o = framework_fn()
        _o.backward(torch.randn_like(_o))
        for _n, _t in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
            assert torch.isfinite(_t).all(), (
                f"{framework} {_n} non-finite before perf; perf number would be meaningless"
            )
        for _leaf in (q, k, v):
            _leaf.grad = None
        del _o

        result = common.benchmark_framework(framework, framework_fn, grad_to_none=[q, k, v])
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {H}, {K}, {V})")
        record_property("op", "fused_chunk_linear_attn")

        del q, k, v, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
