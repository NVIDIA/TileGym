# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_mamba2_fused_chunk as fused_chunk_mamba2

_BACKENDS = ["cutile"]


def naive_mamba2(q, k, v, alpha, scale=None):
    """Mamba-2 reference. q, k, v: [B, T, H, K/K/V]; alpha: [B, T, H]."""
    orig_type = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v = q.float(), k.float(), v.float()
    B, T, H, K = q.shape
    Vd = v.shape[-1]
    S = torch.zeros(B, H, K, Vd, device=q.device, dtype=torch.float32)
    o = torch.empty(B, T, H, Vd, device=q.device, dtype=torch.float32)
    for t in range(T):
        S = alpha[:, t].float()[..., None, None] * S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        o[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale)
    return o.to(orig_type)


def naive_mamba2_state(q, k, v, alpha, scale=None, initial_state=None):
    """Independent Mamba-2 reference that also threads an explicit initial state
    and returns the final recurrent state.

    q, k, v: [B, T, H, K/K/V]; alpha: [B, T, H]; initial_state: [B, H, K, V].
    Returns (o [B, T, H, V], final_state [B, H, K, V]).
    """
    orig_type = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v = q.float(), k.float(), v.float()
    B, T, H, K = q.shape
    Vd = v.shape[-1]
    if initial_state is None:
        S = torch.zeros(B, H, K, Vd, device=q.device, dtype=torch.float32)
    else:
        S = initial_state.float().clone()
    o = torch.empty(B, T, H, Vd, device=q.device, dtype=torch.float32)
    for t in range(T):
        S = alpha[:, t].float()[..., None, None] * S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        o[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale)
    return o.to(orig_type), S


def make_inputs(B, T, H, K, V, dtype, device="cuda"):
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    alpha = 0.85 + 0.15 * torch.rand(B, T, H, dtype=torch.float32, device=device)
    g = alpha.log().to(torch.float32)
    return q, k, v, alpha, g


# Forward covers multiple key/value widths.
SHAPES_FWD = [
    (1, 128, 2, 64, 32),
    (1, 128, 2, 128, 128),
    (1, 64, 2, 32, 32),
    (2, 128, 2, 32, 48),
    (2, 256, 4, 32, 64),
    (1, 128, 8, 32, 32),
]
SHAPES_BWD = [
    (1, 64, 2, 32, 32),
    (1, 128, 8, 32, 32),
    (2, 256, 4, 32, 32),
]


def _assert_close_rms(test, ref, ratio, name=""):
    test32, ref32 = test.float(), ref.float()
    rms_err = (test32 - ref32).square().mean().sqrt()
    rms_ref = ref32.square().mean().sqrt()
    err_ratio = (rms_err / (rms_ref + 1e-8)).item()
    assert err_ratio < ratio, f"{name} RMS err ratio {err_ratio:.3e} exceeds {ratio:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionMamba2FusedChunk(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable). The module-level
        ``fused_chunk_mamba2`` dispatch op then routes to the active backend."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES_FWD)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op(self, B, T, H, K, V, dtype, backend, monkeypatch):
        monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "1")
        self._prep(backend)
        ratio = 1e-2 if dtype == torch.float32 else 3e-2

        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5
        o, _ = fused_chunk_mamba2(q, k, v, g, scale=scale)
        ref = naive_mamba2(q, k, v, alpha, scale=scale)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES_BWD)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward(self, B, T, H, K, V, backend):
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5

        def make_leaves():
            return tuple(x.clone().detach().requires_grad_(True) for x in (q, k, v))

        qt, kt, vt = make_leaves()
        ot, _ = fused_chunk_mamba2(qt, kt, vt, g, scale=scale)
        grad = torch.randn_like(ot)
        ot.backward(grad)

        qr, kr, vr = make_leaves()
        orr = naive_mamba2(qr, kr, vr, alpha, scale=scale)
        orr.backward(grad)

        for name, gt, gr in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad), ("dv", vt.grad, vr.grad)):
            _assert_close_rms(gt, gr, 5e-2, name=name)

    @pytest.mark.parametrize(
        "B,T,H,K,V",
        [(1, 128, 2, 64, 32), (1, 128, 2, 64, 64), (2, 160, 2, 64, 96), (1, 128, 2, 128, 128), (1, 128, 2, 256, 256)],
    )
    @pytest.mark.parametrize(
        "entry", ["linear_attention_mamba2_fused_chunk", "linear_attention_mamba2_fused_recurrent"]
    )
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op_bf16_chunk_route_backward(self, B, T, H, K, V, entry, backend):
        """Dense BF16 chunk routes preserve gradients across chunk boundaries."""
        from tilegym.suites import fla

        self._prep(backend)
        torch.manual_seed(123)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.bfloat16)
        actual_inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        reference_inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        actual, _ = getattr(fla, entry)(*actual_inputs, g)
        reference = naive_mamba2(*reference_inputs, alpha)
        _assert_close_rms(actual, reference, 3e-2, name="o")
        grad = torch.randn_like(actual)
        actual.backward(grad)
        reference.backward(grad)
        for name, value, expected in zip(("dq", "dk", "dv"), actual_inputs, reference_inputs):
            _assert_close_rms(value.grad, expected.grad, 3e-2, name=name)

    # Sequence lengths whose chunk_size = min(64, max(16, next_pow2(T))) does NOT
    # evenly divide T, so the last chunk is partial and the tail mask `m_t = o_t < T`
    # is actually exercised. T=80 -> BT=64 -> chunks {64, 16}; T=96 -> {64, 32};
    # T=160 -> {64, 64, 32}; T=48 -> BT=64 -> single partial chunk of 48.
    SHAPES_TAIL = [
        (1, 48, 2, 32, 32),
        (2, 80, 2, 32, 32),
        (1, 96, 4, 32, 48),
        (2, 160, 2, 32, 64),
    ]

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES_TAIL)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_partial_tail_chunk(self, B, T, H, K, V, dtype, backend):
        """Forward correctness when T is not a multiple of the chunk size."""
        self._prep(backend)
        ratio = 1e-2 if dtype == torch.float32 else 3e-2

        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5
        o, _ = fused_chunk_mamba2(q, k, v, g, scale=scale)
        ref = naive_mamba2(q, k, v, alpha, scale=scale)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(1, 64, 2, 32, 32), (2, 128, 4, 32, 64)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_default_scale(self, B, T, H, K, V, backend):
        """Forward with scale=None exercises the internal default scale path."""
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        o, _ = fused_chunk_mamba2(q, k, v, g, scale=None)
        ref = naive_mamba2(q, k, v, alpha, scale=None)
        _assert_close_rms(o, ref, 1e-2, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(1, 64, 2, 32, 32), (2, 128, 4, 32, 64)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_state(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        ratio = 1e-2 if dtype == torch.float32 else 3e-2

        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
        o, _ = fused_chunk_mamba2(q, k, v, g, scale=scale, initial_state=h0)
        ref, _ = naive_mamba2_state(q, k, v, alpha, scale=scale, initial_state=h0)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(1, 64, 2, 32, 32), (2, 80, 2, 32, 32), (2, 128, 4, 32, 64)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_output_final_state(self, B, T, H, K, V, backend):
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5
        o, final_state = fused_chunk_mamba2(q, k, v, g, scale=scale, output_final_state=True)
        ref_o, ref_state = naive_mamba2_state(q, k, v, alpha, scale=scale)
        _assert_close_rms(o, ref_o, 1e-2, name="o")
        assert final_state is not None
        assert tuple(final_state.shape) == (B, H, K, V)
        _assert_close_rms(final_state, ref_state, 1e-2, name="final_state")

    @pytest.mark.parametrize("B, T, H, K, V", [(1, 64, 2, 32, 32), (2, 128, 4, 32, 32)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_and_final_state(self, B, T, H, K, V, backend):
        """Forward with both initial_state set and output_final_state requested."""
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
        o, final_state = fused_chunk_mamba2(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )
        ref_o, ref_state = naive_mamba2_state(q, k, v, alpha, scale=scale, initial_state=h0)
        _assert_close_rms(o, ref_o, 1e-2, name="o")
        assert final_state is not None
        assert tuple(final_state.shape) == (B, H, K, V)
        _assert_close_rms(final_state, ref_state, 1e-2, name="final_state")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES_BWD)
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_lowprec(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5

        def make_leaves(src):
            return tuple(x.clone().detach().requires_grad_(True) for x in src)

        qt, kt, vt = make_leaves((q, k, v))
        ot, _ = fused_chunk_mamba2(qt, kt, vt, g, scale=scale)
        grad = torch.randn_like(ot)
        ot.backward(grad)

        # fp32 autograd reference (independent of the kernel) for stable gradients.
        qr, kr, vr = make_leaves((q.float(), k.float(), v.float()))
        orr = naive_mamba2(qr, kr, vr, alpha, scale=scale)
        orr.backward(grad.float())

        for name, gt, gr in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad), ("dv", vt.grad, vr.grad)):
            _assert_close_rms(gt, gr, 1e-1, name=name)

    PERF_SHAPES = [
        (1, 8192, 16, 32, 32),
        (2, 4096, 8, 32, 32),
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
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5

        training = True
        q = q.detach().requires_grad_(training)
        k = k.detach().requires_grad_(training)
        v = v.detach().requires_grad_(training)

        framework_fn = lambda g=g, k=k, q=q, v=v: fused_chunk_mamba2(q, k, v, g, scale=scale)[0]

        # Check numerical output before timing.
        with torch.no_grad():
            o_chk = framework_fn()
            o_ref = naive_mamba2(q, k, v, alpha, scale=scale)
        _assert_close_rms(o_chk, o_ref, 1e-1, name="o")
        assert torch.isfinite(o_chk).all(), (
            f"fused_chunk_mamba2 ({B}, {T}, {H}, {K}, {V}) produced non-finite output "
            f"before perf; perf number would be meaningless"
        )
        del o_chk, o_ref

        # Reset gradients between timing samples and verify they are finite.
        if training:
            _o = framework_fn()
            _o.backward(torch.randn_like(_o))
            for _n, _t in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
                assert torch.isfinite(_t).all(), (
                    f"{framework} {_n} non-finite before perf; perf number would be meaningless"
                )
            for _leaf in (q, k, v):
                _leaf.grad = None
            del _o

        result = common.benchmark_framework(framework, framework_fn, grad_to_none=[q, k, v] if training else None)
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {H}, {K}, {V})")
        record_property("op", "fused_chunk_mamba2")

        del q, k, v, alpha, g, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
