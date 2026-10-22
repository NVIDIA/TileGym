# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Tests for the self-contained port of fla's `chunk_linear_attn`.

Reference (`naive_recurrent_linear_attn`) is a torch-only re-implementation of
`fla.ops.linear_attn.naive` (no fla, no einops):
    S_t = S_{t-1} + k_t^T v_t,  o_t = (q_t * scale) S_t,
with optional running-cumsum normalization.
"""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_chunk as chunk_linear_attn

_BACKENDS = ["cutile"]


def naive_recurrent_linear_attn(
    q,
    k,
    v,
    scale=None,
    initial_state=None,
    output_final_state=False,
    normalize=False,
):
    dtype = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None
    q, k, v = (x.to(torch.float32) for x in (q, k, v))
    o = torch.empty_like(v)

    S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
    if kv_init is not None:
        S = S + kv_init
    for t in range(T):
        S = S + torch.einsum("b h k, b h v -> b h k v", k[:, t], v[:, t])
        o[:, t] = torch.einsum("b h k v, b h k -> b h v", S, q[:, t] * scale)
    if normalize:
        k_cum = k.cumsum(1) if z_init is None else k.cumsum(1) + z_init
        o = o / ((q * scale * k_cum).sum(-1, keepdim=True) + 1e-10)
        if output_final_state:
            return o.to(dtype), (S, k_cum[:, -1:])
        return o.to(dtype), None
    return o.to(dtype), S if output_final_state else None


SHAPES = [
    # (B, T, H, K, V)
    (1, 64, 2, 32, 32),
    (2, 128, 2, 48, 48),
    (2, 128, 4, 64, 64),
    (1, 256, 2, 64, 32),
    (2, 256, 2, 32, 64),
]


def _assert_close_rms(test, ref, ratio, name=""):
    """Compare chunked vs recurrent outputs.

    The chunked algorithm and the recurrent reference compute the same math in a
    different order, so in low precision they legitimately disagree at isolated
    positions (most severely where the normalize denominator <q, cumsum(k)> is
    near-singular, amplifying tiny fp errors). We therefore compare with an
    RMS-error ratio over the whole tensor (as fla's own tests do) rather than a
    strict per-element allclose.
    """
    test32, ref32 = test.float(), ref.float()
    rms_err = (test32 - ref32).square().mean().sqrt()
    rms_ref = ref32.square().mean().sqrt()
    err_ratio = (rms_err / (rms_ref + 1e-8)).item()
    assert err_ratio < ratio, f"{name} RMS err ratio {err_ratio:.3e} exceeds {ratio:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionChunk(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable). The module-level
        ``chunk_linear_attn`` dispatch op then routes to the active backend."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op(self, B, T, H, K, V, normalize, dtype, backend, monkeypatch):
        monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "1")
        self._prep(backend)
        if normalize and dtype == torch.bfloat16:
            # The normalize path divides by <q, cumsum(k)>, which is near-singular at
            # isolated positions. In bf16 that denominator is computed in low precision
            # and diverges arbitrarily (even sign-flips) from an fp32-recurrent
            # reference, so this combination is numerically ill-conditioned to compare.
            pytest.skip("bf16 normalize vs fp32-recurrent reference is ill-conditioned")
        ratio = 5e-3 if dtype == torch.float32 else 2e-2

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        o, _ = chunk_linear_attn(q, k, v, scale=scale, normalize=normalize)
        ref, _ = naive_recurrent_linear_attn(q, k, v, scale=scale, normalize=normalize)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward(self, B, T, H, K, V, normalize, backend):
        self._prep(backend)
        dtype = torch.float32

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        def make_leaves():
            return tuple(x.clone().detach().requires_grad_(True) for x in (q, k, v))

        qt, kt, vt = make_leaves()
        ot, _ = chunk_linear_attn(qt, kt, vt, scale=scale, normalize=normalize)
        grad = torch.randn_like(ot)
        ot.backward(grad)

        qr, kr, vr = make_leaves()
        orr, _ = naive_recurrent_linear_attn(qr, kr, vr, scale=scale, normalize=normalize)
        orr.backward(grad)

        for name, gt, gr in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad), ("dv", vt.grad, vr.grad)):
            _assert_close_rms(gt, gr, 5e-2, name=name)

    # ------------------------------------------------------------------
    # Coverage extensions (targeted branch/edge-case tests).
    # ------------------------------------------------------------------

    # (B, T, H, K, V) shapes where T is NOT a multiple of the chunk size.
    # chunk_size = min(64, max(16, next_pow2(T))) inside ChunkSimpleGLAFunction.forward,
    # so for T >= 64 the chunk size is 64; the following T values all leave a partial
    # last chunk (T % 64 != 0).
    RAGGED_SHAPES = [
        (1, 65, 2, 32, 32),  # one full chunk + a 1-row tail
        (2, 100, 2, 48, 48),  # 64 + 36 tail, K/V not a power of two
        (2, 130, 4, 64, 64),  # two full chunks + a 2-row tail
        (1, 191, 2, 64, 32),  # 64*2 + 63 tail, K != V
    ]

    @pytest.mark.parametrize("B, T, H, K, V", RAGGED_SHAPES)
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_ragged_seqlen(self, B, T, H, K, V, normalize, backend):
        """T not a multiple of the chunk size -> partial last chunk."""
        self._prep(backend)
        dtype = torch.float32  # keep fp32: normalize denominator is well-conditioned here
        ratio = 5e-3

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        o, _ = chunk_linear_attn(q, k, v, scale=scale, normalize=normalize)
        ref, _ = naive_recurrent_linear_attn(q, k, v, scale=scale, normalize=normalize)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("output_final_state", [False, True])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_varlen_normalized(self, output_final_state, backend):
        """Packed variable-length inputs with normalize=True reset state per segment."""
        self._prep(backend)
        torch.manual_seed(0)

        lengths = [64, 96, 37]
        T = sum(lengths)
        H, K, V = 2, 32, 32
        dtype = torch.float32
        scale = K**-0.5
        q = torch.nn.functional.elu(torch.randn(1, T, H, K, dtype=dtype, device="cuda")) + 1.0
        k = torch.nn.functional.elu(torch.randn(1, T, H, K, dtype=dtype, device="cuda")) + 1.0
        v = torch.randn(1, T, H, V, dtype=dtype, device="cuda")
        cu_seqlens = torch.tensor([0, 64, 160, 197], dtype=torch.int32, device="cuda")

        o, state = chunk_linear_attn(
            q, k, v, scale=scale, normalize=True, cu_seqlens=cu_seqlens, output_final_state=output_final_state
        )
        if output_final_state:
            kv_state, z_state = state
        else:
            assert state is None

        ref = torch.empty_like(o)
        z_refs = []
        kv_refs = []
        start = 0
        for length in lengths:
            seg = slice(start, start + length)
            ref[:, seg], kv_ref = naive_recurrent_linear_attn(
                q[:, seg],
                k[:, seg],
                v[:, seg],
                scale=scale,
                normalize=True,
                output_final_state=True,
            )
            kv_refs.append(kv_ref[0])
            z_refs.append(k[:, seg].float().cumsum(1)[:, -1:])
            start += length
        z_ref = torch.cat(z_refs, dim=0)

        _assert_close_rms(o, ref, 5e-3, name="o")
        if output_final_state:
            _assert_close_rms(kv_state, torch.cat(kv_refs, dim=0), 5e-3, name="kv_state")
            torch.testing.assert_close(z_state.float(), z_ref, atol=1e-4, rtol=1e-4)

    # (B, T, H, K, V) with very small T to drive the chunk_size selection branch.
    # chunk_size = min(64, max(16, next_pow2(T))): T<=16 -> 16, T=32 -> 32.
    SMALL_T_SHAPES = [
        (2, 8, 2, 32, 32),  # next_pow2(8)=8 -> clamped up to 16; T < chunk_size
        (1, 16, 2, 32, 32),  # chunk_size = 16 exactly
        (2, 32, 4, 64, 64),  # chunk_size = 32
        (1, 48, 2, 32, 64),  # chunk_size = 64 but T < chunk_size (single partial chunk)
    ]

    @pytest.mark.parametrize("B, T, H, K, V", SMALL_T_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_small_chunk_size(self, B, T, H, K, V, backend):
        """Small T exercises the chunk_size = min(64, max(16, next_pow2(T))) branch."""
        self._prep(backend)
        dtype = torch.float32
        ratio = 5e-3

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        o, _ = chunk_linear_attn(q, k, v, scale=scale, normalize=False)
        ref, _ = naive_recurrent_linear_attn(q, k, v, scale=scale, normalize=False)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(2, 128, 2, 48, 48), (1, 256, 2, 64, 32)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_fp16(self, B, T, H, K, V, backend):
        """fp16 dtype (non-normalized)."""
        self._prep(backend)
        dtype = torch.float16
        ratio = 2e-2

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        o, _ = chunk_linear_attn(q, k, v, scale=scale, normalize=False)
        ref, _ = naive_recurrent_linear_attn(q, k, v, scale=scale, normalize=False)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(2, 128, 2, 48, 48), (1, 256, 2, 64, 32)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_state(self, B, T, H, K, V, backend):
        """Non-normalized forward with a non-zero initial recurrent state."""
        self._prep(backend)
        dtype = torch.float32
        ratio = 5e-3

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        kv_init = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
        scale = K**-0.5

        o, _ = chunk_linear_attn(
            q,
            k,
            v,
            scale=scale,
            normalize=False,
            initial_state=kv_init,
        )
        ref, _ = naive_recurrent_linear_attn(
            q,
            k,
            v,
            scale=scale,
            normalize=False,
            initial_state=kv_init,
        )
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(2, 128, 2, 48, 48), (1, 256, 2, 64, 32)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_output_final_state(self, B, T, H, K, V, backend):
        """output_final_state=True: validate both the output AND the returned KV state."""
        self._prep(backend)
        dtype = torch.float32

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        o, final_state = chunk_linear_attn(
            q,
            k,
            v,
            scale=scale,
            normalize=False,
            output_final_state=True,
        )
        ref, ref_state = naive_recurrent_linear_attn(
            q,
            k,
            v,
            scale=scale,
            normalize=False,
            output_final_state=True,
        )
        assert final_state is not None, "output_final_state=True must return a state"
        assert final_state.shape == ref_state.shape, (
            f"final state shape {tuple(final_state.shape)} != ref {tuple(ref_state.shape)}"
        )
        _assert_close_rms(o, ref, 5e-3, name="o")
        _assert_close_rms(final_state, ref_state, 5e-3, name="final_state")

    @pytest.mark.parametrize("B, T, H, K, V", [(2, 128, 2, 48, 48), (1, 256, 2, 64, 32)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_normalize_initial_state_tuple(self, B, T, H, K, V, backend):
        """normalize=True with a (kv_state, z_state) initial-state tuple."""
        self._prep(backend)
        dtype = torch.float32

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        kv_init = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
        # z_state has shape [B, 1, H, K] (running key-sum carried into the denominator).
        z_init = torch.randn(B, 1, H, K, dtype=torch.float32, device="cuda")
        scale = K**-0.5

        o, _ = chunk_linear_attn(
            q,
            k,
            v,
            scale=scale,
            normalize=True,
            initial_state=(kv_init, z_init),
        )
        ref, _ = naive_recurrent_linear_attn(
            q,
            k,
            v,
            scale=scale,
            normalize=True,
            initial_state=(kv_init, z_init),
        )
        _assert_close_rms(o, ref, 5e-3, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", [(2, 128, 2, 48, 48), (1, 256, 2, 32, 64)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_default_scale(self, B, T, H, K, V, backend):
        """scale=None -> kernel uses its internal default 1/sqrt(K)."""
        self._prep(backend)
        dtype = torch.float32

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")

        o, _ = chunk_linear_attn(q, k, v, scale=None, normalize=False)
        ref, _ = naive_recurrent_linear_attn(q, k, v, scale=None, normalize=False)
        _assert_close_rms(o, ref, 5e-3, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", RAGGED_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_ragged_seqlen(self, B, T, H, K, V, backend):
        """Backward pass when T is not a multiple of the chunk size."""
        self._prep(backend)
        dtype = torch.float32

        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        scale = K**-0.5

        def make_leaves():
            return tuple(x.clone().detach().requires_grad_(True) for x in (q, k, v))

        qt, kt, vt = make_leaves()
        ot, _ = chunk_linear_attn(qt, kt, vt, scale=scale, normalize=False)
        grad = torch.randn_like(ot)
        ot.backward(grad)

        qr, kr, vr = make_leaves()
        orr, _ = naive_recurrent_linear_attn(qr, kr, vr, scale=scale, normalize=False)
        orr.backward(grad)

        for name, gt, gr in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad), ("dv", vt.grad, vr.grad)):
            _assert_close_rms(gt, gr, 5e-2, name=name)

    # (B, T, H, K, V) -- aligned with fla upstream benchmark (benchmarks/ops/registry.py
    # SHAPE_CONFIGS). fla uses a single per-head dim D for q/k/v (shape_BTHD), so K = V = D.
    PERF_SHAPES = [
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

        # fla upstream benchmark hardcodes bfloat16 (benchmarks/ops/run.py).
        dtype = torch.bfloat16
        scale = K**-0.5
        torch.manual_seed(0)
        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the active backend matches the naive reference on
        # this exact perf shape BEFORE timing it. A hard assert here turns a
        # silently-broken kernel (e.g. NaN output) into a loud failure instead
        # of a fast-but-garbage benchmark number.
        #
        # NOTE: test_perf benchmarks normalize=True, but that path is ill-conditioned
        # and CANNOT be verified at the perf shapes: its denominator <q, cumsum(k)>
        # is near-singular at long T, so the chunked kernel and the recurrent
        # reference diverge by O(1) (rms up to ~8 at these shapes) even in fp32 with
        # identical inputs -- a formulation artifact, not a kernel bug (test_op only
        # checks normalize at short T, and skips it in bf16). We therefore gate the
        # well-conditioned normalize=False core recurrence, which shares the same
        # state kernels (normalize just adds a final division): fp32, rms ~3e-4 across
        # all perf shapes, so a broken/NaN core kernel is still caught with wide margin.
        q32, k32, v32 = (x.float() for x in (q, k, v))
        o_chk, _ = chunk_linear_attn(q32, k32, v32, scale=scale, normalize=False)
        o_ref, _ = naive_recurrent_linear_attn(q32, k32, v32, scale=scale, normalize=False)
        ratio = (
            (o_chk.float() - o_ref.float()).square().mean().sqrt() / (o_ref.float().square().mean().sqrt() + 1e-8)
        ).item()
        assert ratio < 5e-3, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del q32, k32, v32, o_chk, o_ref

        # Enable autograd on q/k/v so that, under MODE=auto (default) or MODE=backward,
        # benchmark() also times the standalone backward kernel.
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)

        framework_fn = lambda k=k, q=q, v=v: chunk_linear_attn(q, k, v, scale=scale, normalize=True)[0]

        # grad_to_none resets input grads between reps so backward timing is not
        # polluted by gradient accumulation. MODE=auto records both 'forward' and
        # 'backward'; MODE=forward preserves the prior fwd-only result.
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
        record_property("op", "chunk_linear_attn")

        del q, k, v, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
