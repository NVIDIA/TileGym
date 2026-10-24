# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test for fused_recurrent_delta_rule (DeltaNet) - self-contained port from
flash-linear-attention (fla).

The reference is a torch-only naive recurrence (no fla, no einops) ported from
`fla.ops.delta_rule.naive.delta_rule_recurrence`:
    S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t = (q_t * scale) S_t
with q scaled by 1/sqrt(K). The reference operates on [B, H, T, D] layout while
the public op takes [B, T, H, D]; we transpose between the two.

The recurrence is unstable unless beta * ||k||^2 < 2, so we use small,
well-conditioned inputs.
"""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_deltanet_fused_recurrent as fused_recurrent_delta_rule

_BACKENDS = ["cutile"]


# (B, T, H, K, V)
SHAPES = [
    (2, 128, 4, 64, 64),
    (1, 256, 2, 32, 48),
]


def delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    """Torch-only DeltaNet reference; inputs are [B, H, T, D], beta is [B, H, T]."""
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    q, k, v, beta = (x.float() for x in (q, k, v, beta))
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k**-0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    if initial_state is not None:
        S = S + initial_state

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def rms_ratio(test, ref):
    test = test.float()
    ref = ref.float()
    denom = ref.pow(2).mean().sqrt()
    num = (test - ref).pow(2).mean().sqrt()
    return (num / (denom + 1e-12)).item()


def make_inputs(B, T, H, K, V, dtype):
    q = (torch.rand(B, T, H, K, device="cuda") + 0.1) * 0.5
    k = (torch.rand(B, T, H, K, device="cuda") + 0.1) * 0.5
    v = torch.randn(B, T, H, V, device="cuda")
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda")) * (0.8 / K)
    return q.to(dtype), k.to(dtype), v.to(dtype), beta.to(dtype)


class Test_FLA_LinearAttentionDeltanetFusedRecurrent(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable). The module-level
        ``fused_recurrent_delta_rule`` dispatch op then routes to the active backend."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)

        # reference operates on [B, H, T, *]
        o_ref, _ = delta_rule_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            output_final_state=False,
        )
        o_ref = o_ref.transpose(1, 2)

        o_out, _ = fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("K", [192, 256])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_large_k(self, K, dtype, backend):
        """K > 128 (NK = ceil(K/64) >= 3) must contract the full key/state dim."""
        self._prep(backend)
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        B, T, H, V = 1, 512, 4, 64
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)
        o_ref, _ = delta_rule_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            output_final_state=False,
        )
        o_ref = o_ref.transpose(1, 2)

        o_out, _ = fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward(self, B, T, H, K, V, backend):
        self._prep(backend)
        dtype = torch.float32
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)

        # --- reference grads via naive autograd ---
        qr = q.transpose(1, 2).detach().clone().requires_grad_(True)
        kr = k.transpose(1, 2).detach().clone().requires_grad_(True)
        vr = v.transpose(1, 2).detach().clone().requires_grad_(True)
        br = beta.transpose(1, 2).detach().clone()
        o_ref, _ = delta_rule_recurrence(qr, kr, vr, br, output_final_state=False)
        do = torch.randn_like(o_ref)
        o_ref.backward(do)
        dq_ref = qr.grad.transpose(1, 2)
        dk_ref = kr.grad.transpose(1, 2)
        dv_ref = vr.grad.transpose(1, 2)

        # --- kernel grads ---
        qt = q.detach().clone().requires_grad_(True)
        kt = k.detach().clone().requires_grad_(True)
        vt = v.detach().clone().requires_grad_(True)
        bt = beta.detach().clone()
        o_out, _ = fused_recurrent_delta_rule(qt, kt, vt, beta=bt, scale=scale)
        o_out.backward(do.transpose(1, 2).contiguous())

        assert rms_ratio(o_out, o_ref.transpose(1, 2)) < 1e-4
        assert rms_ratio(qt.grad, dq_ref) < 5e-2
        assert rms_ratio(kt.grad, dk_ref) < 5e-2
        assert rms_ratio(vt.grad, dv_ref) < 5e-2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_frozen_initial_state_uses_h0(self, backend):
        """Frozen initial_state must still seed the backward state reconstruction."""
        self._prep(backend)
        torch.manual_seed(6)
        B, T, H, K, V = 1, 4, 1, 16, 16
        dtype = torch.float32
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)
        h0 = (torch.randn(B, H, K, V, device="cuda") * 0.05).to(torch.float32)
        do = torch.randn(B, T, H, V, device="cuda", dtype=dtype)

        def run(initial_state):
            qt = q.detach().clone().requires_grad_(True)
            kt = k.detach().clone().requires_grad_(True)
            vt = v.detach().clone().requires_grad_(True)
            bt = beta.detach().clone()
            out, _ = fused_recurrent_delta_rule(qt, kt, vt, beta=bt, scale=scale, initial_state=initial_state)
            out.backward(do)
            return out.detach(), qt.grad.detach(), kt.grad.detach(), vt.grad.detach()

        h0_trainable = h0.detach().clone().requires_grad_(True)
        out_trainable, dq_trainable, dk_trainable, dv_trainable = run(h0_trainable)
        out_frozen, dq_frozen, dk_frozen, dv_frozen = run(h0.detach().clone())

        assert h0_trainable.grad is not None
        torch.testing.assert_close(out_frozen, out_trainable, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dq_frozen, dq_trainable, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dk_frozen, dk_trainable, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dv_frozen, dv_trainable, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", [(2, 96, 3, 48, 40), (1, 130, 2, 24, 56)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_kv_boundary(self, B, T, H, K, V, dtype, backend):
        """Check feature padding for key and value dimensions that are not powers of two."""
        self._prep(backend)
        torch.manual_seed(0)
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)

        o_ref, _ = delta_rule_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            output_final_state=False,
        )
        o_ref = o_ref.transpose(1, 2)

        o_out, _ = fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_headwise_beta(self, B, T, H, K, V, dtype, backend):
        """Compare per-channel beta values of shape [B, T, H, V] against the recurrence."""
        self._prep(backend)
        torch.manual_seed(1)
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        scale = K**-0.5

        q, k, v, _ = make_inputs(B, T, H, K, V, dtype)
        beta = (torch.sigmoid(torch.randn(B, T, H, V, device="cuda")) * (0.8 / K)).to(dtype)

        o_ref, _ = delta_rule_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            output_final_state=False,
        )
        o_ref = o_ref.transpose(1, 2)

        o_out, _ = fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_and_final_state(self, B, T, H, K, V, dtype, backend):
        """Forward with a nonzero initial_state and output_final_state=True."""
        self._prep(backend)
        torch.manual_seed(2)
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)
        # initial_state is [N(=B), H, K, V]; keep it small for recurrence stability.
        h0 = (torch.randn(B, H, K, V, device="cuda") * 0.05).to(torch.float32)

        o_ref, s_ref = delta_rule_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            initial_state=h0,
            output_final_state=True,
        )
        o_ref = o_ref.transpose(1, 2)

        o_out, s_out = fused_recurrent_delta_rule(
            q,
            k,
            v,
            beta=beta,
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
        )

        assert s_out is not None, "output_final_state=True must return a final state"
        assert s_out.shape == (B, H, K, V)
        assert rms_ratio(o_out, o_ref) < tol
        # Naive S is [B, H, K, V]; final_state from kernel matches that layout.
        assert rms_ratio(s_out, s_ref) < (1e-3 if dtype == torch.float32 else 2e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", [(2, 64, 2, 32, 32)])
    @pytest.mark.parametrize("scale", [1.0, 0.25, 2.0])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_custom_scale(self, B, T, H, K, V, scale, backend):
        """Forward with explicit scale values other than the 1/sqrt(K) default."""
        self._prep(backend)
        torch.manual_seed(3)
        dtype = torch.float32
        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)

        # Independent torch reference with explicit scale (q already scaled here).
        def ref_with_scale(q, k, v, beta, scale):
            # inputs [B, T, H, *]; operate per (b, h) over time.
            qh = q.transpose(1, 2).float()
            kh = k.transpose(1, 2).float()
            vh = v.transpose(1, 2).float()
            bh = beta.transpose(1, 2).float()[..., None]
            b, h, l, dk = qh.shape
            dv = vh.shape[-1]
            o = torch.zeros(b, h, l, dv, device=qh.device)
            S = torch.zeros(b, h, dk, dv, device=qh.device)
            qs = qh * scale
            for i in range(l):
                _k = kh[:, :, i]
                _v = vh[:, :, i] - (S * _k[..., None]).sum(-2)
                _v = _v * bh[:, :, i]
                S = S + _k.unsqueeze(-1) * _v.unsqueeze(-2)
                o[:, :, i] = torch.einsum("bhd,bhdm->bhm", qs[:, :, i], S)
            return o.transpose(1, 2)

        o_ref = ref_with_scale(q, k, v, beta, scale)
        o_out, _ = fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)

        assert rms_ratio(o_out, o_ref) < 1e-4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", [(2, 64, 2, 64, 64), (1, 96, 3, 32, 48)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_qk_l2norm_in_kernel(self, B, T, H, K, V, dtype, backend):
        """Forward with use_qk_l2norm_in_kernel=True (in-kernel L2 norm of q, k)."""
        self._prep(backend)
        torch.manual_seed(4)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        scale = K**-0.5

        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)

        eps = 1e-6
        qn = (q.float() / torch.sqrt(q.float().pow(2).sum(-1, keepdim=True) + eps)).to(dtype)
        kn = (k.float() / torch.sqrt(k.float().pow(2).sum(-1, keepdim=True) + eps)).to(dtype)

        o_ref, _ = delta_rule_recurrence(
            qn.transpose(1, 2),
            kn.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            output_final_state=False,
        )
        o_ref = o_ref.transpose(1, 2)

        o_out, _ = fused_recurrent_delta_rule(
            q,
            k,
            v,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", [(2, 32, 2, 48, 40)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_headwise_beta(self, B, T, H, K, V, backend):
        """Compare gradients for per-channel beta and padded key features against autograd."""
        self._prep(backend)
        torch.manual_seed(5)
        dtype = torch.float32
        scale = K**-0.5

        q, k, v, _ = make_inputs(B, T, H, K, V, dtype)
        beta = (torch.sigmoid(torch.randn(B, T, H, V, device="cuda")) * (0.8 / K)).to(dtype)

        qr = q.transpose(1, 2).detach().clone().requires_grad_(True)
        kr = k.transpose(1, 2).detach().clone().requires_grad_(True)
        vr = v.transpose(1, 2).detach().clone().requires_grad_(True)
        br = beta.transpose(1, 2).detach().clone()
        o_ref, _ = delta_rule_recurrence(qr, kr, vr, br, output_final_state=False)
        do = torch.randn_like(o_ref)
        o_ref.backward(do)
        dq_ref = qr.grad.transpose(1, 2)
        dk_ref = kr.grad.transpose(1, 2)
        dv_ref = vr.grad.transpose(1, 2)

        qt = q.detach().clone().requires_grad_(True)
        kt = k.detach().clone().requires_grad_(True)
        vt = v.detach().clone().requires_grad_(True)
        bt = beta.detach().clone()
        o_out, _ = fused_recurrent_delta_rule(qt, kt, vt, beta=bt, scale=scale)
        o_out.backward(do.transpose(1, 2).contiguous())

        assert rms_ratio(o_out, o_ref.transpose(1, 2)) < 1e-4
        assert rms_ratio(qt.grad, dq_ref) < 5e-2
        assert rms_ratio(kt.grad, dk_ref) < 5e-2
        assert rms_ratio(vt.grad, dv_ref) < 5e-2

    # (B, T, H, K, V)
    PERF_SHAPES = [
        # (B, T, H, K, V) -- aligned with fla upstream benchmark SHAPE_CONFIGS (K = V = D).
        (1, 8192, 96, 128, 128),  # B1_T8192_H96_D128
        (2, 16384, 16, 128, 128),  # B2_T16384_H16_D128
        (4, 2048, 16, 128, 128),  # B4_T2048_H16_D128
        (4, 4096, 64, 128, 128),  # B4_T4096_H64_D128
        (8, 2048, 32, 256, 256),  # B8_T2048_H32_D256
        (8, 1024, 8, 64, 64),  # B8_T1024_H8_D64
        # Non-power-of-2 K / partial-chunk boundary shape (T % 64 != 0, K != 2^n):
        # exercises the chunk-local boundary-zeroing path so its perf is tracked,
        # not just the aligned fast path.
        (2, 4000, 16, 120, 128),  # B2_T4000_H16_K120_V128 (non-pow2 K, partial T)
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
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
        scale = K**-0.5
        q, k, v, beta = make_inputs(B, T, H, K, V, dtype)

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the active backend matches the fp32 recurrence
        # reference on this exact perf shape BEFORE timing it (mirrors test_op's
        # bf16 path: same delta_rule_recurrence ref, same rms_ratio helper, same
        # [B,H,T,*] transpose, same bf16 tolerance 1e-2). The kernel runs in bf16
        # exactly as framework_fn benchmarks it. A hard assert here turns a
        # silently-broken kernel (e.g. NaN output) into a loud failure instead of
        # a fast-but-garbage benchmark number.
        o_chk, _ = fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)
        ref_chk, _ = delta_rule_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            beta.transpose(1, 2),
            output_final_state=False,
        )
        ref_chk = ref_chk.transpose(1, 2)
        ratio = rms_ratio(o_chk, ref_chk)
        assert ratio < 1e-2, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del o_chk, ref_chk

        # Enable autograd on q/k/v so that, under MODE=auto (default) or
        # MODE=backward, benchmark() also times the standalone
        # fused_recurrent_delta_rule_bwd kernel. beta carries no grad in the
        # bwd path (matches test_backward), so it is left as-is.
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)

        framework_fn = lambda beta=beta, k=k, q=q, v=v: fused_recurrent_delta_rule(q, k, v, beta=beta, scale=scale)[0]

        # grad_to_none resets input grads between reps so backward timing is not
        # polluted by gradient accumulation. In MODE=auto this records both
        # 'forward' and 'backward'; MODE=forward preserves the prior fwd-only result.
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
        record_property("op", "fused_recurrent_delta_rule")

        del q, k, v, beta, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
