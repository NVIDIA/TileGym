# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common

_BACKENDS = ["cutile"]


def _resolve(backend, module, *names):
    """Import the backend-specific impl(s) from tilegym.suites.fla.<backend>.<module>."""
    if backend == "cutile":
        from tilegym.suites.fla.cutile import linear_attention_gdn_gate as _m
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return tuple(getattr(_m, n) for n in names) if len(names) != 1 else getattr(_m, names[0])


def naive_gdn_gate(g, A_log, dt_bias=None, output_dtype=torch.float32):
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float()
    return (-A_log.float().exp() * torch.nn.functional.softplus(g)).to(output_dtype)


def _rms_ratio(test, ref):
    test32, ref32 = test.float(), ref.float()
    rms_err = (test32 - ref32).square().mean().sqrt()
    rms_ref = ref32.square().mean().sqrt()
    return (rms_err / (rms_ref + 1e-8)).item()


def _ref_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=None, cu_seqlens=None):
    """Independent torch reference for ``gdn_gate_chunk_cumsum``.

    Mirrors the kernel math: per-(batch, head) sequence, compute the
    elementwise gate ``-exp(A_log) * softplus(g + dt_bias)`` then a *within-chunk*
    (length-``chunk_size``) cumulative sum along time, optionally scaled. For the
    varlen case each sub-sequence delimited by ``cu_seqlens`` is processed
    independently and its own chunks are cumsummed in isolation.
    """
    B, T, H = g.shape
    gate = naive_gdn_gate(g, A_log, dt_bias, output_dtype=torch.float32)  # [B, T, H]
    out = torch.empty_like(gate)

    def _cumsum_seq(seq):  # seq: [L, H]
        L = seq.shape[0]
        res = torch.empty_like(seq)
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            res[start:end] = torch.cumsum(seq[start:end], dim=0)
        return res

    if cu_seqlens is None:
        for b in range(B):
            out[b] = _cumsum_seq(gate[b])
    else:
        cu = cu_seqlens.tolist()
        flat = gate.reshape(-1, H)  # B==1 varlen layout: [total_T, H]
        res = torch.empty_like(flat)
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            res[bos:eos] = _cumsum_seq(flat[bos:eos])
        out = res.reshape(B, T, H)

    if scale is not None:
        out = out * scale
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionGdnGate(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable)."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @staticmethod
    def _fixed_inputs():
        """Fixed-seed inputs replacing the former fla `.pt` fixture."""
        torch.manual_seed(0)
        B, T, H = 2, 128, 8
        chunk_size = 64
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)
        return g, A_log, dt_bias, chunk_size

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_fused_fwd(self, backend):
        """`fused_gdn_gate` / `gdn_gate_fwd` forward vs the torch reference.

        Was previously gated on the legacy fla reference; now self-validates
        against the in-file `naive_gdn_gate`. Retains coverage of the
        `fused_gdn_gate` entry point, which no other test exercises.
        """
        self._prep(backend)
        fused_gdn_gate, gdn_gate_fwd = _resolve(backend, "linear_attention_gdn_gate", "fused_gdn_gate", "gdn_gate_fwd")
        g, A_log, dt_bias, _ = self._fixed_inputs()

        yg = fused_gdn_gate(g, A_log, dt_bias, output_dtype=torch.float32)
        yg_fwd = gdn_gate_fwd(g, A_log, dt_bias, output_dtype=torch.float32)
        ref = naive_gdn_gate(g, A_log, dt_bias, output_dtype=torch.float32)

        assert _rms_ratio(yg, ref) < 1e-3
        assert _rms_ratio(yg_fwd, ref) < 1e-3

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_scale_bias(self, backend):
        """`gdn_gate_chunk_cumsum` with/without scale and bias vs torch reference."""
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        g, A_log, dt_bias, chunk_size = self._fixed_inputs()

        cs = gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias)
        cs_scale = gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=0.5, dt_bias=None)

        assert _rms_ratio(cs, _ref_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias)) < 1e-3
        assert _rms_ratio(cs_scale, _ref_chunk_cumsum(g, A_log, chunk_size, scale=0.5, dt_bias=None)) < 1e-3

    @pytest.mark.parametrize("H", [8, 16])
    @pytest.mark.parametrize("chunk_size", [16, 64, 128])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_aligned_and_tail(self, backend, chunk_size, H):
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        torch.manual_seed(8)
        A_log = torch.randn(H, device="cuda")
        dt_bias = torch.randn(H, device="cuda")
        for T in [2 * chunk_size, 2 * chunk_size + 1]:
            g = torch.randn(2, T, H, device="cuda")
            actual = gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=0.5, dt_bias=dt_bias)
            reference = _ref_chunk_cumsum(g, A_log, chunk_size, scale=0.5, dt_bias=dt_bias)
            assert _rms_ratio(actual, reference) < 1e-3

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_varlen(self, backend):
        """`gdn_gate_chunk_cumsum` varlen path vs the per-sequence torch reference."""
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        torch.manual_seed(1)
        H = 8
        chunk_size = 64
        seqlens = [64, 128, 192]
        total = sum(seqlens)
        cu_seqlens = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device="cuda", dtype=torch.long)
        g_vl = torch.randn(1, total, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)

        cs_vl = gdn_gate_chunk_cumsum(g_vl, A_log, chunk_size, scale=None, dt_bias=dt_bias, cu_seqlens=cu_seqlens)
        ref = _ref_chunk_cumsum(g_vl, A_log, chunk_size, scale=None, dt_bias=dt_bias, cu_seqlens=cu_seqlens)
        assert _rms_ratio(cs_vl, ref) < 1e-3

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_bwd(self, backend):
        """`gdn_gate_bwd` vs autograd through the forward reference `naive_gdn_gate`."""
        self._prep(backend)
        gdn_gate_bwd = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_bwd")
        g, A_log, dt_bias, _ = self._fixed_inputs()
        torch.manual_seed(2)
        dyg = torch.randn_like(g)

        g_r = g.clone().requires_grad_(True)
        A_r = A_log.clone().requires_grad_(True)
        b_r = dt_bias.clone().requires_grad_(True)
        out = naive_gdn_gate(g_r, A_r, b_r, output_dtype=torch.float32)
        out.backward(dyg)

        dg, dA, dbias = gdn_gate_bwd(g, A_log, dt_bias, dyg)
        torch.testing.assert_close(dg, g_r.grad, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(dA, A_r.grad, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(dbias, b_r.grad, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_fwd_no_bias_dtypes(self, dtype, backend):
        self._prep(backend)
        gdn_gate_fwd, fused_gdn_gate = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_fwd", "fused_gdn_gate")
        B, T, H = 2, 100, 8
        g = torch.randn(B, T, H, device="cuda", dtype=dtype)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)

        yg = gdn_gate_fwd(g, A_log, dt_bias=None, output_dtype=torch.float32)
        yg_fused = fused_gdn_gate(g, A_log, dt_bias=None, output_dtype=torch.float32)
        ref = naive_gdn_gate(g, A_log, dt_bias=None, output_dtype=torch.float32)

        tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.float16 else 1e-1)
        assert _rms_ratio(yg, ref) < tol
        assert _rms_ratio(yg_fused, ref) < tol

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_fwd_softplus_saturation(self, backend):
        """Check softplus saturation above 20, including inputs shifted by a bias."""
        self._prep(backend)
        gdn_gate_fwd = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_fwd")
        B, T, H = 1, 64, 4
        g = torch.linspace(-30.0, 60.0, B * T * H, device="cuda").reshape(B, T, H)
        A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.full((H,), 5.0, device="cuda", dtype=torch.float32)

        yg = gdn_gate_fwd(g, A_log, dt_bias=dt_bias, output_dtype=torch.float32)
        ref = naive_gdn_gate(g, A_log, dt_bias=dt_bias, output_dtype=torch.float32)
        assert torch.isfinite(yg).all()
        assert _rms_ratio(yg, ref) < 1e-3

    @pytest.mark.parametrize("T,chunk_size", [(100, 32), (128, 64), (37, 16)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_partial_chunk(self, T, chunk_size, backend):
        """Compare partial and full chunk sums against the per-chunk reference."""
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        B, H = 2, 4
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)

        out = gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias)
        ref = _ref_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias)
        assert _rms_ratio(out, ref) < 1e-3

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_scale_and_no_bias(self, backend):
        """`gdn_gate_chunk_cumsum` with HAS_SCALE=True and dt_bias=None together."""
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        B, T, H, chunk_size = 2, 96, 4, 32
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)

        out = gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=0.25, dt_bias=None)
        ref = _ref_chunk_cumsum(g, A_log, chunk_size, scale=0.25, dt_bias=None)
        assert _rms_ratio(out, ref) < 1e-3

    @pytest.mark.parametrize("chunk_size", [16, 32])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_varlen_uneven(self, chunk_size, backend):
        """`gdn_gate_chunk_cumsum` varlen path with uneven sub-sequences."""
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        H = 4
        lengths = [5, chunk_size + 7, 1, 2 * chunk_size]
        total = sum(lengths)
        cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.long)
        g = torch.randn(1, total, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)

        out = gdn_gate_chunk_cumsum(
            g,
            A_log,
            chunk_size,
            scale=None,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
        )
        ref = _ref_chunk_cumsum(
            g,
            A_log,
            chunk_size,
            scale=None,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
        )
        assert _rms_ratio(out, ref) < 1e-3

    @pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_cumsum_output_dtype(self, output_dtype, backend):
        """`gdn_gate_chunk_cumsum` with non-default low-precision output_dtype."""
        self._prep(backend)
        gdn_gate_chunk_cumsum = _resolve(backend, "linear_attention_gdn_gate", "gdn_gate_chunk_cumsum")
        B, T, H, chunk_size = 2, 80, 4, 32
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)

        out = gdn_gate_chunk_cumsum(
            g,
            A_log,
            chunk_size,
            scale=None,
            dt_bias=dt_bias,
            output_dtype=output_dtype,
        )
        ref = _ref_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias)
        assert out.dtype == output_dtype
        tol = 5e-2 if output_dtype == torch.float16 else 1e-1
        assert _rms_ratio(out, ref) < tol

    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf(self, framework, record_property):
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        fused_gdn_gate, gdn_gate_chunk_cumsum = _resolve(
            framework, "linear_attention_gdn_gate", "fused_gdn_gate", "gdn_gate_chunk_cumsum"
        )
        # Reconstruct inputs inline (no dependency on any saved fla `.pt` fixture).
        torch.manual_seed(0)
        B, T, H = 4, 2048, 8
        chunk_size = 64
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify both benchmarked kernels match the torch reference
        # at this perf shape BEFORE timing (mirrors test_fused_fwd /
        # test_chunk_cumsum_scale_bias).
        y_ref = naive_gdn_gate(g, A_log, dt_bias, output_dtype=torch.float32)
        r_fused = _rms_ratio(fused_gdn_gate(g, A_log, dt_bias, output_dtype=torch.float32), y_ref)
        assert r_fused < 1e-3, f"{framework} fused_gdn_gate mismatch before perf (rms_ratio={r_fused:.3e})"
        cs_ref = _ref_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias)
        r_cs = _rms_ratio(gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias), cs_ref)
        assert r_cs < 1e-3, f"{framework} gdn_gate_chunk_cumsum mismatch before perf (rms_ratio={r_cs:.3e})"

        framework_fn = lambda A_log=A_log, dt_bias=dt_bias, g=g: (
            fused_gdn_gate(g, A_log, dt_bias, output_dtype=torch.float32),
            gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=dt_bias),
        )

        result = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", result)
        record_property("shape", str(tuple(g.shape)))
        record_property("op", "fused_gdn_gate+gdn_gate_chunk_cumsum")

        del g, A_log, dt_bias, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf_bwd(self, framework, record_property):
        # Backward perf for gdn_gate_bwd / gdn_gate_bwd_kernel. The forward
        # test_perf above only benchmarks fused_gdn_gate + gdn_gate_chunk_cumsum;
        # the gate backward (reachable via GDNGateFunction.backward) is otherwise
        # uncovered by any perf path. Call gdn_gate_bwd directly, mirroring the
        # test_bwd correctness test.
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        gdn_gate_bwd = _resolve(framework, "linear_attention_gdn_gate", "gdn_gate_bwd")
        torch.manual_seed(0)
        B, T, H = 4, 2048, 8
        g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)
        dyg = torch.randn_like(g)

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the backward gradients match autograd through the
        # forward reference at this perf shape BEFORE timing (mirrors test_op_bwd,
        # same reference and using the file's _rms_ratio helper). Inputs are fp32,
        # so this gate is already computed against an fp32 reference.
        g_r = g.clone().requires_grad_(True)
        A_r = A_log.clone().requires_grad_(True)
        b_r = dt_bias.clone().requires_grad_(True)
        naive_gdn_gate(g_r, A_r, b_r, output_dtype=torch.float32).backward(dyg)
        dg, dA, dbias = gdn_gate_bwd(g, A_log, dt_bias, dyg)
        ratio = max(
            _rms_ratio(dg, g_r.grad),
            _rms_ratio(dA, A_r.grad),
            _rms_ratio(dbias, b_r.grad),
        )
        assert ratio < 5e-3, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del g_r, A_r, b_r, dg, dA, dbias

        framework_fn = lambda A_log=A_log, dt_bias=dt_bias, dyg=dyg, g=g: gdn_gate_bwd(g, A_log, dt_bias, dyg)[0]

        result = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", result)
        record_property("shape", str(tuple(g.shape)))
        record_property("op", "gdn_gate_bwd")

        del g, A_log, dt_bias, dyg, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
