# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Tests for the self-contained port of fla's KDA intra-chunk forward + backward
helpers (`fla.ops.kda.chunk_intra`): `chunk_kda_fwd_intra` and
`chunk_kda_bwd_intra`.

PRIMARY correctness gate: independent, framework-free PyTorch references that
re-implement each kernel's exact semantics in float32 (`_aqk_reference`,
`_chunk_intra_fwd_wy_torch_ref` for the forward, `_chunk_intra_bwd_torch_ref`
for the backward). The test runs the self-contained port (no `fla`) and the
torch references on identical fixed-seed inputs and asserts the outputs match
within an RMS-error ratio gate. No `fla` import and no saved fla `.pt` fixture
are required. Output tensors that the kernels only partially write are
zero-initialized so unwritten regions are a deterministic 0 instead of `empty`
garbage.
"""

import pytest
import torch

import tilegym
from tests import common

_BASE = common.PyTestCase

_BACKENDS = ["cutile"]


def _resolve(backend, module, *names):
    """Import the backend-specific impl(s) from tilegym.suites.fla.<backend>.<module>."""
    assert module == "linear_attention_kda_chunk_intra", module
    if backend == "cutile":
        from tilegym.suites.fla.cutile import linear_attention_kda_chunk_intra as _m
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return tuple(getattr(_m, n) for n in names) if len(names) != 1 else getattr(_m, names[0])


def _rms_ratio(test: torch.Tensor, ref: torch.Tensor) -> float:
    test32, ref32 = test.float(), ref.float()
    rms_err = (test32 - ref32).square().mean().sqrt()
    rms_ref = ref32.square().mean().sqrt()
    return (rms_err / (rms_ref + 1e-8)).item()


def _aqk_causal_mask(T: int, BT: int, device) -> torch.Tensor:
    """Block-causal write-mask for Aqk [.., T, .., BT].

    For the row at absolute time `t` (chunk i = t // BT), the kernel writes the
    BT columns `c` whose source position `i*BT + c <= t`, i.e. `c <= t % BT`.
    Other columns (strict upper triangle / padding) are left as `empty` garbage,
    so they must be excluded from the comparison in BOTH oracle and port.
    """
    t = torch.arange(T, device=device)
    r = (t % BT).view(T, 1)  # chunk-local row index
    c = torch.arange(BT, device=device).view(1, BT)
    return c <= r  # [T, BT]


# Fixed problem shape and chunk params shared by oracle generator + test.
_B, _T, _H, _K, _V = 2, 128, 3, 64, 64
_HV = _H
_BT = 64
_SCALE = _K**-0.5


def _make_inputs(dev):
    """Build the fixed-seed KDA inputs (q, k, v, gk, beta) used everywhere."""
    torch.manual_seed(0)
    B, T, H, K, HV, V = _B, _T, _H, _K, _HV, _V

    q = torch.randn(B, T, H, K, device=dev)
    q = torch.nn.functional.normalize(q, p=2, dim=-1).to(torch.bfloat16)
    k = torch.randn(B, T, H, K, device=dev)
    k = torch.nn.functional.normalize(k, p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, HV, V, device=dev).to(torch.bfloat16)
    beta = torch.rand(B, T, HV, device=dev).to(torch.bfloat16)
    # gk: per-channel log2-domain cumulative gates, [B, T, HV, K], float32.
    # clamp away from 0: torch.rand() is [0, 1) and can draw an exact 0.0,
    # whose log() is -inf and poisons the gate cumsum (-> exp2(nan) downstream).
    gk = torch.rand(B, T, HV, K, device=dev).clamp_min(1e-7).log().cumsum(1).to(torch.float32)
    return q, k, v, beta, gk


def _make_inputs_shape(dev, B, T, H, K, HV, V, seed):
    """Build fixed-seed KDA inputs for an arbitrary shape (mirrors `_make_inputs`).

    Used by the coverage tests that need shapes other than the single fixed
    shape of the legacy fla test (e.g. grouped-query HV>H, BT=32, non-BT-multiple T).
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=dev)
    q = torch.nn.functional.normalize(q, p=2, dim=-1).to(torch.bfloat16)
    k = torch.randn(B, T, H, K, device=dev)
    k = torch.nn.functional.normalize(k, p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, HV, V, device=dev).to(torch.bfloat16)
    beta = torch.rand(B, T, HV, device=dev).to(torch.bfloat16)
    # clamp away from 0: torch.rand() is [0, 1) and can draw an exact 0.0,
    # whose log() is -inf and poisons the gate cumsum (-> exp2(nan) downstream).
    gk = torch.rand(B, T, HV, K, device=dev).clamp_min(1e-7).log().cumsum(1).to(torch.float32)
    return q, k, v, beta, gk


def _aqk_reference(q, k, gk, scale, BT):
    """Independent fp32 torch reference for the block-causal Aqk produced by
    `chunk_kda_fwd_intra` (combining the diagonal + off-diagonal blocks).

    For an output row at absolute time `t` (chunk `c = t // BT`) and chunk-local
    column `j_local in [0, BT)` mapping to source token `s = c*BT + j_local`, the
    kernel writes (for the block-causal lower triangle `s <= t`):

        Aqk[b, t, hv, j_local]
            = scale * sum_k q[b,t,h,k] * k[b,s,h,k] * exp2(gk[b,t,hv,k] - gk[b,s,hv,k])

    where `h = hv // (HV // H)` maps each value head to its (shared) qk head.
    This is the exact algebraic identity the kernel computes; the kernel only
    differs by fp32 vs bf16 rounding and by the gate-pivot reformulation, so the
    comparison is done in fp32 with an RMS ratio gate. Columns outside the
    block-causal lower triangle are zeroed in both (kernel leaves them garbage).
    """
    B, T, H, K = q.shape
    HV = gk.shape[2]
    G = HV // H
    dev = q.device
    qf = q.float()
    kf = k.float()
    gf = gk.float()

    out = torch.zeros(B, T, HV, BT, device=dev, dtype=torch.float32)
    t_idx = torch.arange(T, device=dev)
    c = t_idx // BT
    t_local = t_idx % BT
    for j_local in range(BT):
        s = c * BT + j_local  # [T] source token for this column
        valid = (s <= t_idx) & (s < T)  # block-causal + within-seq
        s_clamped = s.clamp(max=T - 1)
        for hv in range(HV):
            h = hv // G
            q_t = qf[:, :, h, :]  # [B, T, K]
            k_s = kf[:, s_clamped, h, :]  # [B, T, K]
            g_t = gf[:, :, hv, :]  # [B, T, K]
            g_s = gf[:, s_clamped, hv, :]  # [B, T, K]
            val = scale * (q_t * k_s * torch.exp2(g_t - g_s)).sum(-1)  # [B, T]
            val = torch.where(valid.view(1, T), val, torch.zeros_like(val))
            out[:, :, hv, j_local] = val
    return out


def _aqk_written_mask(T, BT, device):
    """Block-causal write mask for Aqk [.., T, .., BT] (same rule as the kernel:
    column `j_local` is written for output row `t` iff `c*BT + j_local <= t`)."""
    t = torch.arange(T, device=device)
    c = t // BT
    j = torch.arange(BT, device=device).view(1, BT)
    s = c.view(T, 1) * BT + j
    return (s <= t.view(T, 1)) & (s < T)


def _chunk_intra_fwd_wy_torch_ref(k, v, beta, gk, BT):
    """Independent fp32 torch reference for the WY-solve + recompute outputs of
    `chunk_kda_fwd_intra` (everything except Aqk, which `_aqk_reference` covers).

    The port's returned ``Akk`` is the per-chunk *solved* WY matrix, i.e. the
    unit-lower-triangular inverse ``A = (I + M)^-1`` of the strictly-lower
    intra-chunk matrix (cf. fla `naive_chunk_kda`):

        M[b, i, j] = beta[t_i] * sum_d k[t_i,h,d] * k[t_j,h,d]
                                       * exp2(gk[t_i,hv,d] - gk[t_j,hv,d])   (j < i)

    where ``h = hv // (HV // H)`` maps each value head to its (shared) qk head.
    ``A`` is then fed to `recompute_w_u_fwd` (q=None here, so qg is dead):

        u  = A @ (v * beta)
        w  = A @ (k * beta * exp2(gk))
        kg = k * exp2(gk[last_valid_row] - gk)

    Returns (w, u, kg, Akk) as fp32, zero outside each chunk's written region.
    The gate difference is clamped to <= 0 before exp2 (gk is a non-increasing
    cumulative log, so every valid lower-tri entry is already <= 0; the clamp
    only neutralizes the masked-out upper triangle to avoid inf*0 -> nan).
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    kf, vf, bf, gf = k.float(), v.float(), beta.float(), gk.float()

    w = torch.zeros(B, T, HV, K, device=k.device, dtype=torch.float32)
    u = torch.zeros(B, T, HV, V, device=k.device, dtype=torch.float32)
    kg = torch.zeros(B, T, HV, K, device=k.device, dtype=torch.float32)
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=torch.float32)

    NT = (T + BT - 1) // BT
    for c in range(NT):
        t0 = c * BT
        t1 = min(t0 + BT, T)
        n = t1 - t0
        last = t1 - 1
        sl = slice(t0, t1)
        tril = torch.tril(torch.ones(n, n, device=k.device), diagonal=-1).bool()
        eye = torch.eye(n, device=k.device).expand(B, n, n)
        for hv in range(HV):
            h = hv // G
            kh = kf[:, sl, h, :]  # [B, n, K]
            gh = gf[:, sl, hv, :]  # [B, n, K]
            bh = bf[:, sl, hv]  # [B, n]

            # M[i,j] = beta_i * sum_d k_i,d k_j,d exp2(g_i,d - g_j,d), strictly lower.
            diff = (gh[:, :, None, :] - gh[:, None, :, :]).clamp_max(0.0)  # [B,n,n,K]
            kk = kh[:, :, None, :] * kh[:, None, :, :]  # [B,n,n,K]
            M = (kk * torch.exp2(diff)).sum(-1) * bh[:, :, None]  # [B,n,n]
            M = M * tril  # strictly-lower only
            A = torch.linalg.inv(eye + M)  # [B,n,n] unit-lower-triangular inverse
            Akk[:, sl, hv, :n] = A

            vb = vf[:, sl, hv, :] * bh[:, :, None]
            u[:, sl, hv, :] = torch.einsum("bij,bjv->biv", A, vb)
            kb = kh * bh[:, :, None] * torch.exp2(gh)
            w[:, sl, hv, :] = torch.einsum("bij,bjk->bik", A, kb)
            g_last = gf[:, last, hv, :]  # [B, K]
            kg[:, sl, hv, :] = kh * torch.exp2(g_last[:, None, :] - gh)
    return w, u, kg, Akk


def _chunk_intra_bwd_torch_ref(q, k, g, beta, dAqk, dAkk, dq_in, dk_in, db_in, dg_in, BT):
    """Independent fp32 torch reference mirroring `chunk_kda_bwd_intra`
    (`safe_gate=False`): the intra-chunk gradient pass that, given the upstream
    grads `dAqk`/`dAkk` of the raw block-causal matrices, *accumulates* its
    contribution onto the incoming `dq`/`dk`/`db`/`dg` buffers.

    The raw intra-chunk forward (per chunk, head ``hv`` -> qk head ``h = hv//G``,
    ``G = HV//H``) writes, with ``w[i,j,d] = 2**(g[i,d] - g[j,d])``:

        Aqk[i,j] = sum_d q[i,d] k[j,d] w[i,j,d]          (j <= i, incl. diagonal)
        Akk[i,j] = beta[i] * sum_d k[i,d] k[j,d] w[i,j,d]  (j <= i)

    Differentiating and matching the kernel's exact algebra (no `scale`, no
    `ln2` -- those live in the chunk-level wrapper) gives, for each chunk:

        dq2[i,d]     = sum_{j<=i} dAqk[i,j] k[j,d] w[i,j,d]
        dk2row[i,d]  = sum_{j<=i} dAkk[i,j] k[j,d] w[i,j,d]
        db[i]        = sum_d k[i,d] dk2row[i,d]
        dk2beta[i,d] = beta[i] * dk2row[i,d]
        dkt[i,d]     = sum_{r>=i} ( dAqk[r,i] q[r,d]
                                    + dAkk[r,i] beta[r] k[r,d] ) w[r,i,d]

    accumulated as:

        dq[i,d] += dq2[i,d]
        dk[i,d] += dk2beta[i,d] + dkt[i,d]
        db[i]   += db[i]
        dg[i,d] += q[i,d] dq2[i,d] + (dk2beta[i,d] - dkt[i,d]) k[i,d]

    All sums are block-causal *inside* each chunk (incl. the diagonal, matching
    the kernel's `i >= j` masks). Returns (dq, dk, db, dg) as fp32.
    """
    B, T, H, K = k.shape
    HV = g.shape[2]
    G = HV // H
    qf, kf, gf, bf = q.float(), k.float(), g.float(), beta.float()
    dAqk_f, dAkk_f = dAqk.float(), dAkk.float()
    dev = k.device

    dq = dq_in.float().clone()
    dk = dk_in.float().clone()
    db = db_in.float().clone()
    dg = dg_in.float().clone()

    NT = (T + BT - 1) // BT
    for c in range(NT):
        t0 = c * BT
        t1 = min(t0 + BT, T)
        n = t1 - t0
        sl = slice(t0, t1)
        idx = torch.arange(n, device=dev)
        # low[a,b] = (b <= a): the row mask low[i,j]=(j<=i) and the column mask
        # low[r,i]=(r>=i) are both reads of this one block-causal triangle.
        low = (idx[:, None] >= idx[None, :]).float()  # [n,n]
        for hv in range(HV):
            h = hv // G
            qh = qf[:, sl, h, :]  # [B,n,K]
            kh = kf[:, sl, h, :]  # [B,n,K]
            gh = gf[:, sl, hv, :]  # [B,n,K]
            bh = bf[:, sl, hv]  # [B,n]
            dAq = dAqk_f[:, sl, hv, :n]  # [B,n,n] (chunk-local cols of the BT-wide block)
            dAk = dAkk_f[:, sl, hv, :n]  # [B,n,n]

            # Pairwise gate factor w[a,c,d] = 2**(g[a,d] - g[c,d]). Clamp the
            # difference to <= 0 so the (unused) strict-upper entries can't
            # overflow before masking (inf * 0 -> nan); every block-causal entry
            # we keep already has g[a]-g[c] <= 0 since g is a non-increasing
            # cumulative gate. This mirrors the kernel's per-subchunk gate pivot
            # without the factored 2**g[i] * 2**-g[j] form (which over/underflows
            # for the long log-cumsum gates used here).
            w = torch.exp2((gh[:, :, None, :] - gh[:, None, :, :]).clamp_max(0.0))  # [B,n,n,K]
            kbf = kh * bh[:, :, None]  # beta[r] k[r,d]

            # Row contributions (cols j <= i): w[i,j,d].
            dq2 = torch.einsum("bij,bjd,bijd->bid", dAq * low, kh, w)
            dk2row = torch.einsum("bij,bjd,bijd->bid", dAk * low, kh, w)
            dbloc = (kh * dk2row).sum(-1)  # [B,n]
            dk2beta = bh[:, :, None] * dk2row

            # Column contribution dkt (rows r >= i, k[i] as inner column): w[r,i,d].
            dkt = torch.einsum("bri,brd,brid->bid", dAq * low, qh, w) + torch.einsum(
                "bri,brd,brid->bid", dAk * low, kbf, w
            )

            dq[:, sl, hv, :] += dq2
            dk[:, sl, hv, :] += dk2beta + dkt
            db[:, sl, hv] += dbloc
            dg[:, sl, hv, :] += qh * dq2 + (dk2beta - dkt) * kh
    return dq, dk, db, dg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionKdaChunkIntra(_BASE):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable)."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra(self, backend):
        """PRIMARY forward gate: kernel == framework-free torch references.

        Runs the port's `chunk_kda_fwd_intra` and independent fp32 torch
        references on identical fixed-seed inputs at the canonical shape
        (B=2, T=128, H=HV=3, K=V=64, BT=64, bf16, safe_gate=False) and asserts
        every written output agrees within an RMS-ratio gate:
          - Aqk : block-causal q.k gated matrix (`_aqk_reference`, compared only
                  on the written lower triangle -- the kernel leaves the rest
                  as `empty` garbage),
          - Akk : per-chunk unit-lower-triangular WY inverse A = (I + M)^-1,
          - w, u, kg : `recompute_w_u_fwd` outputs derived from A,
        the last three via `_chunk_intra_fwd_wy_torch_ref`. Replaces the former
        fla `.pt` fixture forward gate -- no fla, no saved file.
        """
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=_SCALE,
            cu_seqlens=None,
            chunk_size=_BT,
            safe_gate=False,
            disable_recompute=False,
        )
        # qg is None when disable_recompute is False.
        assert qg is None

        # Aqk: compare only the block-causal written lower triangle.
        self._check_aqk(q, k, gk, _SCALE, _BT, Aqk)

        # w, u, kg, Akk: compare against the WY-solve torch reference.
        rw, ru, rkg, rAkk = _chunk_intra_fwd_wy_torch_ref(k, v, beta, gk, _BT)
        for name, got, ref in (
            ("w", w, rw),
            ("u", u, ru),
            ("kg", kg, rkg),
            ("Akk", Akk, rAkk),
        ):
            ratio = _rms_ratio(got, ref)
            assert ratio < 2e-2, f"{name} RMS ratio {ratio:.3e} exceeds 2e-2"

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_bwd_intra(self, backend):
        """PRIMARY backward gate: kernel == framework-free torch reference.

        Runs the port's `chunk_kda_bwd_intra` and the independent fp32 torch
        reference `_chunk_intra_bwd_torch_ref` on identical fixed-seed inputs at
        the canonical shape (B=2, T=128, H=HV=3, K=V=64, BT=64, safe_gate=False)
        and asserts every accumulated gradient (dq, dk, db, dg) agrees within an
        RMS-ratio gate. Replaces the former fla `.pt` fixture backward
        gate -- no fla, no saved file.
        """
        self._prep(backend)
        chunk_kda_bwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_bwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        B, T, HV, K, BT = _B, _T, _HV, _K, _BT
        dev = q.device

        # Fixed-seed accumulator inputs that the kernel reads and adds onto.
        torch.manual_seed(1)
        dAqk = torch.randn(B, T, HV, BT, device=dev, dtype=torch.float32)
        dAkk = torch.randn(B, T, HV, BT, device=dev, dtype=torch.float32)
        dq_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        dk_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        db_in = torch.randn(B, T, HV, device=dev, dtype=torch.float32)
        dg_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)

        dq, dk, db, dg = chunk_kda_bwd_intra(
            q=q,
            k=k,
            g=gk,
            beta=beta,
            dAqk=dAqk.clone(),
            dAkk=dAkk.clone(),
            dq=dq_in.clone(),
            dk=dk_in.clone(),
            db=db_in.clone(),
            dg=dg_in.clone(),
            cu_seqlens=None,
            chunk_size=BT,
            safe_gate=False,
        )

        rdq, rdk, rdb, rdg = _chunk_intra_bwd_torch_ref(q, k, gk, beta, dAqk, dAkk, dq_in, dk_in, db_in, dg_in, BT)

        for name, got, ref in (
            ("dq", dq, rdq),
            ("dk", dk, rdk),
            ("db", db, rdb),
            ("dg", dg, rdg),
        ):
            ratio = _rms_ratio(got, ref)
            assert ratio < 2e-2, f"{name} RMS ratio {ratio:.3e} exceeds 2e-2"

    # ------------------------------------------------------------------
    # Coverage tests (independent fp32 torch reference; no fla, no saved fla fixture).
    # ------------------------------------------------------------------

    def _check_aqk(self, q, k, gk, scale, BT, Aqk, tol=5e-2):
        """Compare the kernel's written Aqk lower-triangle against the
        independent fp32 reference using the file's RMS-ratio convention."""
        T = q.shape[1]
        m = _aqk_written_mask(T, BT, Aqk.device).view(1, T, 1, BT)
        ref = _aqk_reference(q, k, gk, scale, BT)
        got = torch.where(m, Aqk.float(), torch.zeros_like(Aqk, dtype=torch.float32))
        ref = torch.where(m, ref, torch.zeros_like(ref))
        ratio = _rms_ratio(got, ref)
        assert ratio < tol, f"Aqk RMS ratio {ratio:.3e} exceeds {tol:.1e}"

    @pytest.mark.parametrize("safe_gate", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra_safe_gate(self, safe_gate, backend):
        """Forward Aqk correctness for both safe_gate paths."""
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=_SCALE,
            cu_seqlens=None,
            chunk_size=_BT,
            safe_gate=safe_gate,
            disable_recompute=False,
        )
        self._check_aqk(q, k, gk, _SCALE, _BT, Aqk)

    @pytest.mark.parametrize("BT", [32, 64])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra_chunk_size(self, BT, backend):
        """Forward Aqk correctness for chunk_size 32 and 64."""
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=_SCALE,
            cu_seqlens=None,
            chunk_size=BT,
            safe_gate=False,
            disable_recompute=False,
        )
        self._check_aqk(q, k, gk, _SCALE, BT, Aqk)

    @pytest.mark.parametrize("T", [96, 160])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra_partial_chunk(self, T, backend):
        """Forward Aqk correctness when T is not a multiple of BT (tail masking)."""
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs_shape("cuda", _B, T, _H, _K, _HV, _V, seed=7)
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=_SCALE,
            cu_seqlens=None,
            chunk_size=_BT,
            safe_gate=False,
            disable_recompute=False,
        )
        self._check_aqk(q, k, gk, _SCALE, _BT, Aqk)

    @pytest.mark.parametrize("H,HV", [(2, 4), (1, 4)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra_grouped_query(self, H, HV, backend):
        """Forward Aqk correctness for grouped-value attention (HV > H)."""
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs_shape("cuda", _B, _T, H, _K, HV, _V, seed=11)
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=_SCALE,
            cu_seqlens=None,
            chunk_size=_BT,
            safe_gate=False,
            disable_recompute=False,
        )
        assert Aqk.shape == (_B, _T, HV, _BT)
        assert w.shape == (_B, _T, HV, _K)
        self._check_aqk(q, k, gk, _SCALE, _BT, Aqk)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra_disable_recompute_qg(self, backend):
        """Forward produces qg (STORE_QG path) when disable_recompute=True."""
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=_SCALE,
            cu_seqlens=None,
            chunk_size=_BT,
            safe_gate=False,
            disable_recompute=True,
        )
        assert qg is not None
        assert qg.shape == (_B, _T, _HV, _K)
        G = _HV // _H
        h_of_hv = torch.arange(_HV, device=q.device) // G
        q_bcast = q.float()[:, :, h_of_hv, :]  # [B, T, HV, K]
        qg_ref = q_bcast * torch.exp2(gk.float())
        ratio = _rms_ratio(qg.float(), qg_ref)
        assert ratio < 5e-2, f"qg RMS ratio {ratio:.3e} exceeds 5e-2"

    @pytest.mark.parametrize("BT", [16, 128])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_fwd_intra_invalid_chunk_size(self, BT, backend):
        """chunk_size outside {32, 64} must raise ValueError."""
        self._prep(backend)
        chunk_kda_fwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        with pytest.raises(ValueError):
            chunk_kda_fwd_intra(
                q=q,
                k=k,
                v=v,
                gk=gk,
                beta=beta,
                scale=_SCALE,
                cu_seqlens=None,
                chunk_size=BT,
                safe_gate=False,
                disable_recompute=False,
            )

    @pytest.mark.parametrize("safe_gate", [True, False])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_chunk_kda_bwd_intra_accumulates(self, safe_gate, backend):
        self._prep(backend)
        chunk_kda_bwd_intra = _resolve(backend, "linear_attention_kda_chunk_intra", "chunk_kda_bwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        B, T, HV, K, BT = _B, _T, _HV, _K, _BT
        dev = q.device
        torch.manual_seed(3)
        dAqk = torch.zeros(B, T, HV, BT, device=dev, dtype=torch.float32)
        dAkk = torch.zeros(B, T, HV, BT, device=dev, dtype=torch.float32)
        dq_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        dk_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        db_in = torch.randn(B, T, HV, device=dev, dtype=torch.float32)
        dg_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)

        dq, dk, db, dg = chunk_kda_bwd_intra(
            q=q,
            k=k,
            g=gk,
            beta=beta,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq_in.clone(),
            dk=dk_in.clone(),
            db=db_in.clone(),
            dg=dg_in.clone(),
            cu_seqlens=None,
            chunk_size=BT,
            safe_gate=safe_gate,
        )
        # With zero dAqk/dAkk the intra contribution to dq is exactly zero, so
        # dq must equal the incoming accumulator (accumulate, not overwrite).
        ratio_dq = _rms_ratio(dq, dq_in)
        assert ratio_dq < 1e-5, f"dq must equal incoming accumulator, ratio {ratio_dq:.3e}"
        assert dq.shape == dq_in.shape and dk.shape == dk_in.shape
        assert db.shape == db_in.shape and dg.shape == dg_in.shape

    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf(self, framework, record_property):
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        chunk_kda_fwd_intra = _resolve(framework, "linear_attention_kda_chunk_intra", "chunk_kda_fwd_intra")
        # Reconstruct inputs inline (do NOT depend on any saved fla `.pt` fixture).
        q, k, v, beta, gk = _make_inputs("cuda")
        scale = _SCALE
        BT = _BT
        B, T, H, K, HV, V = _B, _T, _H, _K, _HV, _V

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Run the active backend on these exact perf inputs and
        # verify Aqk + the WY-solve outputs (w, u, kg, Akk) against the file's
        # own fp32 torch references BEFORE timing (mirrors test_chunk_kda_fwd_intra).
        # A hard assert here turns a silently-broken kernel (e.g. NaN) into a loud
        # failure instead of a fast-but-garbage benchmark number.
        _w, _u, _qg, _kg, _Aqk, _Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=scale,
            cu_seqlens=None,
            chunk_size=BT,
            safe_gate=False,
            disable_recompute=False,
        )
        self._check_aqk(q, k, gk, scale, BT, _Aqk)
        _rw, _ru, _rkg, _rAkk = _chunk_intra_fwd_wy_torch_ref(k, v, beta, gk, BT)
        for _name, _got, _ref in (("w", _w, _rw), ("u", _u, _ru), ("kg", _kg, _rkg), ("Akk", _Akk, _rAkk)):
            _ratio = _rms_ratio(_got, _ref)
            assert _ratio < 2e-2, (
                f"{framework} {_name} output mismatch before perf (rms_ratio={_ratio:.3e}); "
                f"perf number would be meaningless"
            )
        del _w, _u, _qg, _kg, _Aqk, _Akk, _rw, _ru, _rkg, _rAkk

        framework_fn = lambda beta=beta, gk=gk, k=k, q=q, v=v: chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=scale,
            cu_seqlens=None,
            chunk_size=BT,
            safe_gate=False,
            disable_recompute=False,
        )[0]

        result = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {HV}, {K}, {V})")
        record_property("op", "chunk_kda_fwd_intra")

        del q, k, v, beta, gk, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf_bwd(self, framework, record_property):
        # Time the standalone backward helper directly; forward timing does not call it.
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        chunk_kda_bwd_intra = _resolve(framework, "linear_attention_kda_chunk_intra", "chunk_kda_bwd_intra")
        q, k, v, beta, gk = _make_inputs("cuda")
        B, T, H, K, HV, V = _B, _T, _H, _K, _HV, _V
        BT = _BT
        dev = q.device

        # Fixed-seed upstream/accumulator gradients the kernel reads and adds onto.
        torch.manual_seed(1)
        dAqk = torch.randn(B, T, HV, BT, device=dev, dtype=torch.float32)
        dAkk = torch.randn(B, T, HV, BT, device=dev, dtype=torch.float32)
        dq_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        dk_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        db_in = torch.randn(B, T, HV, device=dev, dtype=torch.float32)
        dg_in = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the accumulated intra-chunk gradients (dq, dk, db, dg)
        # match the file's own fp32 torch reference on these exact perf inputs
        # BEFORE timing (mirrors test_op_chunk_kda_bwd_intra). Uses cloned buffers
        # so the gate does not perturb the accumulators used in the timed path.
        _dq, _dk, _db, _dg = chunk_kda_bwd_intra(
            q=q,
            k=k,
            g=gk,
            beta=beta,
            dAqk=dAqk.clone(),
            dAkk=dAkk.clone(),
            dq=dq_in.clone(),
            dk=dk_in.clone(),
            db=db_in.clone(),
            dg=dg_in.clone(),
            cu_seqlens=None,
            chunk_size=BT,
            safe_gate=False,
        )
        _rdq, _rdk, _rdb, _rdg = _chunk_intra_bwd_torch_ref(q, k, gk, beta, dAqk, dAkk, dq_in, dk_in, db_in, dg_in, BT)
        for _name, _got, _ref in (("dq", _dq, _rdq), ("dk", _dk, _rdk), ("db", _db, _rdb), ("dg", _dg, _rdg)):
            _ratio = _rms_ratio(_got, _ref)
            assert _ratio < 2e-2, (
                f"{framework} {_name} output mismatch before perf (rms_ratio={_ratio:.3e}); "
                f"perf number would be meaningless"
            )
        del _dq, _dk, _db, _dg, _rdq, _rdk, _rdb, _rdg

        # The kernel accumulates onto dq/dk/db/dg in place; reusing the same
        # buffers across reps only grows their (fp32) magnitude, which does not
        # affect kernel runtime, so pass them directly (no clone in the timed path).
        framework_fn = (
            lambda beta=beta,
            dAkk=dAkk,
            dAqk=dAqk,
            db_in=db_in,
            dg_in=dg_in,
            dk_in=dk_in,
            dq_in=dq_in,
            gk=gk,
            k=k,
            q=q: chunk_kda_bwd_intra(
                q=q,
                k=k,
                g=gk,
                beta=beta,
                dAqk=dAqk,
                dAkk=dAkk,
                dq=dq_in,
                dk=dk_in,
                db=db_in,
                dg=dg_in,
                cu_seqlens=None,
                chunk_size=BT,
                safe_gate=False,
            )[0]
        )

        result = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {HV}, {K}, {V})")
        record_property("op", "chunk_kda_bwd_intra")

        del q, k, v, beta, gk, dAqk, dAkk, dq_in, dk_in, db_in, dg_in, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
