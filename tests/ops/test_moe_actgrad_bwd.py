# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

from tilegym.backend import is_backend_available
from tilegym.backend import set_backend
from tilegym.ops import moe_actgrad_bwd

from .. import common

# GLU-style activations use the split ``[gate | up]`` layout with N = 2 * I.
_GLU_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu"})

_backends = ["cutile"]

_PERF_FRAMEWORKS = ["cutile", "pytorch"]

# Frameworks whose GEGLU uses the tanh approximation rather than erf.
_TANH_GEGLU_FRAMEWORKS = ("pytorch",)

_PERF_CASES = [
    (4096, 4096, 1024, 64, 4, "swiglu"),
    # Mixtral 8x22B-like (geglu variant): H=6144, I=16384, E=8, K=2
    (4096, 6144, 16384, 8, 2, "geglu"),
    # DeepSeek-V3 / R1: H=7168, I=2048, E=256, K=8
    (4096, 7168, 2048, 256, 8, "swiglu"),
    # Qwen1.5-MoE-A2.7B-like: H=2048, I=1408, E=60, K=4
    (4096, 2048, 1408, 60, 4, "swiglu"),
]


def _gelu_tanh_with_derivative(x):
    """Return tanh-approximate GELU and its derivative."""
    sqrt_2_over_pi = 0.7978845608028654
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_val = torch.tanh(tanh_arg)
    gelu = 0.5 * x * (1.0 + tanh_val)
    gelu_prime = 0.5 * (1.0 + tanh_val) + (
        0.5 * x * (1.0 - tanh_val * tanh_val) * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x)
    )
    return gelu, gelu_prime


def _is_cutile_unsupported_error(exc: BaseException) -> bool:
    """Detect cuTile compile failures that indicate the current GPU/toolchain
    combination cannot run the kernel (e.g. cuda-tile installed without support
    for the current SM arch)."""
    msg = str(exc)
    return "Cannot find option named 'sm_" in msg or "No valid config found in search space" in msg


def _build_routing_tensors(T: int, E: int, K: int, device: str, seed: int = 42):
    """Build routing tensors for the MoE grouped layout."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    topk_indices = torch.randint(0, E, (T, K), device=device, dtype=torch.int32, generator=gen)
    flat_indices = topk_indices.view(-1)
    expert_freq = flat_indices.bincount(minlength=E).int()
    expert_frequency_offset = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            expert_freq.cumsum(0).int(),
        ]
    )
    s_scatter_idx = flat_indices.argsort().int()
    x_gather_idx = (s_scatter_idx // K).int()
    topk_scores = (
        torch.softmax(torch.randn(T, K, device=device, dtype=torch.float32, generator=gen), dim=-1)
        .flatten()
        .contiguous()
    )
    return expert_frequency_offset, x_gather_idx, s_scatter_idx, topk_scores


def _make_test_inputs(T, H, I, E, K, activation_type, dtype, device, seed=42):
    expert_frequency_offset, x_gather_idx, s_scatter_idx, topk_scores = _build_routing_tensors(
        T, E, K, device, seed=seed
    )
    total_tokens = T * K
    is_glu = activation_type in _GLU_ACTIVATIONS

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    w2 = torch.randn(E, I, H, dtype=dtype, device=device, generator=gen) * 0.1
    dout = torch.randn(T, H, dtype=dtype, device=device, generator=gen)
    h = torch.randn(total_tokens, I * (2 if is_glu else 1), dtype=dtype, device=device, generator=gen)

    return {
        "dout": dout,
        "w2": w2,
        "h": h,
        "topk_scores": topk_scores,
        "s_scatter_idx": s_scatter_idx,
        "expert_frequency_offset": expert_frequency_offset,
        "x_gather_idx": x_gather_idx,
    }


def pytorch_reference(
    dout,
    h,
    w2,
    topk_scores,
    s_scatter_idx,
    expert_frequency_offset,
    x_gather_idx,
    activation_type,
    geglu_tanh_approx=False,
):
    """
    Pure PyTorch correctness reference for moe_actgrad_bwd.

    Mirrors the kernel's computation:
      1. dy1     = dout[x_gather_idx] @ w2[e]^T     (grouped GEMM)
      2. dh      = act_backward(h, dy1) * s          (activation bwd + score scale)
      3. a_prime = act_forward(h) * s                (score-weighted fwd output)
      4. ds      = (dy1 * act_forward(h)).sum(dim=1) (routing-score gradient)

    activation_type is the lowercase activation name. ``geglu_tanh_approx``
    selects the tanh GeGLU approximation for cross-framework correctness only;
    the default preserves the exact-erf reference.
    """
    E = w2.shape[0]
    I = w2.shape[1]
    TK = h.shape[0]
    dtype = h.dtype
    device = h.device

    is_glu = activation_type in _GLU_ACTIVATIONS
    actual_n_dim = I * 2 if is_glu else I

    # Gather scores: s[i] = topk_scores[s_scatter_idx[i]]
    s = topk_scores[s_scatter_idx].float()  # [TK]

    dh = torch.zeros(TK, actual_n_dim, device=device, dtype=dtype)
    a_prime = torch.zeros(TK, I, device=device, dtype=dtype)
    ds = torch.zeros(TK, device=device, dtype=torch.float32)

    for e in range(E):
        e_start = expert_frequency_offset[e].item()
        e_end = expert_frequency_offset[e + 1].item()
        if e_start >= e_end:
            continue

        m_idx = torch.arange(e_start, e_end, device=device)  # [nm]
        orig = x_gather_idx[m_idx]  # [nm]
        dout_e = dout[orig].float()  # [nm, H]
        w2_e = w2[e].float()  # [I, H]
        dy1 = dout_e @ w2_e.t()  # [nm, I]
        s_e = s[m_idx].unsqueeze(1)  # [nm, 1]

        if is_glu:
            g = h[m_idx, :I].float()
            u = h[m_idx, I:].float()

            if activation_type == "swiglu":
                sig_g = torch.sigmoid(g)
                silu_g = g * sig_g
                fwd = silu_g * u
                dg = dy1 * u * (sig_g * (1.0 + g * (1.0 - sig_g)))
                du = dy1 * silu_g
            elif activation_type == "geglu":
                if geglu_tanh_approx:
                    gelu_g, gelu_prime = _gelu_tanh_with_derivative(g)
                else:
                    INV_SQRT_2 = 0.7071067811865475
                    INV_SQRT_2_PI = 0.3989422804014327
                    erf_val = torch.erf(g * INV_SQRT_2)
                    gelu_g = 0.5 * g * (1.0 + erf_val)
                    pdf = INV_SQRT_2_PI * torch.exp(-0.5 * g * g)
                    gelu_prime = 0.5 * (1.0 + erf_val) + g * pdf
                fwd = gelu_g * u
                dg = dy1 * u * gelu_prime
                du = dy1 * gelu_g
            else:  # REGLU
                relu_g = torch.clamp(g, min=0.0)
                fwd = relu_g * u
                dg = dy1 * u * (g > 0).float()
                du = dy1 * relu_g

            ds[m_idx] = (dy1 * fwd).sum(dim=1)
            dh[m_idx, :I] = (dg * s_e).to(dtype)
            dh[m_idx, I:] = (du * s_e).to(dtype)
            a_prime[m_idx] = (fwd * s_e).to(dtype)

        else:
            h_e = h[m_idx].float()

            if activation_type == "silu":
                sig_z = torch.sigmoid(h_e)
                fwd = h_e * sig_z
                dh_r = dy1 * (sig_z * (1.0 + h_e * (1.0 - sig_z)))
            elif activation_type == "relu":
                fwd = torch.clamp(h_e, min=0.0)
                dh_r = dy1 * (h_e > 0).float()
            elif activation_type == "gelu":
                INV_SQRT_2 = 0.7071067811865475
                INV_SQRT_2_PI = 0.3989422804014327
                erf_val = torch.erf(h_e * INV_SQRT_2)
                fwd = 0.5 * h_e * (1.0 + erf_val)
                pdf = INV_SQRT_2_PI * torch.exp(-0.5 * h_e * h_e)
                dh_r = dy1 * (0.5 * (1.0 + erf_val) + h_e * pdf)
            else:  # RELU_SQ
                relu_h = torch.clamp(h_e, min=0.0)
                fwd = relu_h * relu_h
                dh_r = dy1 * 2.0 * relu_h

            ds[m_idx] = (dy1 * fwd).sum(dim=1)
            dh[m_idx] = (dh_r * s_e).to(dtype)
            a_prime[m_idx] = (fwd * s_e).to(dtype)

    return dh, a_prime, ds


def _alloc_outputs(h: torch.Tensor, I: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-allocate (dh, ds, a_prime) for the new in-place API.

    The kernel accumulates the partial-N routing-score reduction into
    ds, so callers must provide a zero-initialized buffer.
    """
    TK, n = h.shape
    dh = torch.zeros(TK, n, device=h.device, dtype=h.dtype)
    ds = torch.zeros(TK, device=h.device, dtype=torch.float32)
    a_prime = torch.zeros(TK, I, device=h.device, dtype=h.dtype)
    return dh, ds, a_prime


def _call_moe_actgrad_bwd(inputs, activation_type_str: str, max_tokens_per_expert=None):
    """Wrapper for the new in-place API; allocates outputs and returns them."""
    h = inputs["h"]
    w2 = inputs["w2"]
    I = w2.shape[1]
    dh, ds, a_prime = _alloc_outputs(h, I)
    moe_actgrad_bwd(
        dout=inputs["dout"],
        h=h,
        w2=w2,
        dh=dh,
        ds=ds,
        b2=None,
        db2=None,
        a_prime=a_prime,
        topk_scores=inputs["topk_scores"],
        expert_frequency_offset=inputs["expert_frequency_offset"],
        x_gather_idx=inputs["x_gather_idx"],
        s_scatter_idx=inputs["s_scatter_idx"],
        activation_type=activation_type_str,
        max_tokens_per_expert=max_tokens_per_expert,
    )
    return dh, ds, a_prime


def _make_moe_actgrad_bwd_fn(inputs, activation_type_str: str, max_tokens_per_expert=None, zero_ds: bool = False):
    """Build a low-overhead reusable-output callable for perf timing."""
    h = inputs["h"]
    w2 = inputs["w2"]
    I = w2.shape[1]
    TK, n = h.shape
    dh = torch.empty(TK, n, device=h.device, dtype=h.dtype)
    ds = torch.empty(TK, device=h.device, dtype=torch.float32)
    a_prime = torch.empty(TK, I, device=h.device, dtype=h.dtype)

    def fn():
        if zero_ds:
            ds.zero_()
        moe_actgrad_bwd(
            dout=inputs["dout"],
            h=h,
            w2=w2,
            dh=dh,
            ds=ds,
            b2=None,
            db2=None,
            a_prime=a_prime,
            topk_scores=inputs["topk_scores"],
            expert_frequency_offset=inputs["expert_frequency_offset"],
            x_gather_idx=inputs["x_gather_idx"],
            s_scatter_idx=inputs["s_scatter_idx"],
            activation_type=activation_type_str,
            max_tokens_per_expert=max_tokens_per_expert,
        )
        return dh, ds, a_prime

    return fn


def _make_pytorch_grouped_mm_fn(inputs, activation_type_str: str):
    """Build the compiled BF16 PyTorch performance baseline.

    Setup-only layout conversion happens here, outside the timed callable.
    The callable keeps routing gathers in scope, uses one grouped GEMM for all
    experts, and lets Inductor fuse the full-tensor activation epilogue. Its
    first invocation compiles and warms the graph before benchmark capture.
    """
    h = inputs["h"]
    w2 = inputs["w2"]
    if h.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        pytest.skip("torch._grouped_mm performance baseline requires bfloat16 inputs")
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("installed PyTorch does not provide torch._grouped_mm")
    if not hasattr(torch, "compile"):
        pytest.skip("installed PyTorch does not provide torch.compile")
    capability = torch.cuda.get_device_capability(h.device)
    if capability[0] not in (9, 10):
        pytest.skip(f"torch._grouped_mm CUDA Graph fast path requires SM9x/SM10x, got sm{capability[0]}{capability[1]}")

    I = w2.shape[1]
    activation_type = activation_type_str
    is_glu = activation_type in _GLU_ACTIVATIONS

    # torch._grouped_mm consumes cumulative expert end offsets and [E, H, I]
    # weights. Keep this one-time layout conversion out of the measured
    # operator invocation.
    expert_end_offsets = inputs["expert_frequency_offset"][1:].clone()
    w2_transposed = w2.transpose(1, 2).contiguous()

    # Functional outputs avoid non-contiguous out= views and expose the whole
    # post-GEMM graph to Inductor fusion.
    def functional_core(
        dout,
        h,
        w2_transposed,
        topk_scores,
        s_scatter_idx,
        x_gather_idx,
        expert_end_offsets,
    ):
        scores_column = torch.index_select(topk_scores, 0, s_scatter_idx).unsqueeze(1)
        dout_grouped = torch.index_select(dout, 0, x_gather_idx)
        dy1 = torch._grouped_mm(dout_grouped, w2_transposed, offs=expert_end_offsets).float()

        if is_glu:
            g = h[:, :I].float()
            u = h[:, I:].float()

            if activation_type == "swiglu":
                sig_g = torch.sigmoid(g)
                activated_g = g * sig_g
                fwd = activated_g * u
                dg = dy1 * u * (sig_g * (1.0 + g * (1.0 - sig_g)))
                du = dy1 * activated_g
            elif activation_type == "geglu":
                activated_g, gelu_prime = _gelu_tanh_with_derivative(g)
                fwd = activated_g * u
                dg = dy1 * u * gelu_prime
                du = dy1 * activated_g
            else:  # REGLU
                activated_g = torch.clamp(g, min=0.0)
                fwd = activated_g * u
                dg = dy1 * u * (g > 0).float()
                du = dy1 * activated_g

            dh = torch.cat((dg * scores_column, du * scores_column), dim=1).to(h.dtype)
        else:
            h_float = h.float()

            if activation_type == "silu":
                sig_h = torch.sigmoid(h_float)
                fwd = h_float * sig_h
                dh_unscaled = dy1 * (sig_h * (1.0 + h_float * (1.0 - sig_h)))
            elif activation_type == "relu":
                fwd = torch.clamp(h_float, min=0.0)
                dh_unscaled = dy1 * (h_float > 0).float()
            elif activation_type == "gelu":
                inv_sqrt_2 = 0.7071067811865475
                inv_sqrt_2_pi = 0.3989422804014327
                erf_val = torch.erf(h_float * inv_sqrt_2)
                fwd = 0.5 * h_float * (1.0 + erf_val)
                pdf = inv_sqrt_2_pi * torch.exp(-0.5 * h_float * h_float)
                dh_unscaled = dy1 * (0.5 * (1.0 + erf_val) + h_float * pdf)
            else:  # RELU_SQ
                relu_h = torch.clamp(h_float, min=0.0)
                fwd = relu_h * relu_h
                dh_unscaled = dy1 * 2.0 * relu_h

            dh = (dh_unscaled * scores_column).to(h.dtype)

        a_prime = (fwd * scores_column).to(h.dtype)
        ds = torch.sum(dy1 * fwd, dim=1)
        return dh, ds, a_prime

    compiled_core = torch.compile(
        functional_core,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )

    def fn():
        return compiled_core(
            inputs["dout"],
            h,
            w2_transposed,
            inputs["topk_scores"],
            inputs["s_scatter_idx"],
            inputs["x_gather_idx"],
            expert_end_offsets,
        )

    return fn


def _make_perf_framework_fn(framework: str, inputs, activation_type_str: str, dtype, device: str):
    """Create a prepared callable for fair perf timing.

    The returned function excludes setup and layout conversion. Output storage
    is either preallocated or fixed during CUDA Graph capture, so replay times
    the same operator-work boundary for every framework.
    """
    efo = inputs["expert_frequency_offset"]
    if framework in _backends:
        if not is_backend_available(framework):
            pytest.skip(f"{framework} backend not available")
        set_backend(framework)
        max_tokens_per_expert = int((efo[1:] - efo[:-1]).max().item())
        return _make_moe_actgrad_bwd_fn(
            inputs,
            activation_type_str,
            max_tokens_per_expert=max_tokens_per_expert,
            zero_ds=framework in _backends,
        )
    if framework == "pytorch":
        # Keep the framework name stable; the timed implementation is compiled grouped-mm.
        return _make_pytorch_grouped_mm_fn(inputs, activation_type_str)
    raise ValueError(f"Unknown framework: {framework}")


def _assert_perf_correctness(framework: str, outputs, inputs, activation_type):
    """Compare a prepared performance callable against the FP32 oracle."""
    dh, ds, a_prime = outputs
    use_tanh_geglu = framework in _TANH_GEGLU_FRAMEWORKS and activation_type == "geglu"
    dh_ref, a_prime_ref, ds_ref = pytorch_reference(
        dout=inputs["dout"],
        h=inputs["h"],
        w2=inputs["w2"],
        topk_scores=inputs["topk_scores"],
        s_scatter_idx=inputs["s_scatter_idx"],
        expert_frequency_offset=inputs["expert_frequency_offset"],
        x_gather_idx=inputs["x_gather_idx"],
        activation_type=activation_type,
        geglu_tanh_approx=use_tanh_geglu,
    )
    torch.testing.assert_close(dh.float(), dh_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(a_prime.float(), a_prime_ref.float(), atol=1e-2, rtol=1e-2)

    # torch._grouped_mm materializes BF16 dy1 before the FP32 reduction, while
    # the oracle keeps dy1 in FP32. ds therefore needs a reduction-aware
    # absolute tolerance. Other paths retain 1e-2/1e-2.
    if framework == "pytorch":
        reduction_size = inputs["w2"].shape[1]
        ds_tolerance = {"atol": 0.05 * reduction_size**0.5, "rtol": 2e-2}
    else:
        ds_tolerance = {"atol": 1e-2, "rtol": 1e-2}
    torch.testing.assert_close(ds.float(), ds_ref.float(), **ds_tolerance)


def _add_fair_kernel_time_metrics(result, framework: str):
    """Annotate benchmark output with fair all-kernel GPU timing metadata."""
    forward = result[framework].get("forward")
    if not forward:
        return

    event_time_ms = forward.get("median")
    forward["event_time_ms"] = event_time_ms
    forward["gpu_kernel_time_ms"] = event_time_ms
    forward["fair_kernel_time_ms"] = event_time_ms
    forward["fair_kernel_time_source"] = "cuda_event_cudagraph_prepared_callable_median"
    forward["fair_kernel_time_scope"] = (
        "one prepared framework_fn invocation with setup/compilation/layout conversion and allocator overhead excluded"
    )

    kernel_times = forward.get("kernel_times") or []
    if not kernel_times:
        return

    profiled_gpu_kernel_time_ms = sum(item.get("self_time_us", 0.0) for item in kernel_times) / 1000.0
    dominant_kernel = kernel_times[0]

    forward["profiled_gpu_kernel_time_ms"] = profiled_gpu_kernel_time_ms
    forward["profiled_gpu_kernel_time_source"] = "torch.profiler_cuda_self_time_sum"
    forward["dominant_kernel_time_ms"] = dominant_kernel.get("self_time_us", 0.0) / 1000.0
    forward["dominant_kernel_name"] = dominant_kernel.get("name")
    forward["kernel_launch_count"] = sum(item.get("count", 0) for item in kernel_times)


class Test_MoEActgradBwd(common.PyTestCase):
    @pytest.mark.parametrize(
        "T, H, I, E, K, activation_type",
        [
            # ── Activation type sweep (T=4096, H=4096, I=1024, E=64, K=4) ────
            (4096, 4096, 1024, 64, 4, "swiglu"),
            (4096, 4096, 1024, 64, 4, "geglu"),
            (4096, 4096, 1024, 64, 4, "reglu"),
            (4096, 4096, 1024, 64, 4, "silu"),
            (4096, 4096, 1024, 64, 4, "relu"),
            (4096, 4096, 1024, 64, 4, "gelu"),
            (4096, 4096, 1024, 64, 4, "relu_sq"),
            # ── Token count sweep (swiglu, H=4096, I=1024, E=64, K=4) ────────
            (512, 4096, 1024, 64, 4, "swiglu"),
            (2048, 4096, 1024, 64, 4, "swiglu"),
            # ── Top-K sweep (swiglu, T=4096, H=4096, I=1024, E=64) ───────────
            (4096, 4096, 1024, 64, 1, "swiglu"),
            (4096, 4096, 1024, 64, 2, "swiglu"),
            # ── Real-world model shapes ───────────────────────────────────────
            # Mixtral-8x7B-like: H=4096, I=14336, E=8, K=2 (reduced T for CI)
            (512, 4096, 14336, 8, 2, "swiglu"),
            # Qwen1.5-MoE-A2.7B-like: H=2048, I=1408, E=60, K=4
            (4096, 2048, 1408, 60, 4, "swiglu"),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, T, H, I, E, K, activation_type, dtype, backend, arch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend {backend} is not supported: {e}")

        self.setUp()
        device = "cuda"

        inputs = _make_test_inputs(T, H, I, E, K, activation_type, dtype, device)

        try:
            dh, ds, a_prime = _call_moe_actgrad_bwd(inputs, activation_type)
        except Exception as e:
            if _is_cutile_unsupported_error(e):
                pytest.skip(f"cuTile backend cannot run on current GPU/toolchain: {e}")
            raise

        dh_ref, a_prime_ref, ds_ref = pytorch_reference(
            dout=inputs["dout"],
            h=inputs["h"],
            w2=inputs["w2"],
            topk_scores=inputs["topk_scores"],
            s_scatter_idx=inputs["s_scatter_idx"],
            expert_frequency_offset=inputs["expert_frequency_offset"],
            x_gather_idx=inputs["x_gather_idx"],
            activation_type=activation_type,
        )

        torch.testing.assert_close(dh.float(), dh_ref.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(a_prime.float(), a_prime_ref.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ds.float(), ds_ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "T, H, I, E, K, activation_type",
        _PERF_CASES,
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("framework", _PERF_FRAMEWORKS)
    def test_perf(self, T, H, I, E, K, activation_type, dtype, framework, record_property):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.setUp()
        device = "cuda"

        inputs = _make_test_inputs(T, H, I, E, K, activation_type, dtype, device)
        framework_fn = _make_perf_framework_fn(framework, inputs, activation_type, dtype, device)
        if framework == "pytorch":
            record_property("pytorch_backend", "torch.compile(inductor)+torch._grouped_mm")
        if activation_type == "geglu":
            geglu_approximation = "tanh" if framework in _TANH_GEGLU_FRAMEWORKS else "erf"
            record_property("geglu_approximation", geglu_approximation)

        try:
            framework_outputs = framework_fn()
        except Exception as e:
            if _is_cutile_unsupported_error(e):
                pytest.skip(f"cuTile backend cannot run on current GPU/toolchain: {e}")
            raise

        _assert_perf_correctness(framework, framework_outputs, inputs, activation_type)
        del framework_outputs

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            result = common.benchmark_framework(
                framework,
                framework_fn,
                mode="forward",
                use_cudagraph=True,
                use_cupti=False,
            )
        except Exception as e:
            if _is_cutile_unsupported_error(e):
                pytest.skip(f"cuTile backend cannot run on current GPU/toolchain: {e}")
            raise
        _add_fair_kernel_time_metrics(result, framework)
        record_property("benchmark", result)

        torch.cuda.synchronize()
        del framework_fn, inputs
        gc.collect()
        torch.cuda.empty_cache()
