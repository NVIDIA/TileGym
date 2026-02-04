# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Performance benchmarks for Flash Attention backward pass.

Compares cuTile implementation against:
- PyTorch SDPA Flash backend
- PyTorch SDPA Memory Efficient backend
- PyTorch SDPA Math backend
- Triton Flash Attention

Usage:
    python tests/benchmark/bench_attention_backward.py

Or with pytest:
    python -m pytest tests/benchmark/bench_attention_backward.py -v -s
"""

import math
import sys
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import triton
    import triton.testing
    HAS_TRITON_TESTING = True
except ImportError:
    HAS_TRITON_TESTING = False

# Check for triton flash attention
TRITON_FUSED_ATTN_PATH = "/root/triton-fused-attention.py"
HAS_TRITON_FLASH = os.path.exists(TRITON_FUSED_ATTN_PATH)

# Import triton flash attention if available
if HAS_TRITON_FLASH:
    import importlib.util
    spec = importlib.util.spec_from_file_location("triton_fused_attention", TRITON_FUSED_ATTN_PATH)
    triton_fused_attention = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(triton_fused_attention)
        triton_attention = triton_fused_attention.attention
    except Exception as e:
        print(f"Warning: Could not load triton flash attention: {e}")
        HAS_TRITON_FLASH = False
        triton_attention = None

# Import cuTile implementation
from tilegym.backend import set_backend
from tilegym.ops.cutile.attention import tile_fmha_with_backward


def benchmark_fn(fn, warmup=10, rep=100):
    """Simple benchmark function without triton.testing dependency."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / rep
    return elapsed_ms


def compute_flops(batch, heads, seq_len, head_dim, causal, mode="fwd"):
    """Compute FLOPs for attention operation."""
    # Forward: 2 matmuls (QK^T and PV) of size [seq, seq] x [seq, head_dim]
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul  # QK^T and PV

    if causal:
        total_flops *= 0.5  # Only half the computation due to triangular mask

    if mode == "bwd":
        # Backward: roughly 2.5x forward (2x for gradients + 0.5x for recomputation)
        total_flops *= 2.5
    elif mode == "fwd+bwd":
        total_flops *= 3.5  # Forward + backward

    return total_flops


def benchmark_pytorch_sdpa(batch, heads, seq_len, head_dim, dtype, causal, mode="fwd+bwd", backend=None):
    """
    Benchmark PyTorch scaled_dot_product_attention with specific backend.

    Args:
        backend: One of "flash", "mem_efficient", "math", or None (auto)
    """
    device = "cuda"

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Map backend string to SDPBackend enum
    backend_map = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }

    def fwd():
        if backend is not None and backend in backend_map:
            with sdpa_kernel(backend_map[backend]):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, scale=sm_scale
                )
        else:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=sm_scale
            )

    def fwd_bwd():
        # Reset gradients
        if q.grad is not None:
            q.grad = None
            k.grad = None
            v.grad = None
        o = fwd()
        do = torch.randn_like(o)
        o.backward(do)
        return o

    if mode == "fwd":
        fn = fwd
    else:
        fn = fwd_bwd

    try:
        ms = benchmark_fn(fn)
        flops = compute_flops(batch, heads, seq_len, head_dim, causal, mode)
        tflops = flops * 1e-12 / (ms * 1e-3)
        return ms, tflops
    except Exception as e:
        # Backend may not support this configuration
        return None, None


def benchmark_cutile(batch, heads, seq_len, head_dim, dtype, causal, mode="fwd+bwd"):
    """Benchmark cuTile Flash Attention."""
    device = "cuda"

    try:
        set_backend("cutile")
    except Exception:
        return None, None

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

    sm_scale = 1.0 / math.sqrt(head_dim)

    def fwd():
        return tile_fmha_with_backward(q, k, v, scaling=sm_scale, is_causal=causal)

    def fwd_bwd():
        # Reset gradients
        if q.grad is not None:
            q.grad = None
            k.grad = None
            v.grad = None
        o = fwd()
        do = torch.randn_like(o)
        o.backward(do)
        return o

    if mode == "fwd":
        fn = fwd
    else:
        fn = fwd_bwd

    try:
        ms = benchmark_fn(fn)
        flops = compute_flops(batch, heads, seq_len, head_dim, causal, mode)
        tflops = flops * 1e-12 / (ms * 1e-3)
        return ms, tflops
    except Exception as e:
        print(f"cuTile benchmark failed: {e}")
        return None, None


def benchmark_triton(batch, heads, seq_len, head_dim, dtype, causal, mode="fwd+bwd"):
    """Benchmark Triton Flash Attention."""
    if not HAS_TRITON_FLASH or triton_attention is None:
        return None, None

    device = "cuda"

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

    sm_scale = 1.0 / math.sqrt(head_dim)

    def fwd():
        return triton_attention(q, k, v, causal, sm_scale)

    def fwd_bwd():
        # Reset gradients
        if q.grad is not None:
            q.grad = None
            k.grad = None
            v.grad = None
        o = fwd()
        do = torch.randn_like(o)
        o.backward(do)
        return o

    if mode == "fwd":
        fn = fwd
    else:
        fn = fwd_bwd

    try:
        ms = benchmark_fn(fn)
        flops = compute_flops(batch, heads, seq_len, head_dim, causal, mode)
        tflops = flops * 1e-12 / (ms * 1e-3)
        return ms, tflops
    except Exception as e:
        print(f"Triton benchmark failed: {e}")
        return None, None


def format_result(ms, tflops):
    """Format benchmark result for display."""
    if ms is None:
        return "N/A"
    return f"{ms:.2f} / {tflops:.1f}"


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks comparing all backends."""
    print("=" * 120)
    print("Flash Attention Backward Benchmark - Comprehensive Comparison")
    print("=" * 120)
    print()
    print("Backends: cuTile, PyTorch SDPA (Flash, MemEfficient, Math), Triton")
    print("Mode: Forward + Backward pass")
    print("Units: ms (milliseconds) / TFLOPS")
    print()

    # Production-relevant configurations
    configs = [
        # (batch, heads, seq_len, head_dim, dtype, causal)
        # ======== head_dim=128 (Triton OOM on RTX 5070 Ti) ========
        # Standard configs
        (1, 32, 512, 128, torch.float16, True),
        (1, 32, 1024, 128, torch.float16, True),
        (1, 32, 2048, 128, torch.float16, True),
        (1, 32, 4096, 128, torch.float16, True),
        # Larger batch
        (4, 32, 512, 128, torch.float16, True),
        (4, 32, 1024, 128, torch.float16, True),
        # bfloat16
        (1, 32, 1024, 128, torch.bfloat16, True),
        (4, 32, 1024, 128, torch.bfloat16, True),
        # Non-causal
        (1, 32, 1024, 128, torch.float16, False),

        # ======== head_dim=64 (Triton works) ========
        # Small sequence lengths
        (1, 32, 256, 64, torch.float16, True),
        (1, 32, 512, 64, torch.float16, True),
        # Medium sequence lengths
        (1, 32, 1024, 64, torch.float16, True),
        (1, 32, 2048, 64, torch.float16, True),
        (1, 32, 4096, 64, torch.float16, True),
        # Larger batch sizes
        (4, 32, 512, 64, torch.float16, True),
        (4, 32, 1024, 64, torch.float16, True),
        (8, 32, 512, 64, torch.float16, True),
        # bfloat16 with head_dim=64
        (1, 32, 1024, 64, torch.bfloat16, True),
        (4, 32, 1024, 64, torch.bfloat16, True),
        # Non-causal with head_dim=64
        (1, 32, 1024, 64, torch.float16, False),
        (1, 32, 2048, 64, torch.float16, False),
        # Fewer heads (more parallelism per head)
        (1, 16, 2048, 64, torch.float16, True),
        (1, 8, 4096, 64, torch.float16, True),
    ]

    # Print header
    header = f"{'Config':<35} {'cuTile':<14} {'SDPA-Flash':<14} {'SDPA-MemEff':<14} {'SDPA-Math':<14} {'Triton':<14}"
    print(header)
    print("-" * 120)

    results = []

    for batch, heads, seq_len, head_dim, dtype, causal in configs:
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        config_str = f"B={batch} H={heads} S={seq_len} D={head_dim} {dtype_str} {'causal' if causal else 'full'}"

        # Benchmark each implementation
        cutile_ms, cutile_tflops = benchmark_cutile(batch, heads, seq_len, head_dim, dtype, causal)
        flash_ms, flash_tflops = benchmark_pytorch_sdpa(batch, heads, seq_len, head_dim, dtype, causal, backend="flash")
        memeff_ms, memeff_tflops = benchmark_pytorch_sdpa(batch, heads, seq_len, head_dim, dtype, causal, backend="mem_efficient")
        math_ms, math_tflops = benchmark_pytorch_sdpa(batch, heads, seq_len, head_dim, dtype, causal, backend="math")
        triton_ms, triton_tflops = benchmark_triton(batch, heads, seq_len, head_dim, dtype, causal)

        # Store results
        results.append({
            'config': config_str,
            'cutile': (cutile_ms, cutile_tflops),
            'flash': (flash_ms, flash_tflops),
            'memeff': (memeff_ms, memeff_tflops),
            'math': (math_ms, math_tflops),
            'triton': (triton_ms, triton_tflops),
        })

        # Print results
        row = f"{config_str:<35} {format_result(cutile_ms, cutile_tflops):<14} "
        row += f"{format_result(flash_ms, flash_tflops):<14} "
        row += f"{format_result(memeff_ms, memeff_tflops):<14} "
        row += f"{format_result(math_ms, math_tflops):<14} "
        row += f"{format_result(triton_ms, triton_tflops):<14}"
        print(row)

    print()
    print("=" * 120)

    # Print summary
    print("\nSummary (Average TFLOPS across valid benchmarks):")
    print("-" * 60)

    def avg_tflops(backend_key):
        valid = [r[backend_key][1] for r in results if r[backend_key][1] is not None]
        return sum(valid) / len(valid) if valid else 0

    print(f"  cuTile:           {avg_tflops('cutile'):.1f} TFLOPS")
    print(f"  PyTorch SDPA Flash:     {avg_tflops('flash'):.1f} TFLOPS")
    print(f"  PyTorch SDPA MemEff:    {avg_tflops('memeff'):.1f} TFLOPS")
    print(f"  PyTorch SDPA Math:      {avg_tflops('math'):.1f} TFLOPS")
    print(f"  Triton:           {avg_tflops('triton'):.1f} TFLOPS")

    return results


def run_backward_only_benchmarks():
    """Run benchmarks for backward pass only."""
    print()
    print("=" * 120)
    print("Flash Attention Backward-Only Benchmark")
    print("=" * 120)
    print()

    configs = [
        (1, 32, 512, 128, torch.float16, True),
        (1, 32, 1024, 128, torch.float16, True),
        (1, 32, 2048, 128, torch.float16, True),
        (4, 32, 1024, 128, torch.float16, True),
    ]

    header = f"{'Config':<35} {'cuTile':<14} {'SDPA-Flash':<14} {'Triton':<14}"
    print(header)
    print("-" * 80)

    for batch, heads, seq_len, head_dim, dtype, causal in configs:
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        config_str = f"B={batch} H={heads} S={seq_len} D={head_dim} {dtype_str} {'causal' if causal else 'full'}"

        device = "cuda"
        sm_scale = 1.0 / math.sqrt(head_dim)

        # cuTile backward only
        cutile_ms, cutile_tflops = None, None
        try:
            set_backend("cutile")
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

            o = tile_fmha_with_backward(q, k, v, scaling=sm_scale, is_causal=causal)
            do = torch.randn_like(o)

            def bwd_only():
                if q.grad is not None:
                    q.grad = None
                    k.grad = None
                    v.grad = None
                o.backward(do, retain_graph=True)

            cutile_ms = benchmark_fn(bwd_only)
            flops = compute_flops(batch, heads, seq_len, head_dim, causal, "bwd")
            cutile_tflops = flops * 1e-12 / (cutile_ms * 1e-3)
        except Exception as e:
            pass

        # PyTorch SDPA Flash backward only
        flash_ms, flash_tflops = None, None
        try:
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
            do = torch.randn_like(o)

            def bwd_only_flash():
                if q.grad is not None:
                    q.grad = None
                    k.grad = None
                    v.grad = None
                o.backward(do, retain_graph=True)

            flash_ms = benchmark_fn(bwd_only_flash)
            flops = compute_flops(batch, heads, seq_len, head_dim, causal, "bwd")
            flash_tflops = flops * 1e-12 / (flash_ms * 1e-3)
        except Exception as e:
            pass

        # Triton backward only
        triton_ms, triton_tflops = None, None
        if HAS_TRITON_FLASH and triton_attention is not None:
            try:
                q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
                k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
                v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

                o = triton_attention(q, k, v, causal, sm_scale)
                do = torch.randn_like(o)

                def bwd_only_triton():
                    if q.grad is not None:
                        q.grad = None
                        k.grad = None
                        v.grad = None
                    o.backward(do, retain_graph=True)

                triton_ms = benchmark_fn(bwd_only_triton)
                flops = compute_flops(batch, heads, seq_len, head_dim, causal, "bwd")
                triton_tflops = flops * 1e-12 / (triton_ms * 1e-3)
            except Exception as e:
                pass

        row = f"{config_str:<35} {format_result(cutile_ms, cutile_tflops):<14} "
        row += f"{format_result(flash_ms, flash_tflops):<14} "
        row += f"{format_result(triton_ms, triton_tflops):<14}"
        print(row)

    print()


def run_gqa_benchmarks():
    """Run benchmarks for GQA (Grouped Query Attention) configurations."""
    print()
    print("=" * 120)
    print("Flash Attention GQA (Grouped Query Attention) Benchmark")
    print("=" * 120)
    print()
    print("Note: GQA has different number of query heads vs KV heads")
    print()

    # GQA configurations: (batch, q_heads, kv_heads, seq_len, head_dim, dtype, causal)
    configs = [
        # ======== head_dim=128 ========
        # Llama-style GQA (8:1 ratio)
        (1, 32, 4, 512, 128, torch.float16, True),
        (1, 32, 4, 1024, 128, torch.float16, True),
        (1, 32, 4, 2048, 128, torch.float16, True),
        # 4:1 ratio
        (1, 16, 4, 1024, 128, torch.float16, True),
        # 2:1 ratio
        (1, 8, 4, 1024, 128, torch.float16, True),
        # MQA (all query heads share 1 KV head)
        (1, 8, 1, 1024, 128, torch.float16, True),
        (1, 16, 1, 1024, 128, torch.float16, True),

        # ======== head_dim=64 (better for Triton comparison) ========
        # Llama-style GQA (8:1 ratio)
        (1, 32, 4, 512, 64, torch.float16, True),
        (1, 32, 4, 1024, 64, torch.float16, True),
        (1, 32, 4, 2048, 64, torch.float16, True),
        (1, 32, 4, 4096, 64, torch.float16, True),
        # 4:1 ratio
        (1, 16, 4, 1024, 64, torch.float16, True),
        (1, 16, 4, 2048, 64, torch.float16, True),
        # 2:1 ratio
        (1, 8, 4, 1024, 64, torch.float16, True),
        (1, 8, 4, 2048, 64, torch.float16, True),
        # MQA (all query heads share 1 KV head)
        (1, 8, 1, 1024, 64, torch.float16, True),
        (1, 16, 1, 1024, 64, torch.float16, True),
        (1, 32, 1, 1024, 64, torch.float16, True),
        # Larger batches for GQA
        (4, 32, 4, 1024, 64, torch.float16, True),
        (4, 16, 4, 1024, 64, torch.float16, True),
    ]

    header = f"{'Config (B, Hq, Hkv, S, D)':<40} {'cuTile (ms/TFLOPS)':<20}"
    print(header)
    print("-" * 60)

    for batch, q_heads, kv_heads, seq_len, head_dim, dtype, causal in configs:
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        config_str = f"B={batch} Hq={q_heads} Hkv={kv_heads} S={seq_len} D={head_dim}"

        device = "cuda"
        sm_scale = 1.0 / math.sqrt(head_dim)

        cutile_ms, cutile_tflops = None, None
        try:
            set_backend("cutile")
            q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)

            def fwd_bwd():
                if q.grad is not None:
                    q.grad = None
                    k.grad = None
                    v.grad = None
                o = tile_fmha_with_backward(q, k, v, scaling=sm_scale, is_causal=causal)
                do = torch.randn_like(o)
                o.backward(do)
                return o

            cutile_ms = benchmark_fn(fwd_bwd)
            # For GQA, FLOPs calculation uses q_heads (since that's the actual computation)
            flops = compute_flops(batch, q_heads, seq_len, head_dim, causal, "fwd+bwd")
            cutile_tflops = flops * 1e-12 / (cutile_ms * 1e-3)
        except Exception as e:
            print(f"  Error: {e}")

        print(f"{config_str:<40} {format_result(cutile_ms, cutile_tflops):<20}")

    print()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    run_comprehensive_benchmarks()
    run_backward_only_benchmarks()
    run_gqa_benchmarks()
