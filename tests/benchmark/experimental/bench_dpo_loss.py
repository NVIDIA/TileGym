# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark fused linear + sigmoid DPO loss (forward + backward) with Triton perf_report style."""

import torch
import torch.nn.functional as F
import triton

from tilegym.backend import is_backend_available

if is_backend_available("cutile"):
    import tilegym
    from tilegym.suites.liger.ops import dpo_loss
else:
    dpo_loss = None

DEVICE = triton.runtime.driver.active.get_active_torch_device()

ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def _supported_backends():
    return [b for b in ALL_BACKENDS if b is not None]


def _seq_logps(x, w, target, ignore_index=-100):
    logits = x @ w.t()
    logp = torch.log_softmax(logits.float(), dim=-1)
    mask = target != ignore_index
    tok = logp.gather(-1, torch.where(mask, target, 0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1)


def _torch_dpo_loss(x, w, target, ref_x, ref_w, beta=0.1):
    """Unfused reference: materializes the full (B, T, V) logits."""
    n_pairs = x.shape[0] // 2
    logps = _seq_logps(x, w, target)
    with torch.no_grad():
        ref_logps = _seq_logps(ref_x, ref_w, target)
    diff = beta * ((logps[:n_pairs] - ref_logps[:n_pairs]) - (logps[n_pairs:] - ref_logps[n_pairs:]))
    return -F.logsigmoid(diff).sum() / n_pairs


def _create_config(ylabel, kind, hidden_size, vocab_size, seq_len, datatype):
    available = _supported_backends()
    if not available:
        return None
    backends, names, styles = zip(*available)
    dtype_name = str(datatype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["n_pairs"],
        x_vals=[1, 2, 4, 8, 16],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel=ylabel,
        plot_name=f"dpo-loss-{kind}-T{seq_len}-H{hidden_size}-V{vocab_size}-{dtype_name}",
        args={"hidden_size": hidden_size, "vocab_size": vocab_size, "seq_len": seq_len, "datatype": datatype},
    )


def _make(n_pairs, seq_len, hidden_size, vocab_size, datatype, device):
    B = 2 * n_pairs
    x = (torch.randn(B, seq_len, hidden_size, device=device, dtype=datatype) * 0.5).requires_grad_(True)
    w = (torch.randn(vocab_size, hidden_size, device=device, dtype=datatype) * 0.02).requires_grad_(True)
    t = torch.randint(0, vocab_size, (B, seq_len), device=device)
    t[:, : seq_len // 4] = -100  # prompt positions
    ref_x = torch.randn(B, seq_len, hidden_size, device=device, dtype=datatype) * 0.5
    ref_w = torch.randn(vocab_size, hidden_size, device=device, dtype=datatype) * 0.02
    return x, w, t, ref_x, ref_w


def _fwd_bwd(backend, x, w, t, ref_x, ref_w):
    if backend == "cutile":
        tilegym.set_backend("cutile")
        loss = dpo_loss(x, w, t, ref_input=ref_x, ref_weight=ref_w, beta=0.1)[0]
    else:
        loss = _torch_dpo_loss(x, w, t, ref_x, ref_w, beta=0.1)
    loss.backward()


_CONFIGS = [(4096, 128256, 1024, torch.bfloat16)]


@triton.testing.perf_report([_create_config("ms", "fwd-bwd-latency", H, V, T, dt) for H, V, T, dt in _CONFIGS])
def bench_dpo_loss(n_pairs, backend, hidden_size, vocab_size, seq_len, datatype, device=DEVICE):
    x, w, t, ref_x, ref_w = _make(n_pairs, seq_len, hidden_size, vocab_size, datatype, device)
    return triton.testing.do_bench(lambda: _fwd_bwd(backend, x, w, t, ref_x, ref_w), grad_to_none=[x, w])


@triton.testing.perf_report([_create_config("GB", "peak-memory", H, V, T, dt) for H, V, T, dt in _CONFIGS])
def bench_dpo_loss_memory(n_pairs, backend, hidden_size, vocab_size, seq_len, datatype, device=DEVICE):
    x, w, t, ref_x, ref_w = _make(n_pairs, seq_len, hidden_size, vocab_size, datatype, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        _fwd_bwd(backend, x, w, t, ref_x, ref_w)
    except torch.OutOfMemoryError:
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() * 1e-9


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
    else:
        bench_dpo_loss.run(print_data=True)
        bench_dpo_loss_memory.run(print_data=True)
