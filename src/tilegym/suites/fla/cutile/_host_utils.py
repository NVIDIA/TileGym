# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared host guards and sequence indexing for FLA operators."""

import functools
import inspect

import torch
import torch.nn.functional as F


def input_guard(fn=None, *, no_guard_contiguous=False, set_device=True):
    """Make all input tensors contiguous and set the device based on input tensors."""

    def decorator(fn):
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            processed_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"__arg_{i}"
                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if set_device and tensor is not None and tensor.device.type == "cuda":
                ctx = torch.cuda.device(tensor.device.index)
            else:
                import contextlib

                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def _segmented_arange(counts: torch.LongTensor):
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype),
        counts,
    )
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra_idx


def device_cache_key():
    """Identify the CUDA device and architecture of the active launch context."""
    device = torch.cuda.current_device()
    return device, torch.cuda.get_device_capability(device)


def validate_mamba_chunk_training(g):
    """Reject unsupported gate gradients in the fused chunk implementation."""
    if torch.is_grad_enabled() and g is not None and g.requires_grad:
        raise NotImplementedError("Mamba2 fused-chunk does not support gradients for g.")
