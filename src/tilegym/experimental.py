# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct

from tilegym.logger import warn_once

_original_launch = ct.launch


def _default_message(kernel_obj):
    name = (
        getattr(kernel_obj, "__name__", None)
        or getattr(getattr(kernel_obj, "_pyfunc", None), "__name__", None)
        or getattr(kernel_obj, "name", "unknown_kernel")
    )
    return (
        f"{name} is an experimental kernel contributed by "
        "external GitHub TileGym contributors. "
        "This kernel has not been fully validated by the core team."
    )


def experimental_kernel(message_or_kernel=None):
    """
    Decorator to mark a ct.kernel as experimental for one-time message printing.
    Place before @ct.kernel.

    Usage:
        @experimental_kernel
        @ct.kernel(occupancy=2)
        def my_kernel(...):
            ...

        @experimental_kernel()
        @ct.kernel(occupancy=2)
        def my_kernel(...):
            ...

        @experimental_kernel("Custom warning message.")
        @ct.kernel(occupancy=2)
        def my_kernel(...):
            ...
    """

    def decorator(kernel_obj):
        msg = message if message is not None else _default_message(kernel_obj)
        kernel_obj._tracked_message = msg
        return kernel_obj

    # Bare decorator: @experimental_kernel (no parens) — receives the kernel object directly
    if callable(message_or_kernel) and not isinstance(message_or_kernel, str):
        kernel_obj = message_or_kernel
        kernel_obj._tracked_message = _default_message(kernel_obj)
        return kernel_obj

    # Custom message: @experimental_kernel("msg")
    if isinstance(message_or_kernel, str):
        message = message_or_kernel
    else:
        # Empty parens: @experimental_kernel()
        message = None

    return decorator


def _patched_launch(stream, grid, kernel, kernel_args, /):
    msg = getattr(kernel, "_tracked_message", None)
    if msg:
        warn_once(msg, "EXPERIMENTAL")
        kernel._tracked_message = None  # Mark as printed
    return _original_launch(stream, grid, kernel, kernel_args)


def _apply_patch():
    """Apply the monkey-patch to ct.launch. Called once at tilegym import time."""
    ct.launch = _patched_launch


def reset_tracking():
    """Reset all experimental kernel messages to allow re-printing (useful for testing)."""
    # Re-enable messages for all kernels that had them cleared
    pass
