# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Any
from typing import Optional

import torch

# --- cuTile kernel caching utilities (local to autotune module) ---
_CUTILE_KERNEL_CACHE: dict = {}


def _get_device_capability(device: Optional[torch.device]) -> tuple:
    if device is None:
        device = torch.device("cuda")
    return torch.cuda.get_device_capability(device)


def get_cached_ct_kernel(
    pyfunc: Any,
    *,
    num_ctas: Optional[int] = None,
    occupancy: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    # Lazy import cuda.tile
    import cuda.tile as ct

    capability = _get_device_capability(device)
    key = (id(pyfunc), num_ctas, occupancy, capability)
    kernel = _CUTILE_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    kernel_kwargs = {}
    if num_ctas is not None:
        kernel_kwargs["num_ctas"] = num_ctas
    if occupancy is not None:
        kernel_kwargs["occupancy"] = occupancy

    kernel = ct.kernel(**kernel_kwargs)(pyfunc)
    _CUTILE_KERNEL_CACHE[key] = kernel
    return kernel


DISABLE_AUTOTUNE_ENV = "TILEGYM_DISABLE_AUTOTUNE"
_DISABLE_AUTOTUNE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_DISABLE_AUTOTUNE_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def is_autotune_disabled() -> bool:
    """Return whether autotune is disabled for the current process.

    Autotune stays enabled by default. ``TILEGYM_DISABLE_AUTOTUNE`` is the
    single public switch; ``1/true/yes/on`` disables autotune and
    ``0/false/no/off`` keeps the default enabled behavior. Operator code must
    not read environment variables directly or alias removed ad hoc autotune
    flags.
    """
    # Local import keeps this function self-contained.
    import os

    disable_flag = os.environ.get(DISABLE_AUTOTUNE_ENV)
    if disable_flag is None:
        return False

    disable_flag = disable_flag.strip().lower()
    if disable_flag in _DISABLE_AUTOTUNE_TRUE_VALUES:
        return True
    if disable_flag in _DISABLE_AUTOTUNE_FALSE_VALUES:
        return False

    valid_values = ", ".join(sorted(_DISABLE_AUTOTUNE_TRUE_VALUES | _DISABLE_AUTOTUNE_FALSE_VALUES))
    raise ValueError(f"{DISABLE_AUTOTUNE_ENV} must be one of {{{valid_values}}}; got {disable_flag!r}")


def is_autotune_enabled() -> bool:
    """Return the process-wide autotune policy."""
    return not is_autotune_disabled()
