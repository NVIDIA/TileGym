# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
from typing import Any

_DUMP_TYPES = os.environ.get("DUMP_TYPES", "").lower() in ("1", "true", "yes")
_DUMP_TYPES_FILE = os.environ.get("DUMP_TYPES_FILE", "")


def dump_kernel_types(kernel_name: str, *args: Any) -> None:
    """Dump kernel name and argument types if DUMP_TYPES is set."""
    if not _DUMP_TYPES:
        return

    import torch

    types = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            types.append(str(arg.dtype))
        else:
            types.append(type(arg).__name__)

    msg = f"{kernel_name}: {', '.join(types)}\n"

    if _DUMP_TYPES_FILE:
        with open(_DUMP_TYPES_FILE, "a") as f:
            f.write(msg)
    else:
        print(msg, end="")
