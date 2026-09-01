# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Stable schema boundary for kernel-inventory Workloads.

Internal builds use the pinned canonical contract.  The public release strips
that private import block and uses TileGym's documented compatibility models.
Callers must import Workload symbols from this module rather than either
backend directly.
"""

from __future__ import annotations

from typing import Any

USING_CANONICAL_WORKLOAD_SCHEMA = False

if not USING_CANONICAL_WORKLOAD_SCHEMA:
    from tilegym.kernel_inventory._workload_schema_compat import CustomInput
    from tilegym.kernel_inventory._workload_schema_compat import EvalMode
    from tilegym.kernel_inventory._workload_schema_compat import InputSpec
    from tilegym.kernel_inventory._workload_schema_compat import NullInput
    from tilegym.kernel_inventory._workload_schema_compat import RandomInput
    from tilegym.kernel_inventory._workload_schema_compat import SafetensorsInput
    from tilegym.kernel_inventory._workload_schema_compat import SafetensorsShard
    from tilegym.kernel_inventory._workload_schema_compat import ScalarInput
    from tilegym.kernel_inventory._workload_schema_compat import StringInput
    from tilegym.kernel_inventory._workload_schema_compat import ToleranceSpec
    from tilegym.kernel_inventory._workload_schema_compat import Workload


def workload_model_validate(payload: Any) -> Workload:
    """Validate one Workload payload through the selected schema backend."""
    return Workload.model_validate(payload)


def workload_model_dump(workload: Workload) -> dict[str, Any]:
    """Return normalized JSON-compatible Workload data."""
    return workload.model_dump(mode="json")


__all__ = [
    "USING_CANONICAL_WORKLOAD_SCHEMA",
    "CustomInput",
    "EvalMode",
    "InputSpec",
    "NullInput",
    "RandomInput",
    "SafetensorsInput",
    "SafetensorsShard",
    "ScalarInput",
    "StringInput",
    "ToleranceSpec",
    "Workload",
    "workload_model_dump",
    "workload_model_validate",
]
