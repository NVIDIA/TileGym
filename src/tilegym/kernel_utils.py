# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Kernel utility functions for TileGym."""

from typing import Any
from typing import Dict
from typing import Optional

from tilegym.logger import get_logger


def get_kernel_configs(default_configs: Dict[str, Any], provided_configs: Optional[Dict[str, Any]] = None):
    """
    Merge default kernel configs with provided configs.

    Args:
        default_configs: Default kernel configuration dictionary.
        provided_configs: Optional user-provided configuration dictionary.

    Returns:
        Merged configuration dictionary with provided configs overriding defaults.
    """
    logger = get_logger(__name__)

    if provided_configs is None:
        return default_configs
    # log any differences between default_configs and provided_configs
    for key, value in default_configs.items():
        if key not in provided_configs:
            logger.warning(f"Provided kernel config {key} is not in default: {value}")
            continue
        if provided_configs[key] != value:
            logger.info(f"Provided kernel config {key} differs from default: {value} -> {provided_configs[key]}")
    return {**default_configs, **provided_configs}
