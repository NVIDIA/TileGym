# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from importlib import resources

import yaml


def _load_list_yaml(resource_name):
    text = resources.files(__package__).joinpath(resource_name).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


class KernelFilter:
    def __init__(self, config_resource="tilegym_kernel_prefixes.yaml"):
        config = _load_list_yaml(config_resource)
        self.kernel_names_prefix = config.get("prefixes", [])
        self.blacklist_kernel_names = config.get("blacklist", [])

    def get_kernel_names(self):
        return self.kernel_names_prefix

    def contains(self, key):
        for prefix in self.kernel_names_prefix:
            if prefix in key and key not in self.blacklist_kernel_names:
                return True
        return False
