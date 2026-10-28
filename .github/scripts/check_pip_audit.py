#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Gate a pip-audit JSON report.

Every reported vulnerability fails the check, except those in packages passed as
waived: they are still listed in the log and summarized as a warning, but do not fail.

Usage: check_pip_audit.py <pip-audit JSON report> [waived-package ...]
"""

import json
import sys


def main():
    report_path, *waived_args = sys.argv[1:]
    waived_packages = {name.lower() for name in waived_args}

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"::error title=pip-audit::no usable report at {report_path} ({e}); see pip-audit output above")
        return 1

    blocking, waived = [], []
    for dep in report["dependencies"]:
        for vuln in dep.get("vulns", []):
            fixes = ", ".join(vuln["fix_versions"]) or "none"
            line = f"{dep['name']} {dep['version']} {vuln['id']} (fix: {fixes})"
            (waived if dep["name"].lower() in waived_packages else blocking).append(line)

    for line in blocking:
        print(f"BLOCKING  {line}")
    for line in waived:
        print(f"WAIVED    {line}")
    if waived:
        print(
            f"::warning title=pip-audit::{len(waived)} vulnerabilities waived in: {', '.join(sorted(waived_packages))}"
        )
    if blocking:
        print(f"::error title=pip-audit::{len(blocking)} blocking vulnerabilities")
        return 1
    print(f"No blocking vulnerabilities ({len(waived)} waived)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
