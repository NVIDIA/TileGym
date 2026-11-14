#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Gate a pip-audit JSON report.

Every reported vulnerability fails the check. A missing or unreadable report, which is what a
dependency resolution error leaves behind, fails it too.

Usage: check_pip_audit.py <pip-audit JSON report>
"""

import json
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: check_pip_audit.py <pip-audit JSON report>", file=sys.stderr)
        return 2
    report_path = sys.argv[1]

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"::error title=pip-audit::no usable report at {report_path} ({e}); see pip-audit output above")
        return 1

    blocking = []
    for dep in report["dependencies"]:
        for vuln in dep.get("vulns", []):
            fixes = ", ".join(vuln["fix_versions"]) or "none"
            blocking.append(f"{dep['name']} {dep['version']} {vuln['id']} (fix: {fixes})")

    for line in blocking:
        print(f"BLOCKING  {line}")
    if blocking:
        print(f"::error title=pip-audit::{len(blocking)} blocking vulnerabilities")
        return 1
    print("No blocking vulnerabilities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
