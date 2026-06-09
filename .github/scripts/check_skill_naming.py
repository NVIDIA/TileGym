#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Check that new skills added to skills/ follow the tilegym- naming convention.

Any skill directory newly added under skills/ must:
  1. Have a directory name that starts with "tilegym-"
  2. Have a `name:` field in its SKILL.md that starts with "tilegym-"

Only NEW skill directories (added in this PR, compared to the base branch) are
checked. Existing skills are not affected.

Usage:
    python check_skill_naming.py --base-ref <base_ref>

Exit codes:
    0  All new skills comply, or no new skills added
    1  One or more violations found
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

SKILLS_DIR = Path("skills")
REQUIRED_PREFIX = "tilegym-"

# ── helpers ──────────────────────────────────────────────────────────────────


def get_new_skill_dirs(base_ref: str) -> List[str]:
    """Return skill directory names that are *newly added* in this PR."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=A", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: git diff failed: {exc.stderr}", file=sys.stderr)
        sys.exit(1)

    new_files = result.stdout.splitlines()
    new_skill_dirs: set = set()

    for filepath in new_files:
        path = Path(filepath)
        # Match skills/<dir-name>/... — any file inside a new skills subdirectory
        if len(path.parts) >= 2 and path.parts[0] == "skills":
            new_skill_dirs.add(path.parts[1])

    return sorted(new_skill_dirs)


def read_skill_name_from_yaml(skill_dir: Path) -> Optional[str]:
    """Extract the `name:` field from SKILL.md YAML front-matter.

    Uses simple line-by-line parsing so no PyYAML dependency is required.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None

    # Find the closing ---
    end_idx = content.find("\n---", 3)
    if end_idx == -1:
        return None

    front_matter = content[3:end_idx]
    for line in front_matter.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            value = stripped[len("name:"):].strip().strip("\"'")
            return value if value else None

    return None


# ── main check ───────────────────────────────────────────────────────────────


def check_skill_naming(base_ref: str) -> bool:
    """Check newly added skill directories for naming convention compliance.

    Returns True if all checks pass, False if any violations are found.
    """
    new_dirs = get_new_skill_dirs(base_ref)

    if not new_dirs:
        print("No new skills detected under skills/. Nothing to check.")
        return True

    print(f"Checking {len(new_dirs)} new skill(s): {', '.join(new_dirs)}\n")
    violations: List[Tuple[str, List[str]]] = []

    for dir_name in new_dirs:
        skill_path = SKILLS_DIR / dir_name
        errors: List[str] = []

        # Check 1: directory name must start with "tilegym-"
        if not dir_name.startswith(REQUIRED_PREFIX):
            errors.append(
                f'  * Directory name "{dir_name}" does not start with "{REQUIRED_PREFIX}"'
            )

        # Check 2: SKILL.md must exist and its name: field must start with "tilegym-"
        yaml_name = read_skill_name_from_yaml(skill_path)
        if yaml_name is None:
            if not (skill_path / "SKILL.md").exists():
                errors.append(
                    f"  * skills/{dir_name}/SKILL.md is missing"
                )
            else:
                errors.append(
                    f"  * skills/{dir_name}/SKILL.md has no `name:` field in its YAML front-matter"
                )
        elif not yaml_name.startswith(REQUIRED_PREFIX):
            errors.append(
                f'  * `name: "{yaml_name}"` in SKILL.md does not start with "{REQUIRED_PREFIX}"'
            )

        if errors:
            violations.append((dir_name, errors))
        else:
            print(f"  ok  skills/{dir_name}/  (name: {read_skill_name_from_yaml(skill_path)})")

    if not violations:
        print(f"\nAll {len(new_dirs)} new skill(s) follow the \"{REQUIRED_PREFIX}\" naming convention.")
        return True

    # ── print violations ──────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"SKILL NAMING VIOLATION -- {len(violations)} skill(s) must be renamed")
    print(f"{sep}\n")

    for dir_name, errors in violations:
        print(f"Skill: skills/{dir_name}/")
        for err in errors:
            print(err)
        print()

    print(
        "WHY this rule exists\n"
        "--------------------\n"
        "  TileGym skills are published to the nvidia/skills registry, where skills\n"
        "  from many NVIDIA projects coexist. The \"tilegym-\" prefix prevents name\n"
        "  collisions and makes the origin of each skill immediately clear to users.\n"
    )
    print(
        "HOW to fix\n"
        "----------\n"
        "  1. Rename the skill directory:\n"
        "         git mv skills/<name>  skills/tilegym-<name>\n"
        "\n"
        "  2. Update the `name:` field in skills/tilegym-<name>/SKILL.md:\n"
        "         name: \"tilegym-<name>\"\n"
        "\n"
        "  3. If other SKILL.md files reference this skill by name,\n"
        "     update those references too.\n"
    )

    return False


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that new skills in skills/ follow the tilegym- naming convention."
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to diff against (default: origin/main)",
    )
    args = parser.parse_args()

    ok = check_skill_naming(args.base_ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
