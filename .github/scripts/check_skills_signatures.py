# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
Check that /nvskills-ci has been run (and is current) whenever skills/ content is modified.

Compares commits on the current branch against BASE_REF (default: origin/main).

Exit 0 if:
  - No non-.sig skills files changed in this branch
  - Non-.sig skills files changed AND the last nvskills-ci bot commit is newer
    than the last content change

Exit 1 if:
  - Non-.sig skills files changed but no nvskills-ci bot commit exists on this branch
  - Non-.sig skills files changed and the last nvskills-ci bot commit is older
    than the latest content change
"""

import os
import subprocess
import sys


def git(args):
    """Run a git command, return (stdout_stripped, returncode)."""
    result = subprocess.run(["git"] + args, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode


def get_timestamp(commit_hash):
    out, _ = git(["log", "-1", "--format=%ct", commit_hash])
    return int(out) if out else 0


def find_last_content_commit(base_ref):
    """Return the most recent branch commit that changed non-.sig skills files."""
    hashes_out, _ = git(["log", f"{base_ref}..HEAD", "--format=%H", "--", "skills/"])
    for h in hashes_out.splitlines() if hashes_out else []:
        h = h.strip()
        if not h:
            continue
        files_out, _ = git(["diff-tree", "--no-commit-id", "-r", "--name-only", h])
        if any(f.startswith("skills/") and not f.endswith(".sig") for f in files_out.splitlines()):
            return h
    return None


def main():
    base_ref = os.environ.get("BASE_REF", "origin/main")
    sig_title = os.environ.get(
        "NVSKILLS_SIGNATURE_COMMIT_TITLE",
        "Attach NVSkills validation signatures",
    )

    # Step 1: Any non-.sig skills changes in this PR?
    diff_out, rc = git(["diff", "--name-only", f"{base_ref}...HEAD", "--", "skills/"])
    if rc != 0:
        print("Warning: git diff failed. Skipping check.")
        return

    changed = [f for f in diff_out.splitlines() if f]
    non_sig = [f for f in changed if not f.endswith(".sig")]

    if not non_sig:
        print("✓ No skills content changes detected. Check passes.")
        return

    print(f"Detected {len(non_sig)} non-sig skills change(s):")
    for f in non_sig:
        print(f"  • {f}")
    print()

    # Step 2: Find the last nvskills-ci bot commit on this branch
    bot_out, _ = git(["log", f"{base_ref}..HEAD", "--format=%H", f"--grep=^{sig_title}"])
    bot_hashes = [h for h in bot_out.splitlines() if h]
    last_bot = bot_hashes[0] if bot_hashes else None

    # Step 3: Find the last branch commit that changed skills content (non-.sig)
    last_content = find_last_content_commit(base_ref)
    if last_content is None:
        print("✓ No non-sig skills commits on branch. Check passes.")
        return

    # Step 4: Verdict
    if last_bot is None:
        print("❌ Skills content was modified but /nvskills-ci has not been run on this PR.")
        print()
        print("→ Comment '/nvskills-ci' on this PR and wait for the bot to push signature files.")
        sys.exit(1)

    if get_timestamp(last_bot) < get_timestamp(last_content):
        print("❌ Skills content was modified AFTER the last nvskills-ci run — signatures are stale.")
        print(f"   Last content change : {last_content[:8]}")
        print(f"   Last nvskills-ci run: {last_bot[:8]}")
        print()
        print("→ Comment '/nvskills-ci' on this PR again to regenerate signatures.")
        sys.exit(1)

    print("✓ nvskills-ci was run after the latest skills changes. Check passes.")
    print(f"   Last content change : {last_content[:8]}")
    print(f"   Last nvskills-ci run: {last_bot[:8]}")


if __name__ == "__main__":
    main()
