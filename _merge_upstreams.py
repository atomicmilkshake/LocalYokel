"""Fetch Cruz + official and merge into a working branch. Windows-safe, no PowerShell."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print("git", " ".join(args), flush=True)
    r = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        errors="replace",
        capture_output=True,
    )
    if r.stdout:
        print(r.stdout, end="" if r.stdout.endswith("\n") else "\n")
    if r.stderr:
        print(r.stderr, end="" if r.stderr.endswith("\n") else "\n")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.returncode}")
    return r


def main() -> None:
    git("status", "-sb")
    stash = git("stash", "push", "-u", "-m", "windows-ipc-and-local-before-upstream-merge", check=False)
    print("stash rc", stash.returncode)
    git("fetch", "origin")
    git("fetch", "upstream-flash")
    print("--- origin/main ---")
    git("log", "-3", "--oneline", "origin/main")
    print("--- upstream-flash/main ---")
    git("log", "-3", "--oneline", "upstream-flash/main")
    print("--- merge-base ---")
    git("merge-base", "origin/main", "upstream-flash/main")
    print("--- commits Cruz not in official ---")
    git("rev-list", "--count", "upstream-flash/main..origin/main")
    print("--- commits official not in Cruz ---")
    git("rev-list", "--count", "origin/main..upstream-flash/main")

    # Recreate branch from current main, then FF Cruz, then merge official
    git("checkout", "main")
    git("branch", "-D", "merge/official-cruz-tq4", check=False)
    git("checkout", "-B", "merge/official-cruz-tq4")
    git("merge", "--ff-only", "origin/main")
    print("=== merging official into Cruz ===")
    m = git("merge", "--no-edit", "upstream-flash/main", check=False)
    if m.returncode != 0:
        print("MERGE CONFLICTS:")
        git("diff", "--name-only", "--diff-filter=U", check=False)
        print("status after failed merge:")
        git("status", "-sb", check=False)
        sys.exit(2)
    print("MERGE OK")
    git("log", "-8", "--oneline")
    git("status", "-sb")


if __name__ == "__main__":
    main()
