#!/usr/bin/env python3
"""Fail when deleted legacy runtime paths are modified without an explicit override.

The service-composed runtime is now the only supported runtime path. This guard exists
to prevent accidental reintroduction of the deleted compatibility assembly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_PREFIXES = (
    "src/cabi/pipeline/",
    "src/cabi/registry/",
    "src/cabi/factories/",
)


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def is_protected(path: str) -> bool:
    normalized = path.strip()
    return any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def changed_files_from_base(base_ref: str) -> list[str]:
    merge_base = run_git(["merge-base", "HEAD", base_ref])
    diff_output = run_git(["diff", "--name-only", "--diff-filter=ACMRTUXB", f"{merge_base}..HEAD"])
    return [line.strip() for line in diff_output.splitlines() if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block accidental reintroduction of deleted legacy runtime files."
    )
    parser.add_argument(
        "--base",
        default="github/main",
        help="Base ref used to compute changed files (default: github/main). Ignored when --files is set.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Explicit changed files to evaluate instead of querying git diff.",
    )
    parser.add_argument(
        "--allow-override",
        action="store_true",
        help="Allow protected legacy-path changes for explicit archival or cleanup work.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.allow_override:
        print(
            "[legacy-freeze] override enabled via --allow-override; skipping deleted-path failure.",
            file=sys.stderr,
        )
        return 0

    try:
        changed_files = args.files if args.files is not None and len(args.files) > 0 else changed_files_from_base(args.base)
    except RuntimeError as exc:
        print(f"[legacy-freeze] error: {exc}", file=sys.stderr)
        return 2

    protected = [path for path in changed_files if is_protected(path)]
    if not protected:
        print("[legacy-freeze] OK: no deleted legacy runtime paths were modified.")
        return 0

    print("[legacy-freeze] blocked deleted legacy runtime path changes:", file=sys.stderr)
    for path in protected:
        print(f"  - {path}", file=sys.stderr)
    print(
        "[legacy-freeze] the compatibility factory/runtime path has been deleted. "
        "Do not reintroduce files under src/cabi/pipeline, src/cabi/factories, or "
        "src/cabi/registry. If this branch is explicitly performing archival or cleanup "
        "work, rerun with --allow-override and record that justification in the task doc.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
