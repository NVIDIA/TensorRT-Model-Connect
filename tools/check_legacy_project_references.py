#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject retired repository and development-container terminology."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LegacyReference:
    path: str
    line: int
    kind: str


def legacy_terms() -> tuple[tuple[str, str], ...]:
    # Keep the retired spellings out of the checker itself so a whole-repo
    # scan can include this file without an allowlist.
    return (
        ("retired project slug", "trt" + "-transformer"),
        ("retired project display name", "trt" + " transformer"),
        ("retired full project slug", "tensorrt" + "-transformer"),
        ("retired full project display name", "tensorrt" + " transformer"),
        ("retired container prefix", "trtf" + "-dev"),
        ("retired container skill", "trtf" + "-agent-container"),
        ("retired container override", "trtf" + "_container_workdir"),
    )


def tracked_paths(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail or result.returncode}")
    return [
        repo_root / value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value
    ]


def scan_paths(repo_root: Path, paths: Iterable[Path]) -> list[LegacyReference]:
    findings: list[LegacyReference] = []
    terms = tuple((kind, term.casefold()) for kind, term in legacy_terms())

    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root).as_posix()
        folded_path = relative.casefold()
        for kind, term in terms:
            if term in folded_path:
                findings.append(LegacyReference(relative, 0, kind))

        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            folded_line = line.casefold()
            for kind, term in terms:
                if term in folded_line:
                    findings.append(LegacyReference(relative, line_number, kind))
    return findings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check tracked files for retired project/container terminology."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of tools/)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        findings = scan_paths(repo_root, tracked_paths(repo_root))
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if findings:
        for finding in findings:
            location = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"{location}: {finding.kind}")
        print(f"FAILED: found {len(findings)} retired project/container reference(s)")
        return 1

    print("No retired project/container references found in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
