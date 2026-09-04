#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select exact family tests from the physical ownership boundary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


DOC_PREFIXES = ("website/",)
DOC_FILES = {
    "AGENTS.md",
    "CODEOWNERS",
    "CONTRIBUTING.md",
    "README.md",
}
SHARED_PREFIXES = (
    ".github/",
    "apps/",
    "benchmarks/",
    "cmake/",
    "configs/",
    "core/",
    "examples/",
    "include/",
    "plugins/",
    "python/tensorrt_model_connect/",
    "requirements/",
    "scripts/",
    "src/",
    "tests/",
    "third_party/",
    "tools/",
)
SHARED_FILES = {
    ".clang-format",
    ".coderabbit.yaml",
    ".gitignore",
    ".pre-commit-config.yaml",
    "ASSET_LICENSES.md",
    "CMakeLists.txt",
    "Dockerfile",
    "Dockerfile.community-cpu",
    "Dockerfile.dev.aarch64",
    "Dockerfile.dev.x86",
    ".dockerignore",
    "_pyproject_backend.py",
    "conanfile.py",
    "conftest.py",
    "pyproject.toml",
    "ruff.toml",
}


@dataclass(frozen=True)
class Impact:
    scope: str
    families: tuple[str, ...]
    changed_files: tuple[str, ...]
    run_core_tests: bool
    run_docs: bool


def inventory(repo: Path) -> tuple[str, ...]:
    root = repo / "families"
    if not root.is_dir():
        raise ValueError("repository has no families directory")
    return tuple(
        sorted(
            path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")
        )
    )


def _safe_relative(path: str) -> str:
    normalized = PurePosixPath(path).as_posix().removeprefix("./")
    candidate = PurePosixPath(normalized)
    if not normalized or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe changed path: {path!r}")
    return normalized


def classify(repo: Path, files: Sequence[str]) -> Impact:
    known = set(inventory(repo))
    changed = tuple(sorted({_safe_relative(path) for path in files}))
    selected: set[str] = set()
    shared = False
    docs = False
    unknown: list[str] = []

    for path in changed:
        if path == "families/__init__.py":
            shared = True
            continue
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[0] == "families":
            family = parts[1]
            if family not in known:
                raise ValueError(f"changed path names unknown family {family!r}: {path}")
            selected.add(family)
            continue
        if path in DOC_FILES or path.startswith(DOC_PREFIXES):
            docs = True
            continue
        if len(parts) == 1 and path.endswith(".py"):
            shared = True
            continue
        if path in SHARED_FILES or path.startswith(SHARED_PREFIXES):
            shared = True
            continue
        unknown.append(path)

    if unknown:
        raise ValueError("unclassified changed paths: " + ", ".join(unknown))
    if shared:
        return Impact("all", tuple(sorted(known)), changed, True, docs)
    if selected:
        return Impact("families", tuple(sorted(selected)), changed, True, docs)
    return Impact("docs" if docs else "none", (), changed, False, docs)


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", base, head],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTD", merge_base, head],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def validate(repo: Path) -> None:
    violations: list[str] = []
    if any(path.is_file() for path in (repo / "tests/cpp/models").glob("**/*")):
        violations.append("central tests/cpp/models is forbidden")
    family_names = set(inventory(repo))
    for family in inventory(repo):
        root = repo / "families" / family
        for required in (
            "model.py",
            "support.py",
            "runtime/CMakeLists.txt",
            "tests/test_e2e.py",
            "tests/manifests",
            "tests/thresholds",
        ):
            if not (root / required).exists():
                violations.append(f"{family}: missing {required}")
        if list(root.rglob("MODEL.toml")):
            violations.append(f"{family}: MODEL.toml is forbidden")
        for obsolete in ("plugin.py", "tests/runner.py", "tests/e2e_plugins"):
            if (root / obsolete).exists():
                violations.append(f"{family}: obsolete {obsolete}")
        for path in (root / "tests/cpp").glob("**/*"):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu"}:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r'#include\s+[<"](?:src|tests)/', source):
                violations.append(f"{family}: {path.relative_to(root)} imports central internals")
            includes = re.findall(r'#include\s+[<"]families/([^/]+)/([^>"]+)', source)
            for owner, relative in includes:
                if owner in family_names and owner != family:
                    violations.append(f"{family}: {path.relative_to(root)} imports sibling {owner}")
                elif owner == family and not (repo / "families" / owner / relative).is_file():
                    violations.append(
                        f"{family}: {path.relative_to(root)} includes missing {relative}"
                    )
    if violations:
        raise ValueError("; ".join(violations))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--base", default="github/main")
    result.add_argument("--head", default="HEAD")
    result.add_argument("--files", help="comma-separated repo-relative paths")
    result.add_argument("--validate", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        if args.validate:
            validate(repo)
            payload = {"valid": True, "families": list(inventory(repo))}
        else:
            files = (
                args.files.split(",") if args.files else changed_files(repo, args.base, args.head)
            )
            payload = asdict(classify(repo, files))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"test-impact: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
