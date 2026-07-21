#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the selected model-owned EdgeLLM CPU contracts."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[5]
TEST_ROOT = Path(__file__).resolve().parent
PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PROVIDER_TESTS = (
    "tests/builder/test_manifest_validation.py",
    "tests/builder/test_optimized_runtime_capsules.py",
    "tests/builder/test_optimized_runtime_orchestrator.py",
    "tests/builder/test_optimized_runtime_public_routing.py",
    "tests/tools/test_optimized_runtime_packaging.py",
)


def selected_tests(scope: str, profile: str) -> tuple[str, ...]:
    root = TEST_ROOT.relative_to(REPOSITORY).as_posix()
    if scope == "leaf":
        if not PROFILE_RE.fullmatch(profile) or not (TEST_ROOT / profile).is_dir():
            raise ValueError(f"unknown EdgeLLM profile: {profile!r}")
        return (
            f"{root}/{profile}",
            f"{root}/coexistence/test_coexistence_contract.py",
        )
    if profile:
        raise ValueError(f"{scope} scope does not accept a profile")
    if scope == "family":
        return (root,)
    if scope == "provider":
        return (root, *PROVIDER_TESTS)
    raise ValueError(f"unsupported EdgeLLM CPU contract scope: {scope!r}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("leaf", "family", "provider"), required=True)
    parser.add_argument("--profile", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        tests = selected_tests(arguments.scope, arguments.profile)
    except ValueError as exc:
        print(f"qwen-edgellm-contracts: error: {exc}", file=sys.stderr)
        return 2
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu",
            *tests,
        ],
        cwd=REPOSITORY,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
