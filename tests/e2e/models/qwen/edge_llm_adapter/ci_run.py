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
NLOHMANN_PACKAGE = "nlohmann-json3-dev=3.11.3-1"
NLOHMANN_INCLUDE = Path("/usr/include")


def selected_tests(scope: str, profile: str) -> tuple[str, ...]:
    root = TEST_ROOT.relative_to(REPOSITORY).as_posix()
    if scope == "leaf":
        leaf = TEST_ROOT / profile
        if (
            not PROFILE_RE.fullmatch(profile)
            or leaf.is_symlink()
            or not leaf.is_dir()
            or not leaf.resolve(strict=True).is_relative_to(TEST_ROOT.resolve(strict=True))
        ):
            raise ValueError(f"unknown EdgeLLM profile: {profile!r}")
        return (
            f"{root}/{profile}",
            f"{root}/coexistence/test_coexistence_contract.py",
        )
    if profile:
        raise ValueError(f"{scope} scope does not accept a profile")
    if any(path.is_symlink() for path in TEST_ROOT.iterdir()):
        raise ValueError("Qwen EdgeLLM test ownership contains a symlinked directory")
    if scope in {"family", "provider"}:
        return (root,)
    raise ValueError(f"unsupported EdgeLLM CPU contract scope: {scope!r}")


def _provision_runtime() -> tuple[str, dict[str, str]]:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if environment.get("TRTMC_OPTIMIZED_CI_PROVISION") != "1":
        return sys.executable, environment
    virtual_environment = (
        Path(environment.get("RUNNER_TEMP", "/tmp")) / "trtmc-qwen-edgellm-ci"
    )
    python = virtual_environment / "bin/python"
    if not python.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", str(virtual_environment)],
            check=True,
        )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "pytest==9.1.1",
            "numpy==2.4.6",
        ],
        cwd=REPOSITORY,
        check=True,
    )
    header = NLOHMANN_INCLUDE / "nlohmann/json.hpp"
    if not header.is_file():
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(
            [
                "sudo",
                "apt-get",
                "install",
                "--yes",
                "--no-install-recommends",
                NLOHMANN_PACKAGE,
            ],
            check=True,
        )
    if not header.is_file():
        raise RuntimeError(f"Qwen CI dependency is unavailable after provisioning: {header}")
    return str(python), environment


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
    python, environment = _provision_runtime()
    return subprocess.run(
        [
            python,
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
        env=environment,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
