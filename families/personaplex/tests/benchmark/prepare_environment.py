# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned family-owned PersonaPlex reference checkout."""

from importlib.metadata import PackageNotFoundError, version as package_version
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "https://github.com/NVIDIA/personaplex.git"
REVISION = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
SPHN_VERSION = "0.1.4"


def _install_sphn() -> None:
    try:
        installed = package_version("sphn")
    except PackageNotFoundError:
        installed = None
    if installed == SPHN_VERSION:
        return
    environment = dict(os.environ)
    # sphn vendors an older Opus CMake project. CMake 4 requires its legacy
    # policy floor to be selected explicitly when building on aarch64.
    environment.setdefault("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            f"sphn=={SPHN_VERSION}",
        ],
        check=True,
        env=environment,
    )


def main() -> None:
    _install_sphn()
    parent = Path(sys.prefix) / "trtmc-reference"
    destination = parent / "personaplex"
    if (destination / "moshi").is_dir():
        return
    if destination.exists():
        raise RuntimeError(f"incomplete PersonaPlex reference: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="personaplex.", dir=parent))
    try:
        subprocess.run(["git", "init", "--quiet", str(temporary)], check=True)
        subprocess.run(
            ["git", "-C", str(temporary), "remote", "add", "origin", REPOSITORY], check=True
        )
        subprocess.run(
            ["git", "-C", str(temporary), "fetch", "--quiet", "--depth=1", "origin", REVISION],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            check=True,
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
