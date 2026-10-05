# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the native reference environment: sphn (its Opus build needs the CMake policy floor on aarch64)
and the pinned official PersonaPlex source under ``<environment>/trtmc-reference/personaplex``."""

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
    if installed != SPHN_VERSION:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", f"sphn=={SPHN_VERSION}"],
                       check=True, env={**os.environ, "CMAKE_POLICY_VERSION_MINIMUM": os.environ.get(
                           "CMAKE_POLICY_VERSION_MINIMUM", "3.5")})


def main() -> None:
    _install_sphn()
    parent = Path(sys.prefix) / "trtmc-reference"
    destination = parent / "personaplex"
    if (destination / "moshi").is_dir():
        return
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="personaplex.", dir=parent))
    try:
        subprocess.run(["git", "init", "--quiet", str(temporary)], check=True)
        subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", REPOSITORY], check=True)
        subprocess.run(["git", "-C", str(temporary), "fetch", "--quiet", "--depth=1", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
