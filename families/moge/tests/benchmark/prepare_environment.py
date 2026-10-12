# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned official MoGe source inside this family's environment."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SOURCE_REPOSITORY = "https://github.com/microsoft/MoGe.git"
SOURCE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"


def main() -> int:
    parent = Path(sys.prefix) / "trtmc-reference"
    destination = parent / "MoGe"
    required = destination / "moge/model/v2.py"
    if required.is_file():
        return 0
    if destination.exists():
        raise RuntimeError(f"incomplete MoGe reference: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="MoGe.", dir=parent))
    try:
        subprocess.run(["git", "init", "--quiet", str(temporary)], check=True)
        subprocess.run(
            ["git", "-C", str(temporary), "remote", "add", "origin", SOURCE_REPOSITORY],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary),
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                SOURCE_REVISION,
            ],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
