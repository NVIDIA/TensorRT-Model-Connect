# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned family-owned PersonaPlex reference checkout."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "https://github.com/NVIDIA/personaplex.git"
REVISION = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"


def main() -> None:
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
