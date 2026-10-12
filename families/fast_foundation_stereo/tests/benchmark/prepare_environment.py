# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned official reference implementation inside this family environment."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download


SOURCE_REPOSITORY = "https://github.com/NVlabs/Fast-FoundationStereo.git"
SOURCE_REVISION = "a290ba04c1b3ad1ec41a33974a157b2917b624d4"
CHECKPOINT = "nvidia/c-fast-foundationstereo"
CHECKPOINT_REVISION = "9b446878c81ddb27593036767b29b2859d46103e"


def main() -> int:
    parent = Path(sys.prefix) / "trtmc-reference"
    destination = parent / "Fast-FoundationStereo"
    required = (
        destination / "core/foundation_stereo.py",
        destination / "core/submodule.py",
        destination / "weights/23-36-37/model_best_bp2_serialize.pth",
    )
    if all(path.is_file() for path in required):
        return 0
    if destination.exists():
        raise RuntimeError(f"incomplete Fast Foundation Stereo reference: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="Fast-FoundationStereo.", dir=parent))
    try:
        subprocess.run(
            ["git", "init", "--quiet", str(temporary)],
            check=True,
        )
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
        checkpoint = Path(
            hf_hub_download(
                repo_id=CHECKPOINT,
                revision=CHECKPOINT_REVISION,
                filename="model_best_bp2_serialize.pth",
            )
        )
        target = temporary / "weights/23-36-37/model_best_bp2_serialize.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint, target)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
