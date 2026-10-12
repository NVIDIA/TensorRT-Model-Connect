# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the native reference environment: the pinned official Fast-FoundationStereo source and its
serialized checkpoint under ``<environment>/trtmc-reference/Fast-FoundationStereo``."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE_REPOSITORY = "https://github.com/NVlabs/Fast-FoundationStereo.git"
SOURCE_REVISION = "a290ba04c1b3ad1ec41a33974a157b2917b624d4"
CHECKPOINT = "nvidia/c-fast-foundationstereo"
CHECKPOINT_REVISION = "9b446878c81ddb27593036767b29b2859d46103e"
WEIGHTS = "weights/23-36-37/model_best_bp2_serialize.pth"


def main() -> int:
    from huggingface_hub import hf_hub_download

    parent = Path(sys.prefix) / "trtmc-reference"
    destination = parent / "Fast-FoundationStereo"
    if all((destination / name).is_file() for name in ("core/foundation_stereo.py", "core/submodule.py", WEIGHTS)):
        return 0
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="Fast-FoundationStereo.", dir=parent))
    try:
        subprocess.run(["git", "init", "--quiet", str(temporary)], check=True)
        subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", SOURCE_REPOSITORY], check=True)
        subprocess.run(["git", "-C", str(temporary), "fetch", "--quiet", "--depth=1", "origin", SOURCE_REVISION], check=True)
        subprocess.run(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
        checkpoint = Path(hf_hub_download(repo_id=CHECKPOINT, revision=CHECKPOINT_REVISION,
                                          filename="model_best_bp2_serialize.pth"))
        (temporary / WEIGHTS).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint, temporary / WEIGHTS)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
