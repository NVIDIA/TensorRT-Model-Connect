# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned SANA source and a local HF-compatible model directory."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import snapshot_download

REPOSITORY = "https://github.com/NVlabs/Sana.git"
REVISION = "59629fdf790850797cb657bad014fce432bd713d"
MODEL = "Efficient-Large-Model/SANA-WM_bidirectional"


def main() -> None:
    parent = Path(sys.prefix) / "trtmc-reference"
    source = parent / "Sana"
    parent.mkdir(parents=True, exist_ok=True)
    if not (source / "inference_video_scripts/wm/inference_sana_wm.py").is_file():
        if source.exists():
            raise RuntimeError(f"incomplete SANA reference: {source}")
        temporary = Path(tempfile.mkdtemp(prefix="Sana.", dir=parent))
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
            temporary.rename(source)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    model = parent / "SANA-model"
    if not model.exists():
        model.symlink_to(Path(snapshot_download(MODEL)).resolve(), target_is_directory=True)


if __name__ == "__main__":
    main()
