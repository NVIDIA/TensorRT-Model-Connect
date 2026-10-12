# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned family-owned Lance reference checkout."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "https://github.com/bytedance/Lance.git"
REVISION = "4baeee086648996f6ab12e673cbe461b0b149997"


def _prepare_image_only_checkout(repository: Path) -> None:
    dataset = repository / "data/datasets_custom/validation_dataset.py"
    text = dataset.read_text(encoding="utf-8")
    for line in ("import decord\n", "from decord import VideoReader\n"):
        if line in text:
            if text.count(line) != 1:
                raise RuntimeError(f"Lance upstream import contract changed: {line.strip()}")
            text = text.replace(line, "")
    annotation = "video: VideoReader"
    if annotation in text:
        if text.count(annotation) != 1:
            raise RuntimeError("Lance upstream VideoReader annotation contract changed")
        text = text.replace(annotation, "video")
    dataset.write_text(text, encoding="utf-8")


def main() -> None:
    parent = Path(sys.prefix) / "trtmc-reference"
    destination = parent / "Lance"
    if (destination / "inference_lance.py").is_file():
        _prepare_image_only_checkout(destination)
        return
    if destination.exists():
        raise RuntimeError(f"incomplete Lance reference: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="Lance.", dir=parent))
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
        _prepare_image_only_checkout(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
