# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned SANA source and a local HF-compatible model directory."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import snapshot_download

REPOSITORY = "https://github.com/NVlabs/Sana.git"
REVISION = "59629fdf790850797cb657bad014fce432bd713d"
MODEL = "Efficient-Large-Model/SANA-WM_bidirectional"
MODEL_REVISION = "e96271d77398def8ebb9fc595e7c0056dc625ab7"
STAGE1_TEXT_ENCODER = "Efficient-Large-Model/gemma-2-2b-it"
STAGE1_TEXT_ENCODER_REVISION = "569d9809d0c8b6722d4d31b5a77a2ec7a400650a"
OPENCV_HEADLESS_VERSION = "4.11.0.86"


def _install_headless_opencv() -> None:
    # mmcv declares the GUI OpenCV distribution, whose Linux wheel requires
    # libxcb. The reference only uses image-processing APIs, so replace it
    # with the headless wheel inside this family environment.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", "opencv-python"],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            f"opencv-python-headless=={OPENCV_HEADLESS_VERSION}",
        ],
        check=True,
    )


def _materialize_snapshot(snapshot: Path, destination: Path) -> None:
    temporary = Path(tempfile.mkdtemp(prefix="SANA-model.", dir=destination.parent))
    try:
        for source in snapshot.rglob("*"):
            target = temporary / source.relative_to(snapshot)
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not source.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved = source.resolve(strict=True)
            try:
                os.link(resolved, target)
            except OSError:
                shutil.copy2(resolved, target)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _materialize_stage1_text_encoder(model: Path) -> None:
    destination = model / "stage1_text_encoder"
    required = (
        destination / "config.json",
        destination / "model.safetensors.index.json",
        destination / "tokenizer.json",
    )
    if all(path.is_file() for path in required):
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"incomplete SANA stage-1 text encoder: {destination}")
    snapshot = Path(
        snapshot_download(
            STAGE1_TEXT_ENCODER,
            revision=STAGE1_TEXT_ENCODER_REVISION,
        )
    ).resolve()
    _materialize_snapshot(snapshot, destination)


def main() -> None:
    _install_headless_opencv()
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
    if model.is_dir() and not model.is_symlink():
        _materialize_stage1_text_encoder(model)
        return
    if model.exists() or model.is_symlink():
        raise RuntimeError(f"incomplete SANA model directory: {model}")
    snapshot = Path(snapshot_download(MODEL, revision=MODEL_REVISION)).resolve()
    _materialize_snapshot(snapshot, model)
    _materialize_stage1_text_encoder(model)


if __name__ == "__main__":
    main()
