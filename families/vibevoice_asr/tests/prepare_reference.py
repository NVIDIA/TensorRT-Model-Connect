# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the official VibeVoice-ASR reference used by ``test_e2e.py``.

The checkpoint's own code (``reference-source.json``) needs transformers 4.x,
while the repository pins 5.x, so the reference runs in a separate virtual
environment that reuses the host's torch and numpy. Usage::

    python3 families/vibevoice_asr/tests/prepare_reference.py DESTINATION

then export the two printed variables before selecting the E2E cases.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# transformers is the version the checkpoint's config.json records.
REFERENCE_REQUIREMENTS = (
    "transformers==4.57.6",
    "huggingface_hub==0.36.0",
    "tokenizers==0.22.1",
    "accelerate==1.11.0",
    "diffusers==0.35.2",
    "librosa==0.11.0",
    "soundfile==0.13.1",
)


def _source(destination: Path) -> Path:
    pin = json.loads((Path(__file__).with_name("reference-source.json")).read_text("utf-8"))
    source = destination / "VibeVoice"
    if (source / "vibevoice/modular/modeling_vibevoice_asr.py").is_file():
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != pin["revision"]:
            raise SystemExit(f"{source} is at {head}, expected {pin['revision']}")
        return source
    temporary = Path(tempfile.mkdtemp(prefix="VibeVoice.", dir=destination))
    try:
        url = f"https://github.com/{pin['repository']}.git"
        subprocess.run(["git", "init", "--quiet", str(temporary)], check=True)
        subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", url], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary),
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                pin["revision"],
            ],
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
    return source


def _environment(destination: Path) -> Path:
    venv = destination / "venv"
    python = venv / "bin/python"
    if not python.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True
        )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", *REFERENCE_REQUIREMENTS], check=True
    )
    return python


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: prepare_reference.py DESTINATION")
    destination = Path(argv[0]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = _source(destination)
    python = _environment(destination)
    print(f"export TRTMC_REFERENCE_SOURCE_DIR={source}")
    print(f"export TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON={python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
