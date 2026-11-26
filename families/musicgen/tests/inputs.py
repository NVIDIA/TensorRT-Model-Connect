# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the family's pinned offline checkpoint and native tokenizer probe."""

import json
import os
from pathlib import Path

MANIFEST = json.loads((Path(__file__).parent / "manifests/musicgen-medium-fp32.json").read_text())


def checkpoint_dir(hf_id=MANIFEST["hf_id"], revision=MANIFEST["hf_revision"]):
    explicit = os.environ.get("TRTMC_MUSICGEN_MODEL_DIR")
    if explicit is not None:
        if not explicit:
            raise ValueError("TRTMC_MUSICGEN_MODEL_DIR must name a checkpoint directory")
        root = Path(explicit)
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                hf_id, revision=revision, local_files_only=True
            )
        )
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"MusicGen checkpoint has no config.json: {root}")
    return root


def tokenizer_binary():
    root = os.environ.get("TRTMC_NATIVE_BUILD_DIR") or os.environ.get("TRTMC_RUNTIME_ROOT")
    if not root:
        raise RuntimeError("set TRTMC_NATIVE_BUILD_DIR to the MusicGen CTest build directory")
    binary = Path(root) / "test_musicgen_tokenizer"
    if not binary.is_file():
        raise FileNotFoundError(f"build the test_musicgen_tokenizer target: {binary}")
    return binary
