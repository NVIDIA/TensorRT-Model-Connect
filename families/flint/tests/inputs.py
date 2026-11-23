# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the manifest-pinned checkpoint and family-owned native helpers."""

import json
import os
from pathlib import Path

MANIFEST = json.loads((Path(__file__).parent / "manifests/flint-bf16.json").read_text())


def checkpoint_dir():
    explicit = os.environ.get("TRTMC_FLINT_CHECKPOINT")
    if explicit is not None:
        if not explicit:
            raise ValueError("TRTMC_FLINT_CHECKPOINT must name a checkpoint directory")
        root = Path(explicit)
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        ))
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Flint checkpoint has no config.json: {root}")
    return root


def native_binary(name):
    root = os.environ.get("TRTMC_NATIVE_BUILD_DIR") or os.environ.get("TRTMC_RUNTIME_ROOT")
    if not root:
        raise RuntimeError("set TRTMC_NATIVE_BUILD_DIR to the Flint CTest build directory")
    binary = Path(root) / name
    if not binary.is_file():
        raise FileNotFoundError(f"build the {name} target: {binary}")
    return binary
