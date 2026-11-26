# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the pinned offline inputs and separately built native test tools."""

import os
from pathlib import Path


def checkpoint_directory(hf_id, revision):
    name = "TRTMC_MOLMO2_MODEL_DIR"
    if name in os.environ:
        directory = Path(os.environ[name]).absolute()
        assert os.environ[name] and directory.is_dir(), f"{name} is not a directory: {directory}"
    else:
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(hf_id, revision=revision, local_files_only=True))
    assert (directory / "config.json").is_file(), (
        f"Molmo2 checkpoint has no config.json: {directory}"
    )
    return directory


def tokenizer_file(hf_id, revision):
    name = "TRTMC_MOLMO2_TOKENIZER_DIR"
    if name in os.environ:
        directory = Path(os.environ[name]).absolute()
        assert os.environ[name] and directory.is_dir(), f"{name} is not a directory: {directory}"
    else:
        directory = checkpoint_directory(hf_id, revision)
    path = directory / "tokenizer.json"
    assert path.is_file(), f"Molmo2 tokenizer is not a file: {path}"
    return path


def native_tool(name):
    variable = (
        "TRTMC_NATIVE_BUILD_DIR" if "TRTMC_NATIVE_BUILD_DIR" in os.environ else "TRTMC_RUNTIME_ROOT"
    )
    value = os.environ.get(variable)
    assert value, f"Molmo2 tests require {variable}"
    path = Path(value).absolute() / "families/molmo2" / name
    assert path.is_file() and os.access(path, os.X_OK), (
        f"Molmo2 native test tool is not executable: {path}"
    )
    return path
