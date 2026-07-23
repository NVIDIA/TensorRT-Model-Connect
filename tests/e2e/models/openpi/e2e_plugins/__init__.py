# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenPI-owned unified E2E plugins."""

from __future__ import annotations

import functools
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

OPENPI_SNAPSHOT_REPO_ID = "NVIDIA/TensorRT-Model-Connect-OpenPI-Pi05-DROID"
OPENPI_SNAPSHOT_REVISION = "59205df7225305b0a6680dd8fe9a064cfec774d7"
OPENPI_SNAPSHOT_ALLOW_PATTERNS = (
    "openpi_config.json",
    "preprocessor_config.json",
    "trtmc_openpi/**",
)


@functools.lru_cache(maxsize=1)
def openpi_snapshot_root() -> Path:
    """Return the immutable, already-cached OpenPI qualification snapshot."""

    root = (
        Path(
            snapshot_download(
                repo_id=OPENPI_SNAPSHOT_REPO_ID,
                revision=OPENPI_SNAPSHOT_REVISION,
                allow_patterns=list(OPENPI_SNAPSHOT_ALLOW_PATTERNS),
                local_files_only=True,
            )
        )
        .expanduser()
        .resolve(strict=True)
    )
    if not root.is_dir():
        raise FileNotFoundError(f"OpenPI Hugging Face snapshot is not a directory: {root}")
    return root


def openpi_snapshot_path(*parts: str) -> Path:
    """Resolve a fixed path below the pinned OpenPI snapshot."""

    relative = PurePosixPath(*parts)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"Invalid OpenPI snapshot-relative path: {relative}")
    return openpi_snapshot_root().joinpath(*relative.parts)


def openpi_proof_path(*parts: str) -> Path:
    """Resolve an OpenPI-only proof asset inside the pinned snapshot."""

    return openpi_snapshot_path("trtmc_openpi", *parts)


def resolve_model_asset(value: str, model_test_dir: str) -> Path:
    """Resolve a model-local path without searching unrelated workspaces."""

    path = Path(value)
    if path.is_absolute():
        return path
    if model_test_dir:
        return Path(model_test_dir) / path
    return Path(__file__).resolve().parents[1] / path
