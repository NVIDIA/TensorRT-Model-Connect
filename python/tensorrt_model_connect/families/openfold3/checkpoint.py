# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned OpenFold3 checkpoint loading for family-owned TensorRT builders."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import PINNED_OPENFOLD3, PinnedArtifact


_VALIDATED_CHECKPOINTS: set[tuple[Path, int, int]] = set()


@dataclass(frozen=True)
class PairformerConfig:
    """Topology pinned by OpenFold3 v0.5.0's ``model_1`` preset."""

    token_s: int = 384
    token_z: int = 128
    num_blocks: int = 48
    num_heads: int = 16
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4


def _checkpoint_identity(path: Path) -> tuple[Path, int, int]:
    stat = path.stat()
    return path.resolve(strict=True), stat.st_size, stat.st_mtime_ns


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a large asset into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path: Path, expected: PinnedArtifact) -> None:
    """Fail closed unless *path* has the pinned byte count and digest."""

    actual_size = path.stat().st_size
    if actual_size != expected.size_bytes:
        raise ValueError(
            f"OpenFold3 {expected.filename} size mismatch: {actual_size} != {expected.size_bytes}"
        )
    actual_digest = sha256(path)
    if actual_digest != expected.sha256:
        raise ValueError(
            f"OpenFold3 {expected.filename} SHA-256 mismatch: {actual_digest} != {expected.sha256}"
        )


def validate_structure_checkpoint(path: Path) -> None:
    """Validate and remember the exact pinned OpenBind checkpoint."""

    validate_artifact(path, PINNED_OPENFOLD3.checkpoint)
    _VALIDATED_CHECKPOINTS.add(_checkpoint_identity(path))


def _load_checkpoint(checkpoint_path: Path, *, verify: bool) -> Mapping[str, Any]:
    if verify:
        validate_structure_checkpoint(checkpoint_path)
    elif _checkpoint_identity(checkpoint_path) not in _VALIDATED_CHECKPOINTS:
        raise ValueError(
            "OpenFold3 checkpoint verification may only be reused after the exact "
            "file was validated in this process"
        )

    import torch

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(state, Mapping):
        raise ValueError("OpenFold3 checkpoint root must be a tensor mapping")
    if tuple(float(value) for value in state.get("version_tensor", ())) != (2.0, 0.0, 0.0):
        raise ValueError("OpenFold3 checkpoint has an unsupported model version")
    return state


def load_weight_prefixes(
    checkpoint_path: Path,
    prefixes: tuple[str, ...],
    *,
    verify: bool = True,
) -> dict[str, np.ndarray]:
    """Load exact family-owned prefixes as contiguous FP32 host arrays."""

    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("OpenFold3 checkpoint prefixes must be non-empty")
    state = _load_checkpoint(checkpoint_path, verify=verify)
    selected = {
        name: np.ascontiguousarray(value.detach().cpu().float().numpy())
        for name, value in state.items()
        if isinstance(name, str) and name.startswith(prefixes)
    }
    missing = [
        prefix for prefix in prefixes if not any(name.startswith(prefix) for name in selected)
    ]
    if missing:
        raise ValueError(f"OpenFold3 checkpoint prefixes are missing: {', '.join(missing)}")
    return selected


def load_pairformer_weights(
    checkpoint_path: Path,
    *,
    first_block: int,
    block_count: int,
    verify: bool = True,
) -> dict[str, np.ndarray]:
    """Load a validated contiguous range from the 48-block Pairformer."""

    if first_block < 0 or block_count <= 0 or first_block + block_count > 48:
        raise ValueError("OpenFold3 Pairformer range must lie within [0, 48)")
    return load_weight_prefixes(
        checkpoint_path,
        tuple(
            f"pairformer_stack.blocks.{index}."
            for index in range(first_block, first_block + block_count)
        ),
        verify=verify,
    )
