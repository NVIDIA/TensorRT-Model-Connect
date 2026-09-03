# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Boltz-2 checkpoint loading for family-owned TensorRT builders."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import PINNED_BOLTZ2, PinnedArtifact


@dataclass(frozen=True)
class PairformerConfig:
    token_s: int
    token_z: int
    num_blocks: int
    num_heads: int
    pairwise_head_width: int
    pairwise_num_heads: int
    post_layer_norm: bool
    v2: bool


PINNED_PAIRFORMER = PairformerConfig(
    token_s=384,
    token_z=128,
    num_blocks=64,
    num_heads=16,
    pairwise_head_width=32,
    pairwise_num_heads=4,
    post_layer_norm=False,
    v2=True,
)

_VALIDATED_CHECKPOINTS: set[tuple[Path, int, int]] = set()


def _checkpoint_identity(path: Path) -> tuple[Path, int, int]:
    stat = path.stat()
    return path.resolve(strict=True), stat.st_size, stat.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path: Path, expected: PinnedArtifact) -> None:
    """Fail closed unless *path* has an exact pinned size and digest."""

    actual_size = path.stat().st_size
    if actual_size != expected.size_bytes:
        raise ValueError(
            f"Boltz-2 {expected.filename} size mismatch: {actual_size} != {expected.size_bytes}"
        )
    actual_digest = _sha256(path)
    if actual_digest != expected.sha256:
        raise ValueError(
            f"Boltz-2 {expected.filename} SHA-256 mismatch: "
            f"{actual_digest} != {expected.sha256}"
        )


def validate_structure_checkpoint(path: Path) -> None:
    """Fail closed unless *path* is the exact pinned public checkpoint."""

    validate_artifact(path, PINNED_BOLTZ2.structure_checkpoint)
    _VALIDATED_CHECKPOINTS.add(_checkpoint_identity(path))


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Boltz-2 checkpoint {name} must be a positive integer")
    return int(value)


def resolve_pairformer_config(hparams: Mapping[str, Any]) -> PairformerConfig:
    """Read and strictly validate the Pairformer topology in checkpoint metadata."""

    raw = hparams.get("pairformer_args")
    if not isinstance(raw, Mapping):
        raise ValueError("Boltz-2 checkpoint is missing pairformer_args")
    config = PairformerConfig(
        token_s=_positive_int(hparams.get("token_s"), "token_s"),
        token_z=_positive_int(hparams.get("token_z"), "token_z"),
        num_blocks=_positive_int(raw.get("num_blocks"), "pairformer_args.num_blocks"),
        num_heads=_positive_int(raw.get("num_heads"), "pairformer_args.num_heads"),
        pairwise_head_width=_positive_int(
            raw.get("pairwise_head_width", 32),
            "pairformer_args.pairwise_head_width",
        ),
        pairwise_num_heads=_positive_int(
            raw.get("pairwise_num_heads", 4),
            "pairformer_args.pairwise_num_heads",
        ),
        post_layer_norm=bool(raw.get("post_layer_norm", False)),
        # The released checkpoint predates this construction selector. Boltz
        # v2.2.1's public inference entry point supplies PairformerArgsV2 and
        # therefore constructs the checkpoint weights with the v2 semantics.
        v2=bool(raw.get("v2", PINNED_PAIRFORMER.v2)),
    )
    if config != PINNED_PAIRFORMER:
        raise ValueError(
            "Boltz-2 checkpoint Pairformer topology differs from the qualified topology: "
            f"{config!r} != {PINNED_PAIRFORMER!r}"
        )
    return config


def load_pairformer_weights(
    checkpoint_path: Path,
    *,
    first_block: int = 0,
    block_count: int = 1,
    verify: bool = True,
) -> tuple[PairformerConfig, dict[str, np.ndarray]]:
    """Load a contiguous set of Pairformer blocks from the Lightning checkpoint.

    PyTorch is a bundle-creation dependency only. Returned weights are ordinary
    NumPy arrays consumed by TensorRT's network-definition API.
    """

    if first_block < 0 or block_count <= 0:
        raise ValueError("Boltz-2 Pairformer block range must be non-negative and non-empty")

    hparams, state = _load_checkpoint(checkpoint_path, verify=verify)
    config = resolve_pairformer_config(hparams)
    if first_block + block_count > config.num_blocks:
        raise ValueError(
            "Boltz-2 Pairformer block range exceeds the checkpoint topology: "
            f"{first_block} + {block_count} > {config.num_blocks}"
        )

    prefixes = tuple(
        f"pairformer_module.layers.{index}."
        for index in range(first_block, first_block + block_count)
    )
    selected: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if isinstance(name, str) and name.startswith(prefixes):
            selected[name] = np.ascontiguousarray(value.detach().cpu().float().numpy())
    if not selected:
        raise ValueError("Boltz-2 checkpoint contains no weights for the requested blocks")
    return config, selected


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    verify: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if verify:
        validate_structure_checkpoint(checkpoint_path)
    elif _checkpoint_identity(checkpoint_path) not in _VALIDATED_CHECKPOINTS:
        raise ValueError(
            "Boltz-2 checkpoint verification may only be reused after the exact "
            "file was validated in this process"
        )

    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Boltz-2 checkpoint root must be a mapping")
    hparams = checkpoint.get("hyper_parameters")
    state = checkpoint.get("state_dict")
    if not isinstance(hparams, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Boltz-2 checkpoint is missing metadata or state_dict")
    return hparams, state


def load_weight_prefixes(
    checkpoint_path: Path,
    prefixes: tuple[str, ...],
    *,
    verify: bool = True,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    """Load exact model-owned weight prefixes for another Boltz-2 subgraph."""

    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("Boltz-2 checkpoint prefixes must be non-empty")
    hparams, state = _load_checkpoint(checkpoint_path, verify=verify)
    selected = {
        name: np.ascontiguousarray(value.detach().cpu().float().numpy())
        for name, value in state.items()
        if isinstance(name, str) and name.startswith(prefixes)
    }
    missing = [
        prefix for prefix in prefixes if not any(name.startswith(prefix) for name in selected)
    ]
    if missing:
        raise ValueError(f"Boltz-2 checkpoint prefixes are missing: {', '.join(missing)}")
    return hparams, selected
