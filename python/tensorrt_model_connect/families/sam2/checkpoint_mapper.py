# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, weights-only loading for the supported SAM2 checkpoint."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_CHECKPOINT_SHA256 = "89fd676560809c8504411b574cea305c86db1f65bda790ec7fe16cedc6c6ff73"
PUBLIC_CHECKPOINT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
_MAX_CHECKPOINT_BYTES = (1 << 32) - 1


class CheckpointError(RuntimeError):
    """The supplied checkpoint is not the one supported by this builder."""


class Checkpoint:
    """Validated CPU tensors kept alive through TensorRT plan serialization."""

    def __init__(self, state_dict: dict[str, Any]) -> None:
        self._state_dict = state_dict

    def tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        try:
            value = self._state_dict[name]
        except KeyError as exc:
            return self._missing_tensor(name, shape, exc)

        import torch

        if not isinstance(value, torch.Tensor):
            raise CheckpointError(f"state_dict value {name!r} is not a tensor")
        if value.dtype is not torch.float32:
            raise CheckpointError(
                f"checkpoint tensor {name!r} has dtype {value.dtype}, expected torch.float32"
            )
        if tuple(value.shape) != shape:
            raise CheckpointError(
                f"checkpoint tensor {name!r} has shape {tuple(value.shape)}, expected {shape}"
            )
        if not value.is_contiguous():
            raise CheckpointError(f"checkpoint tensor is not contiguous: {name}")
        return value.detach().numpy()

    def _missing_tensor(self, name: str, shape: tuple[int, ...], cause: KeyError) -> np.ndarray:
        raise CheckpointError(f"checkpoint tensor not found: {name!r}") from cause


def _bbox_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for level in range(3):
        for branch in ("cls", "reg"):
            for stack in range(2):
                module = f"image_encoder.bbox_head.{branch}_convs.{level}.{stack}"
                shapes[f"{module}.conv.weight"] = (256, 256, 3, 3)
                for suffix in ("weight", "bias", "running_mean", "running_var"):
                    shapes[f"{module}.bn.{suffix}"] = (256,)
        shapes[f"image_encoder.bbox_head.rtm_cls.{level}.weight"] = (2, 256, 1, 1)
        shapes[f"image_encoder.bbox_head.rtm_cls.{level}.bias"] = (2,)
        shapes[f"image_encoder.bbox_head.rtm_reg.{level}.weight"] = (4, 256, 1, 1)
        shapes[f"image_encoder.bbox_head.rtm_reg.{level}.bias"] = (4,)
    return shapes


_PUBLIC_SYNTHETIC_BBOX_SHAPES = _bbox_shapes()


class PublicCoreCheckpoint(Checkpoint):
    """Pinned public SAM2 core with a deterministic, lazy bbox-head overlay."""

    def _missing_tensor(self, name: str, shape: tuple[int, ...], cause: KeyError) -> np.ndarray:
        expected = _PUBLIC_SYNTHETIC_BBOX_SHAPES.get(name)
        if expected is None:
            return super()._missing_tensor(name, shape, cause)
        if shape != expected:
            raise CheckpointError(
                f"synthetic bbox tensor {name!r} has requested shape {shape}, expected {expected}"
            )

        value = np.zeros(shape, dtype=np.float32)
        if name.endswith(".bn.weight") or name.endswith(".bn.running_var"):
            value.fill(1.0)
        elif name == "image_encoder.bbox_head.cls_convs.0.0.bn.bias":
            value[0] = 1.0
        elif name == "image_encoder.bbox_head.cls_convs.0.1.conv.weight":
            # With padded convolution over the constant first-stage channel,
            # only the top-left anchor omits both negative taps.
            value[0, 0, 0, 1] = -1.0
            value[0, 0, 1, 0] = -1.0
        elif name == "image_encoder.bbox_head.cls_convs.0.1.bn.bias":
            value[0] = 1.0
        elif name == "image_encoder.bbox_head.rtm_cls.0.weight":
            value[1, 0, 0, 0] = 64.0
        elif name == "image_encoder.bbox_head.rtm_cls.0.bias":
            value[:] = (-8.0, -30.0)
        elif name == "image_encoder.bbox_head.rtm_cls.1.bias" or name == (
            "image_encoder.bbox_head.rtm_cls.2.bias"
        ):
            value.fill(-8.0)
        elif name == "image_encoder.bbox_head.rtm_reg.0.bias":
            value[:] = (-15.5, -15.5, 111.5, 111.5)
        return value


def _snapshot(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError(f"unable to open checkpoint {path!s}: {exc}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise CheckpointError(f"checkpoint must be a regular file: {path!s}")
        if status.st_size <= 0:
            raise CheckpointError(f"checkpoint is empty: {path!s}")
        if status.st_size > _MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds the supported archive size")

        chunks: list[bytes] = []
        remaining = status.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024 * 1024))
            if not chunk:
                raise CheckpointError("checkpoint changed while creating its immutable snapshot")
            chunks.append(chunk)
            remaining -= len(chunk)
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino, current.st_size) != (
            status.st_dev,
            status.st_ino,
            status.st_size,
        ):
            raise CheckpointError("checkpoint changed while creating its immutable snapshot")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_state_dict(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Hash an immutable file snapshot, then load only tensors with PyTorch."""

    snapshot = _snapshot(Path(path))
    actual = hashlib.sha256(snapshot).hexdigest()
    if actual != expected_sha256:
        raise CheckpointError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )

    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("the SAM2 builder requires PyTorch") from exc
    try:
        root = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"unable to load authenticated checkpoint: {exc}") from exc
    if not isinstance(root, dict) or list(root) != ["model"]:
        raise CheckpointError("checkpoint root must be exactly {'model': state_dict}")
    state_dict = root["model"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise CheckpointError("checkpoint model value must be a nonempty state_dict")
    if len(state_dict) > 16384 or any(not isinstance(name, str) or not name for name in state_dict):
        raise CheckpointError("checkpoint state_dict has invalid tensor names or size")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise CheckpointError("checkpoint state_dict must contain only tensors")
    return state_dict


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Load the exact delivered checkpoint without any synthesized tensors."""

    return Checkpoint(_load_state_dict(path, SUPPORTED_CHECKPOINT_SHA256))


def load_public_core_checkpoint(path: str | Path) -> PublicCoreCheckpoint:
    """Load the pinned public core and enable only the explicit bbox overlay."""

    return PublicCoreCheckpoint(_load_state_dict(path, PUBLIC_CHECKPOINT_SHA256))
