# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the family-owned LeRobot ACT graph."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from safetensors import safe_open

CHECKPOINT_SHA256 = "42772891cb6eba1e7bc36ad8e12c0fa0723c61f036fa235c725ce6026e6e81df"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(model_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot ACT checkpoint not found: {path}")
    actual_digest = checkpoint_sha256(path)
    if actual_digest != CHECKPOINT_SHA256:
        raise ValueError(
            "Unsupported LeRobot ACT checkpoint digest: "
            f"expected {CHECKPOINT_SHA256}, got {actual_digest}"
        )

    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as reader:
        for name in reader.keys():
            # The VAE encoder is training-only. Inference always uses the
            # deterministic all-zero latent path.
            if name.startswith("model.vae_encoder"):
                continue
            weights[name] = np.ascontiguousarray(reader.get_tensor(name), dtype=np.float32)
    return weights
