# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the family-owned LeRobot ACT graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open


def load_checkpoint(model_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot ACT checkpoint not found: {path}")
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as reader:
        for name in reader.keys():
            # The VAE encoder is training-only. Inference always uses the
            # deterministic all-zero latent path.
            if name.startswith("model.vae_encoder"):
                continue
            weights[name] = np.ascontiguousarray(reader.get_tensor(name), dtype=np.float32)
    if not weights:
        raise ValueError("LeRobot ACT checkpoint contains no inference weights")
    return weights
