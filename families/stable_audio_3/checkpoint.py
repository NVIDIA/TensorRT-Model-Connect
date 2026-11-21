# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the two safetensors files of a Stable Audio 3 checkpoint.

``model.safetensors`` holds the conditioner, the DiT and the SAME autoencoder in
FP32; ``t5gemma-b-b-ul2/model.safetensors`` holds the T5Gemma encoder-decoder in
BF16, of which only the encoder is read.
"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

from .config import TEXT_ENCODER_DIR

DIT_PREFIX = "model.model."
DECODER_PREFIX = "pretransform.model.decoder."
BOTTLENECK_PREFIX = "pretransform.model.bottleneck."
CONDITIONER_PREFIX = "conditioner.conditioners."
TEXT_ENCODER_PREFIX = "model.encoder."


def _open_safetensors(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"stable_audio_3 checkpoint is missing {path}")
    return safe_open(str(path), framework="numpy")


class Weights:
    """Tensors of one prefix, converted to float32 on first use."""

    def __init__(self, path: Path, prefix: str) -> None:
        self._path = Path(path)
        self._prefix = prefix
        with _open_safetensors(self._path) as reader:
            self._keys = {key[len(prefix):] for key in reader.keys() if key.startswith(prefix)}
        if not self._keys:
            raise ValueError(f"{self._path} has no tensors under {prefix!r}")

    def keys(self) -> set[str]:
        return set(self._keys)

    def get_many(self, names: list[str]) -> dict[str, np.ndarray]:
        missing = [name for name in names if name not in self._keys]
        if missing:
            raise KeyError(f"{self._path.name} is missing {self._prefix}{missing[0]}")
        result = {}
        with _open_safetensors(self._path) as reader:
            for name in names:
                tensor = reader.get_tensor(self._prefix + name)
                if tensor.dtype == ml_dtypes.bfloat16:
                    tensor = tensor.astype(np.float32)
                elif tensor.dtype != np.float32:
                    raise TypeError(f"{self._prefix}{name} has unexpected dtype {tensor.dtype}")
                result[name] = tensor
        return result

    def get(self, name: str) -> np.ndarray:
        return self.get_many([name])[name]


def model_weights(model_dir: Path, prefix: str) -> Weights:
    return Weights(Path(model_dir) / "model.safetensors", prefix)


def text_encoder_weights(model_dir: Path) -> Weights:
    return Weights(Path(model_dir) / TEXT_ENCODER_DIR / "model.safetensors", TEXT_ENCODER_PREFIX)


def weight_norm(gain: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Fold ``torch.nn.utils.weight_norm`` (dim 0): ``g * v / ||v||`` per output channel."""
    norm = np.sqrt(np.sum(np.square(direction.astype(np.float64)), axis=tuple(range(1, direction.ndim)),
                          keepdims=True))
    return (gain.astype(np.float64) * direction.astype(np.float64) / norm).astype(np.float32)
