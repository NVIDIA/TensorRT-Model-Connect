# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the original VibeVoice safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy for safetensors)
import numpy as np


def _open_safetensors(path: Path):
    from safetensors import safe_open

    return safe_open(str(path), framework="numpy")


class Checkpoint:
    """Lazy, name-addressed access to the tensors of one sharded checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        root = Path(model_dir)
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"vibevoice checkpoint has no model.safetensors.index.json: {root}"
            )
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self._shard_of = dict(weight_map)
        self._handles = {}
        for shard in sorted(set(weight_map.values())):
            path = root / shard
            if not path.is_file():
                raise FileNotFoundError(f"vibevoice checkpoint shard is missing: {path}")
            self._handles[shard] = _open_safetensors(path)

    def names(self) -> list[str]:
        return sorted(self._shard_of)

    def get(self, name: str) -> np.ndarray:
        shard = self._shard_of.get(name)
        if shard is None:
            raise KeyError(f"vibevoice checkpoint is missing tensor {name}")
        return self._handles[shard].get_tensor(name)

    def f32(self, name: str) -> np.ndarray:
        return np.asarray(self.get(name)).astype(np.float32)
