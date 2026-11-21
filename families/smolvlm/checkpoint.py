# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy SmolVLM safetensors access, single-file or sharded, by checkpoint key."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _plain_file_name(name: object) -> str:
    if (not isinstance(name, str) or not name or Path(name).name != name
            or name in {".", ".."} or "\\" in name or "\0" in name):
        raise ValueError(f"SmolVLM shard name must be a plain file name: {name!r}")
    return name


class CheckpointWeights:
    """Map every checkpoint key to its shard and read tensors as float32 arrays."""

    def __init__(self, model_dir: Path):
        self.root = Path(model_dir)
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("SmolVLM model.safetensors.index.json needs a nonempty weight_map")
            self.mapping = {key: _plain_file_name(shard) for key, shard in mapping.items()}
        else:
            from safetensors import safe_open

            if not (self.root / "model.safetensors").is_file():
                raise ValueError("SmolVLM checkpoint needs model.safetensors or its index")
            with safe_open(self.root / "model.safetensors", framework="pt", device="cpu") as stream:
                self.mapping = {key: "model.safetensors" for key in stream.keys()}
        for shard in set(self.mapping.values()):
            if not (self.root / shard).is_file():
                raise ValueError(f"SmolVLM checkpoint shard is missing: {shard}")

    def keys(self) -> set[str]:
        return set(self.mapping)

    def __getitem__(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        if key not in self.mapping:
            raise KeyError(f"SmolVLM checkpoint has no tensor {key}")
        # Widen BF16/FP16 directly to FP32; the graph owns its computation dtype.
        with safe_open(self.root / self.mapping[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key).float().numpy()
