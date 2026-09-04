# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact NumPy safetensors access for timm RepVGG checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class Checkpoint:
    readers: tuple[Any, ...]
    tensor_map: dict[str, Any]

    @classmethod
    def open(cls, model_dir: Path) -> "Checkpoint":
        single = model_dir / "model.safetensors"
        if single.is_file():
            reader = safe_open(str(single), framework="numpy")
            return cls((reader,), {str(name): reader for name in reader.keys()})

        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"RepVGG checkpoint has no model safetensors: {model_dir}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("RepVGG safetensors index has no weight_map")
        if not all(
            isinstance(name, str) and name and isinstance(shard, str) and shard
            for name, shard in weight_map.items()
        ):
            raise ValueError("RepVGG safetensors index contains invalid entries")
        shard_names = sorted(set(weight_map.values()))
        for name in shard_names:
            path = PurePosixPath(name)
            if path.is_absolute() or len(path.parts) != 1 or path.name != name:
                raise ValueError("RepVGG safetensors shard names must be direct relative files")
        readers = {
            name: safe_open(str(model_dir / name), framework="numpy") for name in shard_names
        }
        return cls(
            tuple(readers[name] for name in shard_names),
            {str(tensor): readers[str(shard)] for tensor, shard in weight_map.items()},
        )

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.tensor_map)

    def tensor(self, name: str) -> np.ndarray:
        reader = self.tensor_map.get(name)
        if reader is None:
            raise KeyError(f"RepVGG checkpoint tensor not found: {name}")
        return np.asarray(reader.get_tensor(name), dtype=np.float32)
