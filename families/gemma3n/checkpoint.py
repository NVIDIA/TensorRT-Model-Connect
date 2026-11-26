# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Gemma 3n safetensors shards through the checkpoint's own index."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import ml_dtypes
import numpy as np

_INDEX = "model.safetensors.index.json"
_SINGLE = "model.safetensors"


class Checkpoint:
    """Lazy, read-only access to one Gemma 3n safetensors checkpoint.

    Tensors are returned in their stored dtype (BF16 as ``ml_dtypes.bfloat16``);
    the graph that consumes a tensor decides its computation precision.
    """

    def __init__(self, model_dir: Path):
        self.root = Path(model_dir)
        index = self.root / _INDEX
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Gemma 3n safetensors index has no weight_map")
        elif (self.root / _SINGLE).is_file():
            from safetensors import safe_open

            with safe_open(self.root / _SINGLE, framework="numpy") as stream:
                mapping = {name: _SINGLE for name in stream.keys()}
        else:
            raise ValueError(f"Gemma 3n checkpoint has no {_INDEX} or {_SINGLE}: {self.root}")
        for name, shard in mapping.items():
            relative = PurePosixPath(shard) if isinstance(shard, str) else None
            # Shards are plain files next to the index, never paths elsewhere.
            if (not isinstance(name, str) or relative is None or len(relative.parts) != 1
                    or relative.name in {".", ".."} or not shard.endswith(".safetensors")
                    or not (self.root / shard).is_file()):
                raise ValueError(f"Gemma 3n checkpoint shard is missing or not local: {shard!r}")
        self.mapping = mapping

    def has(self, name: str) -> bool:
        return name in self.mapping

    def get(self, name: str) -> np.ndarray:
        from safetensors import safe_open

        if name not in self.mapping:
            raise KeyError(f"Gemma 3n checkpoint has no tensor {name!r}")
        with safe_open(self.root / self.mapping[name], framework="numpy") as stream:
            value = stream.get_tensor(name)
        if value.dtype not in (np.float32, np.float16, ml_dtypes.bfloat16):
            raise ValueError(f"Gemma 3n tensor {name} has unsupported dtype {value.dtype}")
        return value

    def get_f32(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Exact widening to float32 (BF16 and FP16 values are representable)."""
        value = self.get(name).astype(np.float32)
        if shape is not None and value.shape != tuple(shape):
            raise ValueError(f"Gemma 3n tensor {name} has shape {value.shape}, expected {shape}")
        return value
