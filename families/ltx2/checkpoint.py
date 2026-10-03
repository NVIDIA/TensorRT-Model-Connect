# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safetensors access for the LTX-2.5 diffusers checkpoint folders."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes  # noqa: F401 - registers the bf16 NumPy dtype used by safetensors
import numpy as np
from safetensors import safe_open

BF16 = ml_dtypes.bfloat16

_INDEX_NAMES = ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
_SINGLE_NAMES = ("model.safetensors", "diffusion_pytorch_model.safetensors")


class Checkpoint:
    """Name -> tensor access over one (possibly sharded) safetensors folder.

    ``get`` converts to the requested NumPy dtype with round-to-nearest-even, the
    same rounding torch applies when a pipeline loads a fp32 tensor as bf16.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self._readers: dict[str, object] = {}
        self._map: dict[str, str] = {}
        for name in _INDEX_NAMES:
            index = self.folder / name
            if index.is_file():
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                self._map = {str(k): str(v) for k, v in weight_map.items()}
                break
        else:
            for name in _SINGLE_NAMES:
                single = self.folder / name
                if single.is_file():
                    reader = self._open(name)
                    self._map = {k: name for k in reader.keys()}
                    break
            else:
                raise FileNotFoundError(f"no safetensors checkpoint in {self.folder}")

    def _open(self, shard: str):
        reader = self._readers.get(shard)
        if reader is None:
            reader = safe_open(str(self.folder / shard), framework="numpy")
            self._readers[shard] = reader
        return reader

    def keys(self) -> list[str]:
        return list(self._map)

    def has(self, name: str) -> bool:
        return name in self._map

    def get(self, name: str, dtype=BF16) -> np.ndarray:
        shard = self._map.get(name)
        if shard is None:
            raise KeyError(f"{self.folder.name}: tensor not found: {name}")
        arr = self._open(shard).get_tensor(name)
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return np.ascontiguousarray(arr)

    def maybe(self, name: str, dtype=BF16) -> np.ndarray | None:
        return self.get(name, dtype) if self.has(name) else None

    def config(self) -> dict:
        path = self.folder / "config.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
