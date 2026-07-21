# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPT checkpoint readers and tensor conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass


class WeightDict(dict):
    """Flat, family-owned weights consumed by the OPT graph builders."""


def _transpose_2d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for {name}, got rank {array.ndim}")
    return np.ascontiguousarray(array.T, dtype=np.float32)


class _TorchBinReader:
    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self):
        return self._state.keys()

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    def __init__(self, readers: list, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        self.tensor_map = tensor_map or {
            name: reader for reader in readers for name in reader.keys()
        }


def _safetensors_framework() -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy"
    return "torch"


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    framework = _safetensors_framework()
    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([safe_open(str(single), framework=framework)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        readers = {
            shard: safe_open(str(model_dir / shard), framework=framework)
            for shard in sorted(set(weight_map.values()))
        }
        return _ReaderCollection(
            list(readers.values()),
            {name: readers[shard] for name, shard in weight_map.items()},
        )

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return _ReaderCollection([_TorchBinReader(bin_path)])
    raise FileNotFoundError(f"No OPT checkpoint found in {model_dir}")


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        if str(getattr(tensor, "dtype", "")) == "torch.float32":
            return tensor.numpy()
        return tensor.float().numpy()
    if tensor.dtype == np.uint16 or str(tensor.dtype) == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
