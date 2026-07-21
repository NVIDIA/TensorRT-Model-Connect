# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 checkpoint readers."""

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
    """GPT-2 graph weights keyed by their family-local tensor names."""


def _detect_framework() -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy"
    return "torch"


class _TorchBinReader:
    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self) -> list[str]:
        return list(self._state)

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    def __init__(self, readers: list, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        self.tensor_map = tensor_map or {key: reader for reader in readers for key in reader.keys()}


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    framework = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework=framework)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        readers = {
            shard: safe_open(str(model_dir / shard), framework=framework)
            for shard in sorted(set(weight_map.values()))
        }
        return _ReaderCollection(
            list(readers.values()),
            {name: readers[shard] for name, shard in weight_map.items()},
        )

    pytorch_bin = model_dir / "pytorch_model.bin"
    if pytorch_bin.exists():
        return _ReaderCollection([_TorchBinReader(pytorch_bin)])
    raise FileNotFoundError(f"No model.safetensors, index, or pytorch_model.bin in {model_dir}")


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        if str(getattr(tensor, "dtype", None)) == "torch.float32":
            return tensor.numpy()
        return tensor.float().numpy()

    dtype = str(tensor.dtype)
    if tensor.dtype == np.uint16 or dtype == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    if dtype == "float16":
        return tensor.astype(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    try:
        reader = readers.tensor_map[name]
    except KeyError:
        raise KeyError(f"Tensor not found: {name}") from None
    return _to_numpy_fp32(reader.get_tensor(name))
