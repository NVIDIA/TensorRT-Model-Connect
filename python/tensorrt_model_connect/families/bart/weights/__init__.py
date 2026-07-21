# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART checkpoint readers and tensor conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open


def _transpose_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Convert a Hugging Face [out, in] projection to BART's [in, out] layout."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=np.float32)


class WeightDict(dict):
    """Flat BART weight map consumed by the family-owned builders."""


def _detect_framework() -> str:
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


class _TorchBinReader:
    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self):
        return self._state.keys()

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    def __init__(self, readers: list):
        super().__init__(readers)
        self.tensor_map = {name: reader for reader in readers for name in reader.keys()}


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    """Open a standard Hugging Face BART safetensors or PyTorch checkpoint."""
    framework = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework=framework)])

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        readers = [
            safe_open(str(model_dir / shard), framework=framework)
            for shard in sorted(set(weight_map.values()))
        ]
        return _ReaderCollection(readers)

    binary = model_dir / "pytorch_model.bin"
    if binary.exists():
        return _ReaderCollection([_TorchBinReader(binary)])

    raise FileNotFoundError(
        f"No model.safetensors, model.safetensors.index.json, or pytorch_model.bin in {model_dir}"
    )


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        return tensor.numpy() if str(tensor.dtype) == "torch.float32" else tensor.float().numpy()

    dtype = str(tensor.dtype)
    if tensor.dtype == np.uint16 or dtype == "bfloat16":
        return (tensor.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "float16":
        return tensor.astype(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    try:
        reader = readers.tensor_map[name]
    except KeyError:
        raise KeyError(f"Tensor not found: {name}") from None
    return _to_numpy_fp32(reader.get_tensor(name))
