# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-NeoX checkpoint I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import ml_dtypes  # noqa: F401  # Register NumPy bfloat16 support.
except ImportError:
    pass

from safetensors import safe_open


def _transpose_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Convert an HF ``[out, in]`` matrix to TRT ``[in, out]`` layout."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=np.float32)


class WeightDict(dict):
    """GPT-NeoX weights in the family builder's logical naming scheme."""


def _detect_framework() -> str:
    """Use torch when available so safetensors can decode BF16 directly."""
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


class _ReaderCollection:
    def __init__(self, tensor_map: dict[str, object]):
        self.tensor_map = tensor_map


def _single_reader(reader) -> _ReaderCollection:
    return _ReaderCollection({name: reader for name in reader.keys()})


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    """Open a GPT-NeoX safetensors checkpoint or legacy PyTorch state dict."""
    framework = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _single_reader(safe_open(str(single), framework=framework))

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        readers = {
            shard: safe_open(str(model_dir / shard), framework=framework)
            for shard in sorted(set(weight_map.values()))
        }
        return _ReaderCollection({name: readers[shard] for name, shard in weight_map.items()})

    legacy = model_dir / "pytorch_model.bin"
    if legacy.exists():
        return _single_reader(_TorchBinReader(legacy))

    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}"
    )


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        return tensor.numpy() if str(tensor.dtype) == "torch.float32" else tensor.float().numpy()
    if tensor.dtype == np.uint16 or str(tensor.dtype) == "bfloat16":
        return (tensor.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    try:
        reader = readers.tensor_map[name]
    except KeyError:
        raise KeyError(f"Tensor not found: {name}") from None
    return _to_numpy_fp32(reader.get_tensor(name))
