# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M2M-100/NLLB checkpoint readers and tensor transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass


def _transpose_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=np.float32)


class WeightDict(dict):
    """Flat M2M-100/NLLB builder weight mapping."""


def _safetensors_framework() -> str:
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


def _reader_map(reader) -> dict[str, object]:
    return {name: reader for name in reader.keys()}


def _open_safetensors(model_dir: Path) -> dict[str, object]:
    framework = _safetensors_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _reader_map(safe_open(str(single), framework=framework))

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        readers = {
            shard: safe_open(str(model_dir / shard), framework=framework)
            for shard in set(weight_map.values())
        }
        return {name: readers[shard] for name, shard in weight_map.items()}

    pytorch_bin = model_dir / "pytorch_model.bin"
    if pytorch_bin.exists():
        return _reader_map(_TorchBinReader(pytorch_bin))

    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}"
    )


def _has_tensor(readers: dict[str, object], name: str) -> bool:
    return name in readers


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        if str(getattr(tensor, "dtype", None)) == "torch.float32":
            return tensor.numpy()
        return tensor.float().numpy()

    dtype_name = str(tensor.dtype)
    if tensor.dtype == np.uint16 or dtype_name == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    if dtype_name == "float16":
        return tensor.astype(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: dict[str, object], name: str) -> np.ndarray:
    reader = readers.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
