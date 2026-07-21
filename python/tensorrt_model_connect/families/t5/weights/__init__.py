# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 checkpoint loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def _transpose_2d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(array.T, dtype=np.float32)


class WeightDict(dict):
    """T5 weights indexed by their family-owned logical names."""


def _framework() -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy"
    return "torch"


class _TorchReader:
    def __init__(self, path: Path):
        import torch

        self.state = torch.load(path, map_location="cpu", weights_only=True)

    def keys(self):
        return self.state.keys()

    def get_tensor(self, name: str):
        return self.state[name]


class _Readers(list):
    def __init__(self, readers: list, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        self.tensor_map = tensor_map or {
            name: reader for reader in readers for name in reader.keys()
        }


def _open_safetensors(model_dir: Path) -> _Readers:
    """Open a T5 safetensors checkpoint, including sharded checkpoints."""
    single = model_dir / "model.safetensors"
    if single.exists():
        return _Readers([safe_open(str(single), framework=_framework())])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        shards = {
            filename: safe_open(str(model_dir / filename), framework=_framework())
            for filename in sorted(set(weight_map.values()))
        }
        return _Readers(
            list(shards.values()),
            {name: shards[filename] for name, filename in weight_map.items()},
        )

    pytorch_bin = model_dir / "pytorch_model.bin"
    if pytorch_bin.exists():
        return _Readers([_TorchReader(pytorch_bin)])
    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}"
    )


def _has_tensor(readers: _Readers, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        return tensor.numpy() if str(tensor.dtype) == "torch.float32" else tensor.float().numpy()
    if tensor.dtype == np.uint16 or str(tensor.dtype) == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _Readers, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
