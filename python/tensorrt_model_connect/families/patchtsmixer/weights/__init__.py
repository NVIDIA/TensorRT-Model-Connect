# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTSMixer checkpoint readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision in {"fp16", "bf16"} else np.float32


class WeightDict(dict):
    """PatchTSMixer tensors keyed by their checkpoint names."""


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

    def keys(self) -> list[str]:
        return list(self._state)

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {key: reader for reader in readers for key in reader.keys()}
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    framework = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework=framework)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework=framework) for shard in shard_files
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map={name: readers_by_file[shard] for name, shard in weight_map.items()},
        )

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return _ReaderCollection([_TorchBinReader(bin_path)])
    raise FileNotFoundError(
        f"No PatchTSMixer safetensors or pytorch_model.bin found in {model_dir}"
    )


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


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
