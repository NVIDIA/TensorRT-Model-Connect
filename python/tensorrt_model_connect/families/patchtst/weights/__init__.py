# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST checkpoint readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open


def _target_np_dtype(precision: str) -> np.dtype:
    if precision in {"fp16", "bf16"}:
        return np.float16
    return np.float32


class WeightDict(dict):
    """PatchTST tensors keyed by their checkpoint names."""


def _detect_framework() -> str:
    """Use 'torch' if available (handles BF16 natively), else 'numpy'."""
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


class _TorchBinReader:
    """Expose a PyTorch state dict through the safetensors reader interface."""

    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self) -> list[str]:
        return list(self._state.keys())

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    """Reader list with a tensor-name lookup."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {}
            for reader in readers:
                for key in reader.keys():
                    tensor_map[key] = reader
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> list:
    """Open a PatchTST safetensors or PyTorch checkpoint."""
    fw = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework=fw)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json

        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework=fw) for shard in shard_files
        }
        tensor_map = {name: readers_by_file[shard] for name, shard in weight_map.items()}
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    bin_single = model_dir / "pytorch_model.bin"
    if bin_single.exists():
        return _ReaderCollection([_TorchBinReader(bin_single)])

    raise FileNotFoundError(f"No PatchTST safetensors or pytorch_model.bin found in {model_dir}")


def _to_numpy_fp32(t) -> np.ndarray:
    """Convert a safetensors/torch tensor to numpy float32 with minimal copies."""
    if hasattr(t, "numpy"):
        dtype = getattr(t, "dtype", None)
        if str(dtype) == "torch.float32":
            return t.numpy()
        return t.float().numpy()

    dtype_str = str(t.dtype)
    if t.dtype == np.uint16 or dtype_str == "bfloat16":
        t = t.view(np.uint16).astype(np.uint32) << 16
        return t.view(np.float32)
    if dtype_str == "float16":
        return t.astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _load_tensor(readers: list, name: str) -> np.ndarray:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return _to_numpy_fp32(reader.get_tensor(name))
    for r in readers:
        if name in r.keys():
            return _to_numpy_fp32(r.get_tensor(name))
    raise KeyError(f"Tensor not found: {name}")
