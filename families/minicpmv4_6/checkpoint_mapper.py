# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the checkpoint's safetensors tensors as FP32 NumPy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Register NumPy bfloat16 support required by safetensors.
import ml_dtypes  # noqa: F401

from safetensors import safe_open

from .config import load_json


class WeightDict(dict):
    """Graph weight name -> FP32 array; projections are stored ``[in, out]``."""


class _ReaderCollection(list):
    """Checkpoint readers with one exact tensor-to-reader index."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {name: reader for reader in readers for name in reader.keys()}
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    """Open ``model.safetensors`` or the shards named by its index."""
    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([safe_open(str(single), framework="numpy")])
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = load_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (not isinstance(weight_map, dict) or not weight_map
                or not all(isinstance(name, str) and isinstance(shard, str) and shard
                           for name, shard in weight_map.items())):
            raise ValueError("model.safetensors.index.json needs a weight_map of tensor names to shard files")
        readers = {shard: safe_open(str(model_dir / shard), framework="numpy")
                   for shard in sorted({str(value) for value in weight_map.values()})}
        return _ReaderCollection(
            list(readers.values()),
            tensor_map={str(name): readers[str(shard)] for name, shard in weight_map.items()},
        )
    raise FileNotFoundError(f"No safetensors checkpoint in {model_dir}")


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


def _load_tensor(readers: _ReaderCollection, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """One weight as FP32; its dtype and shape are checked from the header before it is read."""
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    header = reader.get_slice(name)
    if header.get_dtype() not in _FLOAT_DTYPES:
        raise ValueError(f"{name}: dtype {header.get_dtype()} is not a BF16, FP16 or FP32 weight")
    if tuple(header.get_shape()) != tuple(shape):
        raise ValueError(f"{name}: expected {tuple(shape)}, got {tuple(header.get_shape())}")
    value = np.asarray(reader.get_tensor(name), dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} holds NaN or infinite values")
    return value


def decay_rates(a_log: np.ndarray, name: str) -> np.ndarray:
    """The DeltaNet decay rates ``-exp(A_log)`` in FP32, as the reference computes them; they must stay finite."""
    with np.errstate(over="ignore"):
        rates = -np.exp(np.asarray(a_log, dtype=np.float32))
    if not np.isfinite(rates).all():
        raise ValueError(f"MiniCPM-V 4.6 {name} overflows exp() in FP32")
    return rates


def narrowed(values, dtype, what: str = "weights") -> np.ndarray:
    """``values`` as a contiguous array of an engine constant's dtype; a finite value may not leave its range.

    BF16 and FP32 checkpoint values above 65504 would become infinite FP16 constants.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(values, dtype=dtype)
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"MiniCPM-V 4.6 {what} of shape {array.shape} leave the {array.dtype.name} range")
    return array
