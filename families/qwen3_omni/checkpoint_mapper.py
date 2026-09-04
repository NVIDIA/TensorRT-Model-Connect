# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact NumPy safetensors reader for the pinned Qwen3-Omni checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open


class WeightDict(dict):
    """Family-owned logical TensorRT weights."""


class _ReaderCollection:
    def __init__(self, tensor_map: dict[str, object]):
        self.tensor_map = tensor_map


def _target_np_dtype(precision: str) -> np.dtype:
    if precision != "bf16":
        raise ValueError("Qwen3-Omni checkpoint mapping supports only bf16")
    return np.dtype(ml_dtypes.bfloat16)


def _transpose_2d(array: np.ndarray, name: str, precision: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Qwen3-Omni tensor {name} must be rank 2")
    return np.ascontiguousarray(array.T, dtype=_target_np_dtype(precision))


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Qwen3-Omni requires model.safetensors.index.json: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Qwen3-Omni safetensors index has no weight_map")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != 15 or not all(isinstance(name, str) for name in shard_names):
        raise ValueError("Qwen3-Omni checkpoint requires exactly 15 safetensors shards")
    readers = {}
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Qwen3-Omni checkpoint shard is missing: {shard_path}")
        readers[shard_name] = safe_open(str(shard_path), framework="numpy")
    tensor_map: dict[str, object] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or shard_name not in readers:
            raise ValueError("Qwen3-Omni safetensors index contains an invalid entry")
        tensor_map[tensor_name] = readers[shard_name]
    return _ReaderCollection(tensor_map)


def _get_tensor(readers: _ReaderCollection, name: str):
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Qwen3-Omni checkpoint tensor is missing: {name}")
    return reader.get_tensor(name)


def _load_tensor_as_dtype(readers: _ReaderCollection, name: str, dtype: np.dtype) -> np.ndarray:
    return np.array(_get_tensor(readers, name), dtype=dtype, order="C", copy=True)


def _load_transposed_tensor(
    readers: _ReaderCollection,
    name: str,
    transpose_name: str,
    dtype: np.dtype,
) -> np.ndarray:
    source = np.asarray(_get_tensor(readers, name))
    if source.ndim != 2:
        raise ValueError(f"Qwen3-Omni tensor {transpose_name} must be rank 2")
    return np.array(source.T, dtype=dtype, order="C", copy=True)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    return np.asarray(_get_tensor(readers, name), dtype=np.float32)
