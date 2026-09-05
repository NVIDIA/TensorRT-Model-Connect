# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the exact safetensors checkpoint owned by timm VGG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ml_dtypes  # noqa: F401
from safetensors import safe_open


WeightDict = dict[str, np.ndarray]


class _ReaderCollection:
    def __init__(self, readers: list, tensor_map: dict[str, object]):
        self.readers = readers
        self.tensor_map = tensor_map


def _target_np_dtype(precision: str) -> np.dtype:
    if precision == "fp16":
        return np.float16
    if precision == "fp32":
        return np.float32
    raise ValueError(f"Unsupported timm_vgg precision: {precision}")


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    single = model_dir / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework="numpy")
        return _ReaderCollection([reader], {name: reader for name in reader.keys()})

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing timm VGG safetensors checkpoint in {model_dir}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    shard_files = sorted(set(str(value) for value in weight_map.values()))
    readers = {shard: safe_open(str(model_dir / shard), framework="numpy") for shard in shard_files}
    return _ReaderCollection(
        list(readers.values()),
        {str(name): readers[str(shard)] for name, shard in weight_map.items()},
    )


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return np.asarray(reader.get_tensor(name), dtype=np.float32)
