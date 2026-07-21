# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint helpers for the Eagle VLM safetensor checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


class WeightDict(dict):
    """Eagle VLM weights keyed by their family-local graph names."""


def _transpose_2d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for {name}, got rank {array.ndim}")
    return np.ascontiguousarray(array.T, dtype=np.float32)


class _Readers(list):
    def __init__(self, readers: list):
        super().__init__(readers)
        self.by_name = {name: reader for reader in readers for name in reader.keys()}


def _open_safetensors(model_dir: Path) -> _Readers:
    """Open a single-file or indexed-shard Hugging Face checkpoint."""
    single = model_dir / "model.safetensors"
    if single.is_file():
        files = [single]
    else:
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"No Eagle VLM safetensors checkpoint in {model_dir}")
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        files = [model_dir / name for name in sorted(set(weight_map.values()))]

    try:
        import torch  # noqa: F401
    except ImportError:
        framework = "numpy"
    else:
        framework = "torch"
    return _Readers([safe_open(str(path), framework=framework) for path in files])


def _has_tensor(readers: _Readers, name: str) -> bool:
    return name in readers.by_name


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "float"):
        return tensor.float().numpy()
    if tensor.dtype == np.uint16 or str(tensor.dtype) == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _Readers, name: str) -> np.ndarray:
    reader = readers.by_name.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
