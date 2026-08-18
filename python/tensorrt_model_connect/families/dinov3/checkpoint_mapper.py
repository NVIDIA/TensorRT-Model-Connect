# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 safetensors loading helpers.

The runtime graph consumes explicit NumPy arrays and never creates or imports
an ONNX graph.  Both the legacy Transformers ``layer.*`` layout and the newer
``model.layer.*`` layout are accepted because published DINOv3 checkpoints
span both Transformers naming generations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open


class WeightDict(dict[str, np.ndarray]):
    """Family-owned logical weight map."""


class _Readers(list):
    def __init__(self, readers: list, tensor_map: dict[str, object]):
        super().__init__(readers)
        self.tensor_map = tensor_map


def _framework() -> str:
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


def open_checkpoint(model_dir: str | Path) -> _Readers:
    root = Path(model_dir)
    framework = _framework()
    single = root / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework=framework)
        return _Readers([reader], {name: reader for name in reader.keys()})

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"DINOv3 requires model.safetensors or model.safetensors.index.json in {root}"
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    by_file = {
        shard: safe_open(str(root / shard), framework=framework)
        for shard in sorted(set(weight_map.values()))
    }
    return _Readers(
        list(by_file.values()),
        {name: by_file[shard] for name, shard in weight_map.items()},
    )


def has_tensor(readers: _Readers, name: str) -> bool:
    return name in readers.tensor_map


def load_tensor(readers: _Readers, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    value = reader.get_tensor(name)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            return value.numpy()
        except TypeError:
            return value.float().numpy()
    return np.asarray(value)


def _first_key(readers: _Readers, names: tuple[str, ...]) -> str:
    for name in names:
        if has_tensor(readers, name):
            return name
    raise KeyError("Tensor not found; tried: " + ", ".join(names))


def load_first(readers: _Readers, *names: str) -> np.ndarray:
    return load_tensor(readers, _first_key(readers, names))


def layer_key(readers: _Readers, layer: int, suffix: str) -> str:
    """Resolve the legacy or current HF encoder-layer prefix."""
    return _first_key(
        readers,
        (f"model.layer.{layer}.{suffix}", f"layer.{layer}.{suffix}"),
    )


def target_dtype(precision: str) -> np.dtype:
    if precision == "fp16":
        return np.dtype(np.float16)
    if precision == "fp32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported DINOv3 precision: {precision}")


def as_weight(value: np.ndarray, precision: str) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=target_dtype(precision))


def transpose_linear(value: np.ndarray, name: str, precision: str) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError(f"Expected rank-2 linear weight for {name}, got {value.shape}")
    return np.ascontiguousarray(value.T, dtype=target_dtype(precision))
