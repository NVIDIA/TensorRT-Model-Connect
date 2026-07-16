# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load SAM3 tracker parameters directly from safetensors into NumPy."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
from safetensors import safe_open


_PREFIX = "tracker_model."


def _as_float32(value: np.ndarray) -> np.ndarray:
    """Return a contiguous FP32 array, including for raw bfloat16 storage."""

    array = np.asarray(value)
    if array.dtype == np.uint16 or str(array.dtype) == "bfloat16":
        words = array.view(np.uint16).astype(np.uint32) << 16
        array = words.view(np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


class TrackerWeights(Mapping[str, np.ndarray]):
    """Lazy, tracker-only view over one or more safetensors readers.

    Keys retain their checkpoint names. Convolution kernels therefore remain
    OIHW/IOHW, while :meth:`linear_weight` transposes a learned projection from
    the checkpoint's ``[out, in]`` layout to TensorRT matmul ``[in, out]``.
    """

    def __init__(self, readers: Mapping[str, object]) -> None:
        self._readers = dict(readers)
        self._cache: dict[str, np.ndarray] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._readers)

    def __len__(self) -> int:
        return len(self._readers)

    def __getitem__(self, key: str) -> np.ndarray:
        full_key = key if key.startswith(_PREFIX) else _PREFIX + key
        if full_key not in self._readers:
            raise KeyError(f"Missing SAM3 tracker parameter: {full_key}")
        cached = self._cache.get(full_key)
        if cached is None:
            reader = self._readers[full_key]
            cached = _as_float32(reader.get_tensor(full_key))
            self._cache[full_key] = cached
        return cached

    def linear_weight(self, prefix: str) -> np.ndarray:
        weight = self[f"{prefix}.weight"]
        if weight.ndim != 2:
            raise ValueError(
                f"Expected rank-2 SAM3 tracker projection {prefix!r}, got {weight.shape}"
            )
        return np.ascontiguousarray(weight.T, dtype=np.float32)

    def linear_bias(self, prefix: str) -> np.ndarray:
        return self[f"{prefix}.bias"]


def _reader_map(model_dir: Path) -> dict[str, object]:
    single = model_dir / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework="numpy")
        return {key: reader for key in reader.keys() if key.startswith(_PREFIX)}

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(
            "SAM3 tracker plans require model.safetensors or model.safetensors.index.json"
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"Could not read SAM3 checkpoint index {index_path}") from error

    shard_names = sorted({shard for key, shard in weight_map.items() if key.startswith(_PREFIX)})
    readers = {shard: safe_open(str(model_dir / shard), framework="numpy") for shard in shard_names}
    return {key: readers[shard] for key, shard in weight_map.items() if key.startswith(_PREFIX)}


def load_tracker_weights(model_dir: str | Path) -> TrackerWeights:
    """Open the model's tracker-only safetensors view without a framework model."""

    resolved = Path(model_dir)
    weights = TrackerWeights(_reader_map(resolved))
    if not weights:
        raise RuntimeError(f"No {_PREFIX} parameters were found in {resolved}")
    return weights
