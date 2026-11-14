# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only memory-mapped safetensors views that keep tensors in their stored precision."""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import ml_dtypes
import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16}


class _Shard:
    def __init__(self, path: Path):
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            (length,) = struct.unpack("<Q", prefix)
            if length > path.stat().st_size - 8:
                raise ValueError(f"Invalid safetensors header length: {path}")
            self.header = json.loads(stream.read(length))
            self.mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.offset = 8 + length

    def tensor(self, name: str) -> np.ndarray:
        item = self.header[name]
        if item["dtype"] not in _DTYPES:
            raise ValueError(f"Unsupported MiniCPM-V 4.7 tensor dtype {item['dtype']} for {name}")
        dtype = np.dtype(_DTYPES[item["dtype"]])
        begin, end = item["data_offsets"]
        shape = tuple(item["shape"])
        if (any(type(n) is not int or n < 0 for n in shape)
                or not 0 <= begin <= end <= len(self.mapping) - self.offset
                or int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != end - begin):
            raise ValueError(f"Invalid tensor extent for {name}")
        return np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=self.offset + begin)


class Checkpoint:
    """Exact tensor-name index over every shard of one checkpoint directory."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        single = model_dir / "model.safetensors"
        index = model_dir / "model.safetensors.index.json"
        if single.is_file():
            shard = _Shard(single)
            self._index = {name: shard for name in shard.header if name != "__metadata__"}
        elif index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("model.safetensors.index.json has no weight_map")
            shards = {name: _Shard(model_dir / name) for name in sorted(set(weight_map.values()))}
            self._index = {}
            for name, shard_name in weight_map.items():
                if name not in shards[shard_name].header:
                    raise ValueError(f"{shard_name} does not contain indexed tensor {name}")
                self._index[name] = shards[shard_name]
        else:
            raise FileNotFoundError(f"No safetensors checkpoint in {model_dir}")
        self._used: set[str] = set()

    def names(self) -> set[str]:
        return set(self._index)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def get(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Return a zero-copy view of one tensor in its stored dtype."""
        shard = self._index.get(name)
        if shard is None:
            raise KeyError(f"MiniCPM-V 4.7 checkpoint is missing {name}")
        value = shard.tensor(name)
        if shape is not None and value.shape != tuple(shape):
            raise ValueError(f"{name} has shape {value.shape}, expected {tuple(shape)}")
        self._used.add(name)
        return value

    def unused(self) -> list[str]:
        return sorted(set(self._index) - self._used)
