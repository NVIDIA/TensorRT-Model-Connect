# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy, byte-preserving access to the indexed Flash-Next checkpoint."""

import json
import math
import mmap
from pathlib import Path
import struct

import ml_dtypes
import numpy as np


_DTYPES = {
    "BF16": ml_dtypes.bfloat16,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "F16": np.float16,
    "F32": np.float32,
    "U8": np.uint8,
    "I8": np.int8,
    "I32": np.int32,
    "I64": np.int64,
}


class FlashCheckpoint:
    """Parse the index once, opening each used shard at most once per build."""

    def __init__(self, directory):
        self.directory = Path(directory)
        index = self.directory / "model.safetensors.index.json"
        self.keys_to_files = json.loads(index.read_text())["weight_map"]
        self._shards = None

    def __enter__(self):
        if self._shards is not None:
            raise RuntimeError("Flash-Next checkpoint is already open")
        self._shards = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        # Views retain their own mmap reference. Never unmap a row or a
        # network constant while a consumer may still hold it.
        self._shards = None

    def _shard(self, key):
        if self._shards is None:
            raise RuntimeError("Flash-Next checkpoint is not open")
        name = self.keys_to_files[key]
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid checkpoint shard for {key}")
        if name not in self._shards:
            with (self.directory / name).open("rb") as stream:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            if len(mapping) < 8:
                raise ValueError(f"Truncated Flash-Next checkpoint shard: {name}")
            size = struct.unpack_from("<Q", mapping)[0]
            if not 0 < size <= min(100_000_000, len(mapping) - 8):
                raise ValueError(f"Invalid Flash-Next shard header size: {name}")
            header = json.loads(mapping[8 : 8 + size])
            self._shards[name] = header, mapping, 8 + size
        return self._shards[name]

    def array(self, key):
        """Return a read-only NumPy view; no table or expert-bank copy."""
        header, mapping, start = self._shard(key)
        item = header[key]
        dtype = np.dtype(_DTYPES[item["dtype"]]).newbyteorder("<")
        shape = tuple(item["shape"])
        first, end = item["data_offsets"]
        if (
            type(first) is not int
            or type(end) is not int
            or any(type(n) is not int or n < 0 for n in shape)
            or not 0 <= first <= end <= len(mapping) - start
            or end - first != math.prod(shape) * dtype.itemsize
        ):
            raise ValueError(f"Invalid Flash-Next tensor extent: {key}")
        return np.ndarray(shape, dtype=dtype, buffer=mapping, offset=start + first)

    def tensor(self, key, dtype, shape):
        item = self.array(key)
        if item.dtype != dtype or item.shape != tuple(shape):
            raise ValueError(f"Invalid Flash-Next checkpoint tensor: {key}")
        return item

    def bf16(self, key, shape, *, fp32=False):
        param = self.tensor(key, ml_dtypes.bfloat16, shape)
        return param.astype(np.float32 if fp32 else np.float16)
