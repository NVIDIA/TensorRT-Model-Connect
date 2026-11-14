# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read checkpoint tensors at their stored precision.

The safetensors files are memory-mapped and parsed directly, so BF16 tensors
reach the TensorRT constants as their exact stored bits (a BF16 engine keeps
the checkpoint's native precision; an FP32 engine widens them exactly).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_STORED_DTYPES = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
}


class Checkpoint:
    """Tensor index over every safetensors file of one checkpoint directory."""

    def __init__(self, model_dir: str | Path) -> None:
        root = Path(model_dir)
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = sorted(path.name for path in root.glob("*.safetensors"))
        if not files:
            raise ValueError(f"no safetensors files in {root}")
        self._entries: dict[str, tuple[Path, str, tuple[int, ...], int, int]] = {}
        for name in files:
            self._index_file(root / name)

    def _index_file(self, path: Path) -> None:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        for tensor, record in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in self._entries:
                raise ValueError(f"tensor {tensor} is stored twice")
            begin, end = record["data_offsets"]
            self._entries[tensor] = (
                path,
                record["dtype"],
                tuple(int(dim) for dim in record["shape"]),
                data_start + int(begin),
                int(end) - int(begin),
            )

    def names(self) -> frozenset[str]:
        return frozenset(self._entries)

    def shape(self, name: str) -> tuple[int, ...]:
        return self._entry(name)[2]

    def _entry(self, name: str) -> tuple[Path, str, tuple[int, ...], int, int]:
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"checkpoint tensor {name} is missing") from None

    def stored(self, name: str) -> tuple[str, np.ndarray]:
        """Return the stored dtype name and a read-only view of the raw values."""
        path, dtype, shape, offset, length = self._entry(name)
        if dtype not in _STORED_DTYPES:
            raise ValueError(f"tensor {name} has unsupported dtype {dtype}")
        element = _STORED_DTYPES[dtype]
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if count * element.itemsize != length:
            raise ValueError(f"tensor {name} byte length does not match its shape")
        values = np.memmap(path, dtype=element, mode="r", offset=offset, shape=(count,))
        return dtype, values.reshape(shape)

    def float32(self, name: str) -> np.ndarray:
        """Return the tensor widened exactly to float32."""
        dtype, values = self.stored(name)
        if dtype == "BF16":
            widened = values.astype(np.uint32) << np.uint32(16)
            return np.ascontiguousarray(widened.view(np.float32))
        return np.ascontiguousarray(values, dtype=np.float32)

    def bfloat16_bits(self, name: str) -> np.ndarray:
        """Return the stored bfloat16 bit patterns of a BF16 tensor."""
        dtype, values = self.stored(name)
        if dtype != "BF16":
            raise ValueError(f"tensor {name} is stored as {dtype}, not BF16")
        return np.ascontiguousarray(values, dtype=np.uint16)
