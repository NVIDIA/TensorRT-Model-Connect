# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral checkpoint reader for Transformers safetensors exports.

The supported checkpoint stores floating-point tensors (BF16, F16 or F32) in the
PyTorch layout under the Transformers names (``model.safetensors`` or the shards
named by ``model.safetensors.index.json``). Tensors are widened to float32
exactly here, so every engine precision rounds the checkpoint's values once.
Before anything is built, the header must hold exactly the tensors the
configuration implies (``config.expected_tensors``), with their shapes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .config import load_json

_DTYPES = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
}
_INDEX = "model.safetensors.index.json"
_SINGLE = "model.safetensors"
_FP16_MAX = float(np.finfo(np.float16).max)
# The safetensors reference implementation refuses headers above 100 MB.
_MAX_HEADER_BYTES = 100_000_000


def bfloat16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Widen raw bfloat16 bit patterns to float32 exactly."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (nearest, ties to even) and return the raw bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def engine_array(values: np.ndarray, precision: str) -> np.ndarray:
    """Finite float32 values in the storage format of ``precision`` (bfloat16 as raw uint16 bits).

    A value that is not finite, or that becomes infinite when rounded to the engine precision, is refused.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("a weight is not finite")
    if precision == "fp32":
        return values
    if precision == "fp16":
        if values.size and float(np.abs(values).max()) > _FP16_MAX:
            raise ValueError("a weight exceeds the float16 range; build with precision bf16 or fp32")
        return values.astype(np.float16)
    if precision == "bf16":
        bits = float32_to_bfloat16_bits(values)
        if np.any((bits & np.uint16(0x7F80)) == np.uint16(0x7F80)):
            raise ValueError("a weight exceeds the bfloat16 range; build with precision fp32")
        return bits
    raise ValueError(f"unsupported precision {precision!r}")


def _check_entry(where: str, name: str, entry, data_size: int) -> None:
    """A header entry: a dtype name, a shape of JSON integers and a byte range inside the data."""
    if not isinstance(entry, dict) or not {"dtype", "shape", "data_offsets"} <= set(entry):
        raise ValueError(f"{where}: malformed header entry for {name}")
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if not isinstance(dtype, str):
        raise ValueError(f"{where}: dtype of {name} must be a string")
    if not isinstance(shape, list) or not all(type(value) is int and value >= 0 for value in shape):
        raise ValueError(f"{where}: shape of {name} must be a list of non-negative integers")
    if (not isinstance(offsets, list) or len(offsets) != 2 or not all(type(value) is int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size):
        raise ValueError(f"{where}: data offsets of {name} lie outside the data")
    known = _DTYPES.get(dtype)
    if known is not None:
        count = 1
        for value in shape:
            count *= value
        if offsets[1] - offsets[0] != count * known.itemsize:
            raise ValueError(f"{where}: byte range of {name} does not match its shape")


class SafetensorsFile:
    """Memory-mapped view of one safetensors file; tensors are read on demand."""

    def __init__(self, path: Path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{self.path.name}: not a safetensors file")
            (length,) = struct.unpack("<Q", prefix)
            if length > min(size - 8, _MAX_HEADER_BYTES):
                raise ValueError(f"{self.path.name}: header length {length} exceeds the file or 100 MB")
            header = load_json(handle.read(length), f"{self.path.name} header")
        if not isinstance(header, dict):
            raise ValueError(f"{self.path.name}: the header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or not all(isinstance(value, str) for value in metadata.values()):
            raise ValueError(f"{self.path.name}: __metadata__ must map names to strings")
        self.metadata = metadata
        self._data_start = 8 + length
        self._data_size = size - self._data_start
        for name, entry in header.items():
            _check_entry(self.path.name, name, entry, self._data_size)
        # The tensors must tile the data region exactly: no two share bytes, no byte is left over.
        position = 0
        for begin, end, name in sorted((entry["data_offsets"][0], entry["data_offsets"][1], name)
                                       for name, entry in header.items()):
            if begin != position:
                raise ValueError(f"{self.path.name}: the data of {name} overlaps another tensor or leaves a gap")
            position = end
        if position != self._data_size:
            raise ValueError(f"{self.path.name}: {self._data_size - position} data bytes belong to no tensor")
        self.entries = header
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def dtype(self, name: str) -> str:
        return self.entries[name]["dtype"]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.entries[name]["shape"])

    def raw(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        dtype = _DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{self.path.name}: unsupported dtype {entry['dtype']} for {name}")
        begin, end = entry["data_offsets"]
        data = self._map[self._data_start + begin : self._data_start + end]
        return data.view(dtype).reshape(tuple(entry["shape"]))


def _shard_names(model_dir: Path) -> list[str]:
    index = model_dir / _INDEX
    if not index.is_file():
        if not (model_dir / _SINGLE).is_file():
            raise FileNotFoundError(f"voxtral checkpoint has neither {_INDEX} nor {_SINGLE}")
        return [_SINGLE]
    names = set(_weight_map(index).values())
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"{_INDEX} names an invalid shard {name!r}")
    return sorted(names)


def _weight_map(index: Path) -> dict[str, str]:
    content = load_json(index.read_text(encoding="utf-8"), _INDEX)
    weight_map = content.get("weight_map") if isinstance(content, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{_INDEX} has no weight_map")
    if not all(isinstance(value, str) for value in weight_map.values()):
        raise ValueError(f"{_INDEX} maps a tensor to something other than a file name")
    return weight_map


class Checkpoint:
    """Float32 tensors of a Transformers Voxtral checkpoint."""

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self._files = [SafetensorsFile(model_dir / name) for name in _shard_names(model_dir)]
        for file in self._files:
            layout = file.metadata.get("format", "pt")
            if layout != "pt":
                raise ValueError(f"voxtral reads PyTorch-layout safetensors; {file.path.name} declares "
                                 f"format {layout!r}")
        self._owner: dict[str, SafetensorsFile] = {}
        for file in self._files:
            for name in file.entries:
                if name in self._owner:
                    raise ValueError(f"tensor {name} appears in more than one safetensors file")
                self._owner[name] = file
        index = model_dir / _INDEX
        if index.is_file():
            weight_map = _weight_map(index)
            misplaced = sorted(name for name, file in self._owner.items()
                               if weight_map.get(name) != file.path.name)
            missing = sorted(set(weight_map) - set(self._owner))
            if misplaced or missing:
                raise ValueError(f"{_INDEX} does not match the shards: {(misplaced + missing)[:8]}")
        self.consumed: set[str] = set()

    def names(self) -> list[str]:
        return sorted(self._owner)

    def validate(self, expected: dict[str, tuple[int, ...]]) -> None:
        """The checkpoint must hold exactly ``expected`` (name to shape), in a floating-point dtype."""
        missing = sorted(set(expected) - set(self._owner))
        extra = sorted(set(self._owner) - set(expected))
        if missing or extra:
            raise ValueError(f"voxtral checkpoint tensors differ from the configuration: missing {missing[:8]}, "
                             f"unexpected {extra[:8]}")
        for name, shape in expected.items():
            file = self._owner[name]
            if file.dtype(name) not in _DTYPES:
                raise ValueError(f"{name} has unsupported dtype {file.dtype(name)}")
            if file.shape(name) != tuple(shape):
                raise ValueError(f"{name} has shape {file.shape(name)}, expected {tuple(shape)}")

    def tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        """One finite tensor as float32, checked against ``shape``."""
        file = self._owner.get(name)
        if file is None:
            raise KeyError(f"checkpoint tensor {name} is missing")
        if file.shape(name) != tuple(shape):
            raise ValueError(f"{name} has shape {file.shape(name)}, expected {tuple(shape)}")
        value = file.raw(name)
        value = bfloat16_bits_to_float32(value) if file.dtype(name) == "BF16" else value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} holds non-finite values")
        self.consumed.add(name)
        return value

    def linear(self, module: str, out_features: int, in_features: int) -> np.ndarray:
        """Dense ``[out, in]`` float32 weight of a linear module."""
        return self.tensor(f"{module}.weight", (out_features, in_features))

    def vector(self, name: str, size: int) -> np.ndarray:
        return self.tensor(name, (size,))

    def matrix(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        return self.tensor(name, tuple(shape))

    def conv1d(self, module: str, out_channels: int, in_channels: int, kernel: int) -> np.ndarray:
        """Convolution kernel in the PyTorch ``[out, in, kernel]`` layout."""
        return self.tensor(f"{module}.weight", (out_channels, in_channels, kernel))

    def unconsumed(self) -> list[str]:
        return sorted(set(self._owner) - self.consumed)
