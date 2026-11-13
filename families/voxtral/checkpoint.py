# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral checkpoint reader for MLX affine-quantized safetensors exports.

The supported export stores every quantized linear as three tensors:
``<module>.weight`` (uint32 words holding a little-endian bit stream of
``bits``-wide codes per output row), ``<module>.scales`` and ``<module>.biases``
(one value per output row and ``group_size`` input columns). The dense weight is

    w[o, g * group_size + j] = scales[o, g] * q[o, g * group_size + j] + biases[o, g]

and is reconstructed in float32 here, so every engine precision rounds the
checkpoint's values exactly once. Convolution kernels are stored as
``[out, kernel, in]`` and are returned in the ``[out, in, kernel]`` layout.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
}
_ROW_BLOCK = 4096


def bfloat16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Widen raw bfloat16 bit patterns to float32 exactly."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def unpack_codes(packed: np.ndarray, bits: int) -> np.ndarray:
    """Unpack ``[rows, words]`` uint32 words into ``[rows, words * 32 // bits]`` codes.

    Each row is one little-endian bit stream: code ``i`` occupies bits
    ``[i * bits, (i + 1) * bits)``. Eight codes span exactly ``bits`` bytes for
    every width, so the stream is read eight codes at a time.
    """
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported MLX code width: {bits} bits")
    if packed.dtype != np.dtype("<u4") or packed.ndim != 2:
        raise ValueError("packed MLX weights must be a 2-D little-endian uint32 array")
    rows, words = packed.shape
    if (words * 32) % (8 * bits):
        raise ValueError("packed MLX row does not hold a whole number of eight-code blocks")
    stream = np.ascontiguousarray(packed).view(np.uint8).reshape(rows, -1, bits)
    block = np.zeros(stream.shape[:2], dtype=np.uint64)
    for index in range(bits):
        block |= stream[:, :, index].astype(np.uint64) << np.uint64(8 * index)
    shifts = np.arange(8, dtype=np.uint64) * np.uint64(bits)
    mask = np.uint64((1 << bits) - 1)
    codes = (block[:, :, None] >> shifts) & mask
    return codes.reshape(rows, -1).astype(np.uint8)


def dequantize(
    packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int, group_size: int
) -> np.ndarray:
    """Return the float32 ``[rows, in_features]`` weight of one MLX affine tensor."""
    rows, words = packed.shape
    in_features = words * 32 // bits
    if group_size <= 0 or in_features % group_size:
        raise ValueError(f"group size {group_size} does not divide {in_features} input columns")
    groups = in_features // group_size
    if scales.shape != (rows, groups) or biases.shape != (rows, groups):
        raise ValueError(
            f"scales/biases {scales.shape}/{biases.shape} do not match ({rows}, {groups})"
        )
    output = np.empty((rows, in_features), dtype=np.float32)
    for start in range(0, rows, _ROW_BLOCK):
        stop = min(rows, start + _ROW_BLOCK)
        codes = unpack_codes(packed[start:stop], bits).astype(np.float32)
        codes = codes.reshape(stop - start, groups, group_size)
        scale = scales[start:stop, :, None].astype(np.float32)
        bias = biases[start:stop, :, None].astype(np.float32)
        output[start:stop] = (codes * scale + bias).reshape(stop - start, in_features)
    return output


def float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (nearest, ties to even) and return the raw bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def engine_array(values: np.ndarray, precision: str) -> np.ndarray:
    """Values in the storage format of ``precision`` (bfloat16 as raw uint16 bits)."""
    if precision == "fp32":
        return np.ascontiguousarray(values, dtype=np.float32)
    if precision == "fp16":
        return np.ascontiguousarray(values, dtype=np.float16)
    if precision == "bf16":
        return float32_to_bfloat16_bits(values)
    raise ValueError(f"unsupported precision {precision!r}")


class SafetensorsFile:
    """Memory-mapped view of one safetensors file; tensors are read on demand."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        self.metadata = header.pop("__metadata__", None) or {}
        self.entries = header
        self._data_start = 8 + length
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def raw(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        dtype = _DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{self.path.name}: unsupported dtype {entry['dtype']} for {name}")
        begin, end = entry["data_offsets"]
        shape = tuple(int(value) for value in entry["shape"])
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if end - begin != count * dtype.itemsize:
            raise ValueError(f"{self.path.name}: byte range of {name} does not match its shape")
        data = self._map[self._data_start + begin : self._data_start + end]
        return data.view(dtype).reshape(shape)

    def dtype(self, name: str) -> str:
        return self.entries[name]["dtype"]


class MlxCheckpoint:
    """Dense float32 tensors of an MLX-exported Voxtral checkpoint."""

    def __init__(self, model_dir: Path, quantization: dict):
        model_dir = Path(model_dir)
        index = model_dir / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            names = sorted(set(weight_map.values()))
        else:
            names = ["model.safetensors"]
        self._files = [SafetensorsFile(model_dir / name) for name in names]
        formats = {file.metadata.get("format") for file in self._files}
        if formats != {"mlx"}:
            raise ValueError(
                "voxtral reads MLX affine-quantized exports only; the safetensors metadata "
                f"declares format {sorted(str(value) for value in formats)}"
            )
        self._owner = {}
        for file in self._files:
            for name in file.entries:
                if name in self._owner:
                    raise ValueError(f"tensor {name} appears in more than one safetensors file")
                self._owner[name] = file
        self._quantization = quantization
        self.consumed: set[str] = set()

    def names(self) -> list[str]:
        return sorted(self._owner)

    def has(self, name: str) -> bool:
        return name in self._owner

    def _plain(self, name: str) -> np.ndarray:
        file = self._owner.get(name)
        if file is None:
            raise KeyError(f"checkpoint tensor {name} is missing")
        self.consumed.add(name)
        value = file.raw(name)
        if file.dtype(name) == "BF16":
            return bfloat16_bits_to_float32(value)
        if file.dtype(name) == "U32":
            raise ValueError(f"{name} is a packed tensor without a quantization entry")
        return value.astype(np.float32)

    def quantization_of(self, module: str) -> dict | None:
        spec = self._quantization.get(module, None)
        if spec is None or spec is False:
            return None
        return spec

    def linear(self, module: str, out_features: int, in_features: int) -> np.ndarray:
        """Dense ``[out, in]`` float32 weight of a linear module."""
        spec = self.quantization_of(module)
        name = f"{module}.weight"
        if spec is None:
            if self.has(f"{module}.scales"):
                raise ValueError(f"{module} carries scales but no quantization entry")
            weight = self._plain(name)
        else:
            file = self._owner.get(name)
            if file is None or file.dtype(name) != "U32":
                raise ValueError(f"{module} is declared quantized but its weight is not packed")
            self.consumed.add(name)
            weight = dequantize(
                file.raw(name),
                self._plain(f"{module}.scales"),
                self._plain(f"{module}.biases"),
                int(spec["bits"]),
                int(spec["group_size"]),
            )
        if weight.shape != (out_features, in_features):
            raise ValueError(
                f"{module} weight has shape {weight.shape}, expected ({out_features}, {in_features})"
            )
        return weight

    def packed_equal(self, first: str, second: str) -> bool:
        """Whether two quantized modules store identical codes, scales and biases."""
        for suffix in ("weight", "scales", "biases"):
            a, b = f"{first}.{suffix}", f"{second}.{suffix}"
            if a not in self._owner or b not in self._owner:
                return False
            left, right = self._owner[a], self._owner[b]
            if left.dtype(a) != right.dtype(b) or not np.array_equal(left.raw(a), right.raw(b)):
                return False
            self.consumed.update((a, b))
        return True

    def unconsumed(self) -> list[str]:
        return sorted(set(self._owner) - self.consumed)

    def vector(self, name: str, size: int) -> np.ndarray:
        value = self._plain(name)
        if value.shape != (size,):
            raise ValueError(f"{name} has shape {value.shape}, expected ({size},)")
        return value

    def matrix(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = self._plain(name)
        if value.shape != tuple(shape):
            raise ValueError(f"{name} has shape {value.shape}, expected {tuple(shape)}")
        return value

    def conv1d(self, module: str, out_channels: int, in_channels: int, kernel: int) -> np.ndarray:
        """Convolution kernel as ``[out, in, kernel]`` (the export stores ``[out, kernel, in]``)."""
        if self.quantization_of(module) is not None:
            raise ValueError(f"quantized convolution {module} is not supported")
        value = self._plain(f"{module}.weight")
        if value.shape != (out_channels, kernel, in_channels):
            raise ValueError(
                f"{module}.weight has shape {value.shape}, expected "
                f"({out_channels}, {kernel}, {in_channels}) in the MLX layout"
            )
        return np.ascontiguousarray(value.transpose(0, 2, 1))
