# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time import of Comfy's public Qwen3-VL NVFP4 AWQ checkpoint.

The released checkpoint explicitly requests full-precision matrix products.
Decode its packed weights to BF16 for the native TensorRT language engine;
this is checkpoint compatibility, not FP4 activation or matrix computation.
The same file contains the unquantized BF16 vision tower. AWQ input scales
remain separate so the language graph preserves their BF16 rounding point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
from typing import Iterable

import ml_dtypes
import numpy as np

from .quantized_checkpoint import (
    CHECKPOINT_REVISION,
    MODEL_ID,
    QuantizedSourceFileIdentity,
    _read_header,
    _read_marker,
    _source_file_identity,
)

CHECKPOINT_FILENAME = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
CHECKPOINT_BYTES = 15_687_142_551
_NVFP4_MARKER = {"format": "nvfp4", "full_precision_matrix_mult": True}
_EMBEDDING = "model.embed_tokens"
_DTYPES = {
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F32": np.dtype("<f4"),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
}


@dataclass(frozen=True)
class QuantizedTextCheckpointIdentity:
    model_id: str = MODEL_ID
    revision: str = CHECKPOINT_REVISION
    filename: str = CHECKPOINT_FILENAME
    size_bytes: int = CHECKPOINT_BYTES
    tensor_count: int = 2_054
    quantized_weight_count: int = 350
    source_file_identity: QuantizedSourceFileIdentity | None = None

    def bundle_metadata(self) -> dict[str, object]:
        """Public source and compute identity, without local paths."""
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "revision": self.revision,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "tensor_count": self.tensor_count,
            "quantized_weight_count": self.quantized_weight_count,
            "quantization": "nvfp4_awq",
            "embedding_quantization": "int8_tensorwise",
            "full_precision_matrix_mult": True,
            "engine_weights_dtype": "bfloat16",
            "matmul_dtype": "bfloat16",
            "runtime_framework": None,
        }


QUANTIZED_TEXT_CHECKPOINT_IDENTITY = QuantizedTextCheckpointIdentity()


def _physical_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    specs = {}

    def tensor(name, shape, dtype="BF16"):
        specs[name] = (dtype, tuple(shape))

    tensor(f"{_EMBEDDING}.weight", (151936, 5120), "I8")
    tensor(f"{_EMBEDDING}.weight_scale", (151936, 1), "F32")
    tensor(f"{_EMBEDDING}.comfy_quant", (29,), "U8")
    for index in range(50):
        prefix = f"model.layers.{index}"
        for name, width in (("input_layernorm", 5120), ("post_attention_layernorm", 5120),
                            ("self_attn.q_norm", 128), ("self_attn.k_norm", 128)):
            tensor(f"{prefix}.{name}.weight", (width,))
        for name, rows, cols in (
            ("self_attn.q_proj", 8192, 5120), ("self_attn.k_proj", 1024, 5120),
            ("self_attn.v_proj", 1024, 5120), ("self_attn.o_proj", 5120, 8192),
            ("mlp.gate_proj", 25600, 5120), ("mlp.up_proj", 25600, 5120),
            ("mlp.down_proj", 5120, 25600),
        ):
            base = f"{prefix}.{name}"
            tensor(f"{base}.weight", (rows, cols // 2), "U8")
            tensor(f"{base}.weight_scale", (rows, cols // 16), "F8_E4M3")
            tensor(f"{base}.weight_scale_2", (), "F32")
            tensor(f"{base}.comfy_quant", (55,), "U8")
            if name in ("self_attn.o_proj", "mlp.down_proj"):
                tensor(f"{base}.pre_quant_scale", (cols,))

    def linear(base, rows, cols):
        tensor(f"{base}.weight", (rows, cols))
        tensor(f"{base}.bias", (rows,))

    def norm(base, width):
        tensor(f"{base}.weight", (width,))
        tensor(f"{base}.bias", (width,))

    tensor("visual.patch_embed.proj.weight", (1152, 3, 2, 16, 16))
    tensor("visual.patch_embed.proj.bias", (1152,))
    tensor("visual.pos_embed.weight", (2304, 1152))
    for index in range(27):
        prefix = f"visual.blocks.{index}"
        norm(f"{prefix}.norm1", 1152)
        norm(f"{prefix}.norm2", 1152)
        linear(f"{prefix}.attn.qkv", 3456, 1152)
        linear(f"{prefix}.attn.proj", 1152, 1152)
        linear(f"{prefix}.mlp.linear_fc1", 4304, 1152)
        linear(f"{prefix}.mlp.linear_fc2", 1152, 4304)
    for prefix in ("visual.merger", *(f"visual.deepstack_merger_list.{i}" for i in range(3))):
        norm(f"{prefix}.norm", 1152 if prefix == "visual.merger" else 4608)
        linear(f"{prefix}.linear_fc1", 4608, 4608)
        linear(f"{prefix}.linear_fc2", 5120, 4608)
    return specs


_PHYSICAL_SPECS = _physical_specs()


def _authenticate_source(path: Path) -> None:
    identity = QUANTIZED_TEXT_CHECKPOINT_IDENTITY
    relative = Path(identity.filename)
    root = path.parents[len(relative.parts) - 1]
    if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(root / relative)):
        raise ValueError("MiniMax-H3 text checkpoint must retain its released text_encoders path")
    if root.parent.name == "snapshots":
        if root.name != identity.revision or root.parent.parent.name != "models--Comfy-Org--MiniMax-H3":
            raise ValueError("MiniMax-H3 text checkpoint is not the pinned Hugging Face snapshot")
        return
    metadata_path = root / ".cache/huggingface/download" / f"{relative.as_posix()}.metadata"
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("Download the MiniMax-H3 text checkpoint with hf download and pinned revision metadata") from error
    if not lines or lines[0] != identity.revision:
        raise ValueError("MiniMax-H3 text checkpoint does not match the pinned Hugging Face revision")


def _validate_inventory(header, data_offset: int, file_size: int):
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if set(tensors) != set(_PHYSICAL_SPECS):
        raise ValueError("MiniMax-H3 NVFP4 text checkpoint tensor inventory mismatch")
    regions = []
    for name, (dtype, shape) in _PHYSICAL_SPECS.items():
        entry = tensors[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid MiniMax-H3 text tensor header: {name}")
        if entry["dtype"] != dtype or entry["shape"] != list(shape):
            raise ValueError(f"MiniMax-H3 text tensor contract mismatch: {name}")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)):
            raise ValueError(f"Invalid MiniMax-H3 text tensor offsets: {name}")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * _DTYPES[dtype].itemsize:
            raise ValueError(f"Invalid MiniMax-H3 text tensor byte range: {name}")
        regions.append((start, end))
    cursor = 0
    for start, end in sorted(regions):
        if start != cursor:
            raise ValueError("MiniMax-H3 text tensor storage has gaps or overlaps")
        cursor = end
    if data_offset + cursor != file_size:
        raise ValueError("MiniMax-H3 text checkpoint payload size mismatch")
    return tensors


def validate_quantized_text_checkpoint(checkpoint_file: str | Path) -> QuantizedTextCheckpointIdentity:
    """Check the pinned file, header and all tiny markers; never hash its payload."""
    path = Path(checkpoint_file)
    identity = QUANTIZED_TEXT_CHECKPOINT_IDENTITY
    before = _source_file_identity(path)
    if path.name != Path(identity.filename).name or before.size_bytes != identity.size_bytes:
        raise ValueError("MiniMax-H3 NVFP4 text checkpoint filename or size mismatch")
    _authenticate_source(path)
    header, offset = _read_header(path)
    tensors = _validate_inventory(header, offset, before.size_bytes)
    for name, entry in tensors.items():
        if name.endswith(".comfy_quant"):
            expected = {"format": "int8_tensorwise"} if name == f"{_EMBEDDING}.comfy_quant" else _NVFP4_MARKER
            if _read_marker(path, offset, entry) != expected:
                raise ValueError(f"Unsupported MiniMax-H3 text comfy_quant marker: {name}")
    if _source_file_identity(path) != before:
        raise ValueError("MiniMax-H3 text checkpoint changed during validation")
    return replace(identity, source_file_identity=before)


def _unblock_scales(scales: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Invert Comfy's cuBLAS 128x4 block-scale storage on the build CPU.

    Format reference: Comfy-Org/comfy-kitchen, float_utils.py::from_blocked.
    """
    row_blocks, col_blocks = (rows + 127) // 128, (cols + 3) // 4
    if scales.shape != (row_blocks * 128, col_blocks * 4):
        raise ValueError("MiniMax-H3 NVFP4 block-scale shape mismatch")
    tiles = scales.reshape(row_blocks, col_blocks, 32, 4, 4)
    plain = tiles.transpose(0, 1, 3, 2, 4).reshape(row_blocks, col_blocks, 128, 4)
    plain = plain.transpose(0, 2, 1, 3).reshape(row_blocks * 128, col_blocks * 4)
    return plain[:rows, :cols]


def _dequantize_nvfp4(packed, scales, global_scale) -> np.ndarray:
    """Match Comfy CUDA decode: FP32 scale/product, then one BF16 rounding.

    Comfy stores the even element in the HIGH nibble. Its CUDA reference is
    backends/cuda/ops/quantize_nvfp4.cu::dequantize_nvfp4_kernel; the eager CPU
    backend has extra intermediate BF16 rounding and is not used as an oracle.
    """
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] % 8:
        raise ValueError("MiniMax-H3 NVFP4 weights require packed uint8 [out,in/2], in divisible by 16")
    rows, cols = packed.shape[0], packed.shape[1] * 2
    if np.asarray(global_scale).shape != () or not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("MiniMax-H3 NVFP4 global scale must be a positive finite scalar")
    plain_scales = _unblock_scales(scales, rows, cols // 16)
    lut = np.asarray((0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6), np.float32)
    output = np.empty((rows, cols), dtype=ml_dtypes.bfloat16)
    for start in range(0, rows, 128):
        end = min(rows, start + 128)
        scale = plain_scales[start:end].astype(np.float32) * np.float32(global_scale)
        if not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("MiniMax-H3 NVFP4 block scales must be finite and non-negative")
        values = np.empty((end - start, cols), np.float32)
        values[:, 0::2] = lut[packed[start:end] >> 4]
        values[:, 1::2] = lut[packed[start:end] & 15]
        values.reshape(end - start, cols // 16, 16)[:] *= scale[..., None]
        output[start:end] = values
    return output


def _physical_name(logical_name: str) -> str:
    for logical, physical in (("model.language_model.", "model."), ("model.visual.", "visual.")):
        if logical_name.startswith(logical):
            name = physical + logical_name.removeprefix(logical)
            if name in _PHYSICAL_SPECS and name.endswith((".weight", ".bias")):
                return name
    raise ValueError(f"Unsupported MiniMax-H3 quantized text logical tensor: {logical_name}")


def load_selected_quantized_text_weights(
    checkpoint_file: str | Path, logical_names: Iterable[str]
) -> dict[str, np.ndarray]:
    """Load requested language/vision tensors and their optional AWQ input scales.

    Only requested payloads are touched. Memory-mapped BF16 vision/norm tensors
    retain their file owner; decoded weights use bounded per-row temporaries.
    """
    path = Path(checkpoint_file)
    identity = validate_quantized_text_checkpoint(path)
    requested = tuple(logical_names)
    if len(requested) != len(set(requested)):
        raise ValueError("MiniMax-H3 text weight request contains duplicates")
    names = {logical: _physical_name(logical) for logical in requested}
    header, offset = _read_header(path)

    def read(name):
        entry = header[name]
        return np.memmap(path, mode="r", dtype=_DTYPES[entry["dtype"]],
                         offset=offset + entry["data_offsets"][0], shape=tuple(entry["shape"]))

    result = {}
    for logical, physical in names.items():
        value = read(physical)
        base = physical.removesuffix(".weight")
        if f"{base}.weight_scale_2" in header:
            value = _dequantize_nvfp4(value, read(f"{base}.weight_scale"), read(f"{base}.weight_scale_2"))
            prescale = f"{base}.pre_quant_scale"
            if prescale in header:
                result[f"{logical.removesuffix('.weight')}.pre_quant_scale"] = read(prescale)
        elif physical == f"{_EMBEDDING}.weight":
            scale = read(f"{base}.weight_scale")
            decoded = np.empty(value.shape, dtype=ml_dtypes.bfloat16)
            for start in range(0, value.shape[0], 128):
                chunk_scale = scale[start:start + 128]
                if not np.isfinite(chunk_scale).all() or (chunk_scale < 0).any():
                    raise ValueError("MiniMax-H3 embedding scales must be finite and non-negative")
                decoded[start:start + 128] = value[start:start + 128].astype(np.float32) * chunk_scale
            value = decoded
        result[logical] = value
    if _source_file_identity(path) != identity.source_file_identity:
        raise ValueError("MiniMax-H3 text checkpoint changed during loading")
    return result
