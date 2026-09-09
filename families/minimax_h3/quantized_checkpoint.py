# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict loader for the public full MiniMax-H3 INT8 ConvRot transformer.

The Comfy-Org single-file checkpoint uses the original MiniMax module names,
contiguous Q/K/V thirds, and ``[gate; value]`` SwiGLU rows.  The native
TensorRT builders consume the frozen Diffusers names, the same contiguous Q/K/V
layout, and ``[value; gate]`` rows.  This module performs the lossless SwiGLU
row permutation while preserving the checkpoint's INT8 data and per-output-row
FP32 scales.

Large tensor payloads are selected through :mod:`safetensors`; validation reads
only the JSON header and the 250 tiny ``comfy_quant`` markers. The released
file size is checked without rereading the 34 GB payload before every build.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Iterable

import numpy as np

from .checkpoint import numpy_state


MODEL_ID = "Comfy-Org/MiniMax-H3"
CHECKPOINT_REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"
CHECKPOINT_FILENAME = "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"
CHECKPOINT_BYTES = 34_038_892_334

_MAX_HEADER_BYTES = 8 << 20
_NUM_HEADS = 56
_HEAD_DIM = 128
_HIDDEN_SIZE = 5_376
_INNER_DIM = _NUM_HEADS * _HEAD_DIM
_FFN_DIM = 14_336
_TIME_DIM = 2_688
_QUANT_FORMAT = "int8_tensorwise"
_HF_CACHE_REPOSITORY = "models--Comfy-Org--MiniMax-H3"
_HF_LOCAL_DOWNLOAD_METADATA = Path(".cache/huggingface/download")


@dataclass(frozen=True)
class ConvRotInt8Weight:
    """One Comfy tensorwise-INT8 weight in Diffusers logical row order.

    ``packed_parent`` and ``row_slice`` let the fused-QKV graph path recover a
    single packed parent matrix when the loader returned three
    logical ``to_q``/``to_k``/``to_v`` entries.  No concatenation of those
    child views is necessary.
    """

    qweight: np.ndarray
    scale: np.ndarray
    group_size: int
    packed_parent: ConvRotInt8Weight | None = None
    row_slice: tuple[int, int] | None = None
    is_full_fused_qkv: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.qweight, np.ndarray) or self.qweight.dtype != np.int8:
            raise TypeError("MiniMax-H3 ConvRot qweight must be a NumPy int8 array")
        if self.qweight.ndim != 2 or not self.qweight.flags.c_contiguous:
            raise ValueError("MiniMax-H3 ConvRot qweight must be contiguous [out,in]")
        if not isinstance(self.scale, np.ndarray) or self.scale.dtype != np.float32:
            raise TypeError("MiniMax-H3 ConvRot scale must be a NumPy float32 array")
        if self.scale.shape not in ((self.qweight.shape[0],), (self.qweight.shape[0], 1)):
            raise ValueError("MiniMax-H3 ConvRot scale must contain one value per output row")
        if not self.scale.flags.c_contiguous:
            raise ValueError("MiniMax-H3 ConvRot scale must be contiguous")
        _validate_group_size(self.group_size, self.qweight.shape[1])
        if self.is_full_fused_qkv:
            if self.packed_parent is not None or self.row_slice is not None:
                raise ValueError("A full fused-QKV ConvRot weight cannot itself be a child view")
        elif self.packed_parent is not None:
            if not self.packed_parent.is_full_fused_qkv or self.row_slice is None:
                raise ValueError("A fused-QKV child requires a full parent and row_slice")
            start, end = self.row_slice
            if not 0 <= start < end <= self.packed_parent.qweight.shape[0]:
                raise ValueError("MiniMax-H3 fused-QKV child row_slice is out of bounds")
            if end - start != self.qweight.shape[0]:
                raise ValueError("MiniMax-H3 fused-QKV child row_slice does not match qweight")
        elif self.row_slice is not None:
            raise ValueError("MiniMax-H3 ConvRot row_slice requires packed_parent")


@dataclass(frozen=True)
class QuantizedSourceFileIdentity:
    """Path-free local identity used only to make staged resume fail closed."""

    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    def receipt_metadata(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class QuantizedCheckpointIdentity:
    """Path-free identity of the pinned public Comfy-Org checkpoint."""

    model_id: str
    revision: str
    filename: str
    size_bytes: int
    tensor_count: int
    quantized_weight_count: int
    source_file_identity: QuantizedSourceFileIdentity | None = None

    def bundle_metadata(self) -> dict[str, object]:
        """Return stable public provenance without a workstation path."""

        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "revision": self.revision,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "tensor_count": self.tensor_count,
            "quantized_weight_count": self.quantized_weight_count,
            "quantization": "int8_tensorwise_convrot",
            "runtime_framework": None,
        }


QUANTIZED_CHECKPOINT_IDENTITY = QuantizedCheckpointIdentity(
    model_id=MODEL_ID,
    revision=CHECKPOINT_REVISION,
    filename=CHECKPOINT_FILENAME,
    size_bytes=CHECKPOINT_BYTES,
    tensor_count=1_035,
    quantized_weight_count=250,
)


_EXPECTED_CONFIG: dict[str, object] = {
    "transformer": {
        "hidden_size": 5_376,
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "ffn_hidden_size": 14_336,
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 5_120,
        "timestep_input_dim": 256,
        "time_embed_hidden_size": 5_376,
        "time_embed_dim": 2_688,
        "adaln_out_features": 96_768,
        "final_adaln_out_features": 10_752,
        "rope_inv_freq_len": 16,
        "norm_eps": 1.0e-5,
        "qk_norm_eps": 1.0e-5,
        "final_norm_eps": 1.0e-5,
        "image_model": "minimax_h3",
    }
}


@dataclass(frozen=True)
class _TensorSpec:
    dtype: str
    shape: tuple[int, ...]


def _add_quantized(
    specs: dict[str, _TensorSpec],
    groups: dict[str, int],
    base: str,
    shape: tuple[int, int],
    group_size: int,
) -> None:
    specs[f"{base}.weight"] = _TensorSpec("I8", shape)
    specs[f"{base}.weight_scale"] = _TensorSpec("F32", (shape[0], 1))
    marker_bytes = len(
        json.dumps(
            {
                "format": _QUANT_FORMAT,
                "convrot": True,
                "convrot_groupsize": group_size,
            }
        ).encode("utf-8")
    )
    specs[f"{base}.comfy_quant"] = _TensorSpec("U8", (marker_bytes,))
    groups[base] = group_size


def _physical_contract() -> tuple[dict[str, _TensorSpec], dict[str, int]]:
    specs: dict[str, _TensorSpec] = {
        "video_patch_proj.weight": _TensorSpec("F32", (_HIDDEN_SIZE, 96)),
        "video_patch_proj.bias": _TensorSpec("F32", (_HIDDEN_SIZE,)),
        "audio_patch_proj.weight": _TensorSpec("F32", (_HIDDEN_SIZE, 32)),
        "audio_patch_proj.bias": _TensorSpec("F32", (_HIDDEN_SIZE,)),
        "condition_proj.weight": _TensorSpec("BF16", (_HIDDEN_SIZE, 5_120)),
        "condition_proj.bias": _TensorSpec("BF16", (_HIDDEN_SIZE,)),
        "time_embedder.proj_in.weight": _TensorSpec("F32", (_HIDDEN_SIZE, 256)),
        "time_embedder.proj_in.bias": _TensorSpec("F32", (_HIDDEN_SIZE,)),
        "time_embedder.proj_out.weight": _TensorSpec("F32", (_TIME_DIM, _HIDDEN_SIZE)),
        "time_embedder.proj_out.bias": _TensorSpec("F32", (_TIME_DIM,)),
        "token_refiner.final_norm.weight": _TensorSpec("BF16", (_HIDDEN_SIZE,)),
        "final_layer.norm.weight": _TensorSpec("BF16", (_HIDDEN_SIZE,)),
        "final_layer.adaln_proj.linear.weight": _TensorSpec(
            "BF16", (2 * _HIDDEN_SIZE, _TIME_DIM)
        ),
        "final_layer.adaln_proj.linear.bias": _TensorSpec("BF16", (2 * _HIDDEN_SIZE,)),
        "final_layer.video_out.weight": _TensorSpec("F32", (96, _HIDDEN_SIZE)),
        "final_layer.video_out.bias": _TensorSpec("F32", (96,)),
        "final_layer.audio_out.weight": _TensorSpec("F32", (32, _HIDDEN_SIZE)),
        "final_layer.audio_out.bias": _TensorSpec("F32", (32,)),
        "rope.inv_freq": _TensorSpec("F32", (16,)),
    }
    groups: dict[str, int] = {}
    for index in range(50):
        prefix = f"blocks.{index}"
        specs[f"{prefix}.norm1.weight"] = _TensorSpec("BF16", (_HIDDEN_SIZE,))
        specs[f"{prefix}.norm2.weight"] = _TensorSpec("BF16", (_HIDDEN_SIZE,))
        specs[f"{prefix}.attn.q_norm.weight"] = _TensorSpec("BF16", (_HEAD_DIM,))
        specs[f"{prefix}.attn.k_norm.weight"] = _TensorSpec("BF16", (_HEAD_DIM,))
        specs[f"{prefix}.adaln_proj.linear.bias"] = _TensorSpec(
            "BF16", (18 * _HIDDEN_SIZE,)
        )
        _add_quantized(
            specs,
            groups,
            f"{prefix}.adaln_proj.linear",
            (18 * _HIDDEN_SIZE, _TIME_DIM),
            64,
        )
        _add_quantized(
            specs,
            groups,
            f"{prefix}.attn.qkv_proj",
            (3 * _INNER_DIM, _HIDDEN_SIZE),
            256,
        )
        _add_quantized(
            specs,
            groups,
            f"{prefix}.attn.out_proj",
            (_HIDDEN_SIZE, _INNER_DIM),
            256,
        )
        _add_quantized(
            specs,
            groups,
            f"{prefix}.mlp.fc1",
            (2 * _FFN_DIM, _HIDDEN_SIZE),
            256,
        )
        _add_quantized(
            specs,
            groups,
            f"{prefix}.mlp.fc2",
            (_HIDDEN_SIZE, _FFN_DIM),
            256,
        )
    for index in range(2):
        prefix = f"token_refiner.blocks.{index}"
        specs.update(
            {
                f"{prefix}.norm1.weight": _TensorSpec("BF16", (_HIDDEN_SIZE,)),
                f"{prefix}.norm2.weight": _TensorSpec("BF16", (_HIDDEN_SIZE,)),
                f"{prefix}.attn.qkv_proj.weight": _TensorSpec(
                    "BF16", (3 * _INNER_DIM, _HIDDEN_SIZE)
                ),
                f"{prefix}.attn.q_norm.weight": _TensorSpec("BF16", (_HEAD_DIM,)),
                f"{prefix}.attn.k_norm.weight": _TensorSpec("BF16", (_HEAD_DIM,)),
                f"{prefix}.attn.out_proj.weight": _TensorSpec(
                    "BF16", (_HIDDEN_SIZE, _INNER_DIM)
                ),
                f"{prefix}.mlp.fc1.weight": _TensorSpec(
                    "BF16", (2 * _FFN_DIM, _HIDDEN_SIZE)
                ),
                f"{prefix}.mlp.fc2.weight": _TensorSpec("BF16", (_HIDDEN_SIZE, _FFN_DIM)),
            }
        )
    if len(specs) != 1_035 or len(groups) != 250:
        raise RuntimeError("MiniMax-H3 quantized checkpoint contract is incomplete")
    return specs, groups


_PHYSICAL_SPECS, _QUANT_GROUPS = _physical_contract()


def _logical_contract() -> dict[str, tuple[str, str]]:
    direct = {
        "proj_in.weight": "video_patch_proj.weight",
        "proj_in.bias": "video_patch_proj.bias",
        "audio_proj_in.weight": "audio_patch_proj.weight",
        "audio_proj_in.bias": "audio_patch_proj.bias",
        "context_embedder.weight": "condition_proj.weight",
        "context_embedder.bias": "condition_proj.bias",
        "time_embedder.linear_1.weight": "time_embedder.proj_in.weight",
        "time_embedder.linear_1.bias": "time_embedder.proj_in.bias",
        "time_embedder.linear_2.weight": "time_embedder.proj_out.weight",
        "time_embedder.linear_2.bias": "time_embedder.proj_out.bias",
        "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
        "norm_out.norm.weight": "final_layer.norm.weight",
        "norm_out.linear.weight": "final_layer.adaln_proj.linear.weight",
        "norm_out.linear.bias": "final_layer.adaln_proj.linear.bias",
        "proj_out.weight": "final_layer.video_out.weight",
        "proj_out.bias": "final_layer.video_out.bias",
        "audio_proj_out.weight": "final_layer.audio_out.weight",
        "audio_proj_out.bias": "final_layer.audio_out.bias",
    }
    result = {logical: (physical, "direct") for logical, physical in direct.items()}
    for physical_root, logical_root, count, has_adaln in (
        ("blocks", "transformer_blocks", 50, True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", 2, False),
    ):
        for index in range(count):
            physical = f"{physical_root}.{index}"
            logical = f"{logical_root}.{index}"
            result[f"{logical}.norm1.weight"] = (f"{physical}.norm1.weight", "direct")
            result[f"{logical}.norm2.weight"] = (f"{physical}.norm2.weight", "direct")
            for qkv_index, name in enumerate(("q", "k", "v")):
                result[f"{logical}.attn.to_{name}.weight"] = (
                    f"{physical}.attn.qkv_proj.weight",
                    f"qkv:{qkv_index}",
                )
            result[f"{logical}.attn.norm_q.weight"] = (
                f"{physical}.attn.q_norm.weight",
                "direct",
            )
            result[f"{logical}.attn.norm_k.weight"] = (
                f"{physical}.attn.k_norm.weight",
                "direct",
            )
            result[f"{logical}.attn.to_out.0.weight"] = (
                f"{physical}.attn.out_proj.weight",
                "direct",
            )
            result[f"{logical}.ff.net.0.proj.weight"] = (
                f"{physical}.mlp.fc1.weight",
                "swap_swiglu",
            )
            result[f"{logical}.ff.net.2.weight"] = (
                f"{physical}.mlp.fc2.weight",
                "direct",
            )
            if has_adaln:
                result[f"{logical}.adaln_proj.linear.weight"] = (
                    f"{physical}.adaln_proj.linear.weight",
                    "direct",
                )
                result[f"{logical}.adaln_proj.linear.bias"] = (
                    f"{physical}.adaln_proj.linear.bias",
                    "direct",
                )
    return result


_LOGICAL_MAP = _logical_contract()


def _validate_group_size(group_size: int, in_features: int) -> None:
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 4:
        raise ValueError("MiniMax-H3 ConvRot group_size must be a power of four")
    value = group_size
    while value > 1 and value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError("MiniMax-H3 ConvRot group_size must be a power of four")
    if in_features % group_size:
        raise ValueError("MiniMax-H3 ConvRot group_size must divide the input width")


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("MiniMax-H3 quantized checkpoint has an incomplete header")
        (header_bytes,) = struct.unpack("<Q", prefix)
        if not 0 < header_bytes <= _MAX_HEADER_BYTES:
            raise ValueError("MiniMax-H3 quantized checkpoint header size is invalid")
        payload = stream.read(header_bytes)
    if len(payload) != header_bytes:
        raise ValueError("MiniMax-H3 quantized checkpoint has an incomplete header")
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MiniMax-H3 quantized checkpoint header is invalid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("MiniMax-H3 quantized checkpoint header must be an object")
    return header, 8 + header_bytes


def _validate_header_inventory(
    header: dict[str, Any], data_offset: int, file_size: int
) -> dict[str, dict[str, Any]]:
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or set(metadata) != {"config"}:
        raise ValueError("MiniMax-H3 quantized checkpoint metadata schema is invalid")
    try:
        config = json.loads(metadata["config"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("MiniMax-H3 quantized checkpoint config metadata is invalid") from error
    if config != _EXPECTED_CONFIG:
        raise ValueError(
            "MiniMax-H3 quantized checkpoint config is not the full released 50-layer architecture"
        )

    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    actual = set(tensors)
    expected = set(_PHYSICAL_SPECS)
    if actual != expected:
        raise ValueError(
            "MiniMax-H3 quantized checkpoint tensor inventory mismatch: "
            f"missing={sorted(expected - actual)[:8]}, unexpected={sorted(actual - expected)[:8]}"
        )

    dtype_bytes = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1}
    regions: list[tuple[int, int, str]] = []
    for name, spec in _PHYSICAL_SPECS.items():
        entry = tensors[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"MiniMax-H3 quantized tensor header is invalid: {name}")
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if entry["dtype"] != spec.dtype or shape != list(spec.shape):
            raise ValueError(
                f"MiniMax-H3 quantized tensor contract mismatch for {name}: "
                f"expected={spec.dtype}{spec.shape}, actual={entry.get('dtype')}{shape}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ValueError(f"MiniMax-H3 quantized tensor offsets are invalid: {name}")
        start, end = offsets
        expected_bytes = math.prod(spec.shape) * dtype_bytes[spec.dtype]
        if start < 0 or end - start != expected_bytes:
            raise ValueError(f"MiniMax-H3 quantized tensor byte range is invalid: {name}")
        regions.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(regions):
        if start != cursor:
            raise ValueError(f"MiniMax-H3 quantized tensor storage is not contiguous before {name}")
        cursor = end
    if data_offset + cursor != file_size:
        raise ValueError("MiniMax-H3 quantized checkpoint payload size is invalid")
    return tensors


def _read_marker(path: Path, data_offset: int, entry: dict[str, Any]) -> dict[str, Any]:
    start, end = entry["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(data_offset + start)
        payload = stream.read(end - start)
    try:
        marker = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MiniMax-H3 comfy_quant marker is invalid JSON") from error
    if not isinstance(marker, dict):
        raise ValueError("MiniMax-H3 comfy_quant marker must be an object")
    return marker


def _source_file_identity(path: Path) -> QuantizedSourceFileIdentity:
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError("MiniMax-H3 quantized checkpoint is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("MiniMax-H3 quantized checkpoint must be a regular file")
    return QuantizedSourceFileIdentity(
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        size_bytes=int(metadata.st_size),
        mtime_ns=int(metadata.st_mtime_ns),
        ctime_ns=int(metadata.st_ctime_ns),
    )


def _authenticate_quantized_source(path: Path) -> None:
    """Validate the pinned HF cache/local-dir layout without rereading 34 GB."""

    relative = Path(CHECKPOINT_FILENAME)
    try:
        source_root = path.parents[len(relative.parts) - 1]
    except IndexError as error:
        raise ValueError("MiniMax-H3 quantized checkpoint path is invalid") from error
    expected_path = source_root / relative
    if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(expected_path)):
        raise ValueError(
            "MiniMax-H3 quantized checkpoint must retain its released diffusion_models path"
        )

    if source_root.parent.name == "snapshots":
        repository_root = source_root.parent.parent
        if source_root.name != CHECKPOINT_REVISION or repository_root.name != _HF_CACHE_REPOSITORY:
            raise ValueError(
                "MiniMax-H3 quantized checkpoint is not from the pinned Hugging Face snapshot"
            )
        return

    metadata_path = source_root / _HF_LOCAL_DOWNLOAD_METADATA / f"{relative.as_posix()}.metadata"
    try:
        metadata = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(
            "Cannot validate the MiniMax-H3 quantized checkpoint source. "
            "Download it with `hf download Comfy-Org/MiniMax-H3 --revision "
            f"{CHECKPOINT_REVISION} --include \"{CHECKPOINT_FILENAME}\" --local-dir <directory>` "
            "so Hugging Face writes pinned download metadata."
        ) from error
    if not metadata or metadata[0] != CHECKPOINT_REVISION:
        raise ValueError(
            "MiniMax-H3 quantized checkpoint Hugging Face metadata does not match the pinned "
            "revision"
        )


def validate_quantized_transformer_checkpoint(
    checkpoint_file: str | Path,
) -> QuantizedCheckpointIdentity:
    """Validate the pinned full FL2VA INT8 ConvRot single-file checkpoint.

    A Hugging Face cache symlink is accepted, but the lexical filename, exact
    byte size, complete safetensors inventory, architecture config, and every
    embedded quantization marker are required.
    """

    path = Path(checkpoint_file)
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 quantized checkpoint is missing: {path}")
    expected_name = Path(QUANTIZED_CHECKPOINT_IDENTITY.filename).name
    if path.name != expected_name:
        raise ValueError(
            f"MiniMax-H3 quantized checkpoint filename must be {expected_name}, got {path.name}"
        )
    size = path.stat().st_size
    if size != QUANTIZED_CHECKPOINT_IDENTITY.size_bytes:
        raise ValueError(
            "MiniMax-H3 quantized checkpoint size mismatch: "
            f"expected={QUANTIZED_CHECKPOINT_IDENTITY.size_bytes}, actual={size}"
        )
    source_identity = _source_file_identity(path)
    _authenticate_quantized_source(path)
    header, data_offset = _read_header(path)
    tensors = _validate_header_inventory(header, data_offset, size)
    for base, group_size in sorted(_QUANT_GROUPS.items()):
        marker_name = f"{base}.comfy_quant"
        marker = _read_marker(path, data_offset, tensors[marker_name])
        expected_marker = {
            "format": _QUANT_FORMAT,
            "convrot": True,
            "convrot_groupsize": group_size,
        }
        if marker != expected_marker:
            raise ValueError(
                f"MiniMax-H3 unsupported comfy_quant config for {base}: {marker!r}"
            )
        _validate_group_size(group_size, _PHYSICAL_SPECS[f"{base}.weight"].shape[1])
    if _source_file_identity(path) != source_identity:
        raise ValueError("MiniMax-H3 quantized checkpoint changed while it was validated")
    return replace(
        QUANTIZED_CHECKPOINT_IDENTITY,
        source_file_identity=source_identity,
    )


def _swap_swiglu_rows(array: np.ndarray) -> np.ndarray:
    if array.shape[0] % 2:
        raise ValueError("MiniMax-H3 fused SwiGLU weight must have an even output width")
    half = array.shape[0] // 2
    return np.ascontiguousarray(np.concatenate((array[half:], array[:half]), axis=0))


def _make_quantized_weight(
    qweight: np.ndarray,
    scale: np.ndarray,
    group_size: int,
    *,
    transform: str,
) -> ConvRotInt8Weight:
    qweight = np.asarray(qweight)
    scale = np.asarray(scale)
    if transform == "swap_swiglu":
        qweight = _swap_swiglu_rows(qweight)
        scale = _swap_swiglu_rows(scale)
    elif transform != "direct":
        raise RuntimeError(f"MiniMax-H3 quantized transform is invalid: {transform}")
    return ConvRotInt8Weight(qweight=qweight, scale=scale, group_size=group_size)


def _make_qkv_values(
    qweight: np.ndarray,
    scale: np.ndarray | None,
    group_size: int | None,
) -> tuple[Any, Any, Any]:
    packed_weight = np.ascontiguousarray(qweight)
    expected_rows = 3 * _NUM_HEADS * _HEAD_DIM
    if packed_weight.shape[0] != expected_rows:
        raise ValueError(
            f"MiniMax-H3 fused QKV has {packed_weight.shape[0]} rows, expected {expected_rows}"
        )
    width = packed_weight.shape[0] // 3
    if scale is None:
        return tuple(packed_weight[index * width : (index + 1) * width] for index in range(3))
    if group_size is None:
        raise RuntimeError("MiniMax-H3 QKV quantization group is missing")
    packed_scale = np.ascontiguousarray(scale)
    parent = ConvRotInt8Weight(
        qweight=packed_weight,
        scale=packed_scale,
        group_size=group_size,
        is_full_fused_qkv=True,
    )
    children = []
    for index in range(3):
        start, end = index * width, (index + 1) * width
        children.append(
            ConvRotInt8Weight(
                qweight=parent.qweight[start:end],
                scale=parent.scale[start:end],
                group_size=group_size,
                packed_parent=parent,
                row_slice=(start, end),
            )
        )
    return tuple(children)


def load_selected_quantized_transformer_weights(
    checkpoint_file: str | Path, logical_names: Iterable[str]
) -> dict[str, Any]:
    """Load only requested frozen-Diffusers logical transformer weights."""

    path = Path(checkpoint_file)
    validate_quantized_transformer_checkpoint(path)
    requested = tuple(logical_names)
    if len(requested) != len(set(requested)):
        raise ValueError("MiniMax-H3 quantized weight request contains duplicate logical names")
    unknown = sorted(set(requested) - set(_LOGICAL_MAP))
    if unknown:
        raise ValueError(f"Unsupported MiniMax-H3 quantized logical tensor names: {unknown}")
    if not requested:
        return {}

    physical_names: set[str] = set()
    for logical_name in requested:
        physical_weight, _transform = _LOGICAL_MAP[logical_name]
        physical_names.add(physical_weight)
        base = physical_weight.removesuffix(".weight")
        if base in _QUANT_GROUPS:
            physical_names.add(f"{base}.weight_scale")

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as reader:
        state = {name: reader.get_tensor(name) for name in sorted(physical_names)}
    arrays = numpy_state(state)

    result: dict[str, Any] = {}
    qkv_cache: dict[str, tuple[Any, Any, Any]] = {}
    quant_cache: dict[tuple[str, str], ConvRotInt8Weight] = {}
    for logical_name in requested:
        physical_weight, transform = _LOGICAL_MAP[logical_name]
        if transform.startswith("qkv:"):
            values = qkv_cache.get(physical_weight)
            if values is None:
                base = physical_weight.removesuffix(".weight")
                scale = arrays.get(f"{base}.weight_scale")
                values = _make_qkv_values(
                    arrays[physical_weight],
                    scale,
                    _QUANT_GROUPS.get(base),
                )
                qkv_cache[physical_weight] = values
            result[logical_name] = values[int(transform.removeprefix("qkv:"))]
            continue
        base = physical_weight.removesuffix(".weight")
        if base in _QUANT_GROUPS:
            cache_key = (physical_weight, transform)
            value = quant_cache.get(cache_key)
            if value is None:
                value = _make_quantized_weight(
                    arrays[physical_weight],
                    arrays[f"{base}.weight_scale"],
                    _QUANT_GROUPS[base],
                    transform=transform,
                )
                quant_cache[cache_key] = value
            result[logical_name] = value
        elif transform == "swap_swiglu":
            result[logical_name] = _swap_swiglu_rows(arrays[physical_weight])
        elif transform == "direct":
            result[logical_name] = arrays[physical_weight]
        else:
            raise RuntimeError(f"MiniMax-H3 logical transform is invalid: {transform}")
    return result
