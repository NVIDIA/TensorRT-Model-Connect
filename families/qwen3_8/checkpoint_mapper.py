# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1:1 port of standard_checkpoint_mapper.cpp + tensor_math.cpp to Python.

Loads HF safetensors and maps keys to the flat weight dict expected by
standard_decoder_builder.py. All projections are transposed from HF
[out, in] layout to [in, out] for TRT matmul.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision in ("fp16", "bf16"):
        return np.float16
    return np.float32


def _transpose_2d(arr: np.ndarray, name: str, precision: str = "fp32") -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous target dtype."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=_target_np_dtype(precision))


class WeightDict(dict):
    """A dict mapping logical weight names to flat float32 arrays.

    Keys follow the convention used by standard_decoder_builder.py:
      - embedding: [vocab, hidden]
      - layer.{i}.input_norm: [hidden]
      - layer.{i}.w_q: [hidden, attention_size]
      - layer.{i}.w_k: [hidden, kv_attention_size]
      - layer.{i}.w_v: [hidden, kv_attention_size]
      - layer.{i}.q_bias: [attention_size]       (optional)
      - layer.{i}.k_bias: [kv_attention_size]    (optional)
      - layer.{i}.v_bias: [kv_attention_size]    (optional)
      - layer.{i}.q_norm: [attention_size]        (optional)
      - layer.{i}.k_norm: [kv_attention_size]     (optional)
      - layer.{i}.w_o: [attention_size, hidden]
      - layer.{i}.post_attn_norm: [hidden]
      - layer.{i}.w_gate: [hidden, mlp_size]
      - layer.{i}.w_up: [hidden, mlp_size]
      - layer.{i}.w_down: [mlp_size, hidden]
      - final_norm: [hidden]
      - w_out: [hidden, vocab]
    """


class _ReaderCollection(list):
    """Reader list with a cached tensor-name -> reader lookup table."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {}
            for reader in readers:
                for key in reader.keys():
                    tensor_map[key] = reader
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> list:
    """Open one Hugging Face safetensors checkpoint."""
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework="pt")])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework="pt")
            for shard in shard_files
        }
        tensor_map = {name: readers_by_file[shard] for name, shard in weight_map.items()}
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    raise FileNotFoundError(f"No model.safetensors checkpoint in {model_dir}")


def _has_tensor(readers: list, name: str) -> bool:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        return name in tensor_map
    for r in readers:
        if name in r.keys():
            return True
    return False


def _to_numpy_fp32(t) -> np.ndarray:
    """Convert a safetensors/torch tensor to numpy float32 with minimal copies."""
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.float32:
            return t.numpy()
        return t.float().numpy()

    dtype_str = str(t.dtype)
    if t.dtype == np.uint16 or dtype_str == "bfloat16":
        t = t.view(np.uint16).astype(np.uint32) << 16
        return t.view(np.float32)
    if dtype_str == "float16":
        return t.astype(np.float32)
    return np.asarray(t, dtype=np.float32)



# Qwen3.8 FP8 checkpoints store the large projections as float8_e4m3 together
# with a companion "<name>_scale_inv" tensor holding one bf16 scale per
# weight_block_size block (128x128 for Qwen3.8-27B-FP8). Converting the raw
# float8 values without applying those scales silently yields wrong weights, so
# the two are always resolved together.
_FP8_DTYPE_NAMES = frozenset({
    "torch.float8_e4m3fn", "float8_e4m3fn", "float8_e4m3",
    "torch.float8_e5m2", "float8_e5m2",
})
_SCALE_INV_SUFFIX = "_scale_inv"


def _is_fp8_tensor(t) -> bool:
    return str(getattr(t, "dtype", "")) in _FP8_DTYPE_NAMES


def _apply_block_scales(values: np.ndarray, scale_inv: np.ndarray) -> np.ndarray:
    """Scale a dequantized FP8 weight in place by its per-block scales.

    ``scale_inv`` carries one entry per block; block extents are derived from the
    two shapes so the loader does not need to read weight_block_size, and a
    trailing partial block is handled by clipping. The multiply is done block by
    block rather than by expanding the scales, because an expanded scale array
    would be as large as the weight itself.
    """
    if values.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"FP8 block dequantization expects 2-D weight and scales, got "
            f"{values.ndim}-D and {scale_inv.ndim}-D")
    rows, cols = values.shape
    s_rows, s_cols = scale_inv.shape
    block_r = -(-rows // s_rows)
    block_c = -(-cols // s_cols)
    for i in range(s_rows):
        r0, r1 = i * block_r, min((i + 1) * block_r, rows)
        if r0 >= r1:
            break
        for j in range(s_cols):
            c0, c1 = j * block_c, min((j + 1) * block_c, cols)
            if c0 >= c1:
                break
            values[r0:r1, c0:c1] *= scale_inv[i, j]
    return values



# ModelOpt MIXED_PRECISION checkpoints (RadixArk/Qwen3.8-27B-NVFP4) carry two
# schemes side by side, described by quantization_config.config_groups:
#
#   FP8   attention and DeltaNet projections: float8_e4m3 weights with a single
#         per-tensor "<name>.weight_scale".
#   NVFP4 MLP projections and lm_head: 4-bit E2M1 values packed two per uint8,
#         a per-16-element "<name>.weight_scale" stored as float8_e4m3, and a
#         global "<name>.weight_scale_2".
#
# "<name>.input_scale" describes activations and is unused here: the graph runs
# in fp16 and quantizes nothing at runtime.
_PER_TENSOR_SCALE_SUFFIX = ".weight_scale"
_GLOBAL_SCALE_SUFFIX = ".weight_scale_2"

# E2M1: one sign bit, two exponent bits, one mantissa bit, exponent bias 1.
# Subnormals give 0 and 0.5; the normals are 1, 1.5, 2, 3, 4, 6.
_E2M1_MAGNITUDES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _scale_key(name: str, suffix: str) -> str:
    """Map "<...>.weight" to its sibling scale tensor name."""
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    return base + suffix


def _decode_e2m1(nibbles: np.ndarray) -> np.ndarray:
    magnitudes = _E2M1_MAGNITUDES[nibbles & 0x07]
    return np.where(nibbles & 0x08, -magnitudes, magnitudes)


def _unpack_nvfp4(packed: np.ndarray) -> np.ndarray:
    """Expand uint8 pairs of E2M1 values into a float32 array of twice the width.

    The low nibble holds the even column and the high nibble the odd one, which
    is the packing ModelOpt emits.
    """
    if packed.ndim != 2:
        raise ValueError(f"NVFP4 weights must be 2-D, got {packed.ndim}-D")
    packed = packed.astype(np.uint8, copy=False)
    rows, packed_cols = packed.shape
    out = np.empty((rows, packed_cols * 2), dtype=np.float32)
    out[:, 0::2] = _decode_e2m1(packed & 0x0F)
    out[:, 1::2] = _decode_e2m1(packed >> 4)
    return out


def _apply_group_scales(values: np.ndarray, group_scale: np.ndarray,
                        global_scale: float) -> np.ndarray:
    """Scale unpacked NVFP4 values by their per-group and global scales.

    Reshaping to (rows, groups, group_size) lets the per-group multiply happen in
    place; expanding the scales to the weight's own size would cost as much
    memory again, and lm_head alone is 248320 x 5120.
    """
    rows, cols = values.shape
    s_rows, s_cols = group_scale.shape
    if s_rows != rows or cols % s_cols:
        raise ValueError(
            f"NVFP4 scale shape {group_scale.shape} does not tile weight shape "
            f"{values.shape}")
    group_size = cols // s_cols
    view = values.reshape(rows, s_cols, group_size)
    view *= group_scale.astype(np.float32).reshape(rows, s_cols, 1)
    return (values * np.float32(global_scale)) if global_scale != 1.0 else values


def _load_tensor(readers: list, name: str) -> np.ndarray:
    raw = _get_raw_tensor(readers, name)

    # NVFP4: 4-bit values packed two per uint8, identified by the global scale
    # that only this scheme carries. Checked before the dtype tests because the
    # payload arrives as plain uint8.
    global_key = _scale_key(name, _GLOBAL_SCALE_SUFFIX)
    if _has_tensor(readers, global_key):
        group_key = _scale_key(name, _PER_TENSOR_SCALE_SUFFIX)
        if not _has_tensor(readers, group_key):
            raise KeyError(
                f"NVFP4 tensor {name!r} has {global_key!r} but no {group_key!r}; "
                "cannot dequantize")
        packed = np.asarray(raw.numpy() if hasattr(raw, "numpy") else raw)
        group_scale = _to_numpy_fp32(_get_raw_tensor(readers, group_key))
        global_scale = float(
            _to_numpy_fp32(_get_raw_tensor(readers, global_key)).reshape(-1)[0])
        return _apply_group_scales(_unpack_nvfp4(packed), group_scale, global_scale)

    values = _to_numpy_fp32(raw)
    if not _is_fp8_tensor(raw):
        return values

    # Qwen native FP8: one scale per weight_block_size block.
    block_key = name + _SCALE_INV_SUFFIX
    if _has_tensor(readers, block_key):
        block_scale = _to_numpy_fp32(_get_raw_tensor(readers, block_key))
        return _apply_block_scales(values, block_scale)

    # ModelOpt FP8: a single per-tensor scale.
    tensor_key = _scale_key(name, _PER_TENSOR_SCALE_SUFFIX)
    if _has_tensor(readers, tensor_key):
        scale = _to_numpy_fp32(_get_raw_tensor(readers, tensor_key)).reshape(-1)[0]
        values *= np.float32(scale)
        return values

    # Refuse to return unscaled float8 values: they look like a plausible weight
    # tensor and would corrupt the engine silently.
    raise KeyError(
        f"FP8 tensor {name!r} has no companion {block_key!r} or {tensor_key!r}; "
        "cannot dequantize")


def _get_raw_tensor(readers: list, name: str):
    """Fetch a tensor from the shard collection without dtype conversion."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return reader.get_tensor(name)
    for r in readers:
        if name in r.keys():
            return r.get_tensor(name)
    raise KeyError(f"Tensor not found: {name}")
