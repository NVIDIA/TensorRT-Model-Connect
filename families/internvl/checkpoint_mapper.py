# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1:1 port of standard_checkpoint_mapper.cpp + tensor_math.cpp to Python.

Loads HF safetensors and maps keys to the flat weight dict expected by
standard_decoder_builder.py. All projections are transposed from HF
[out, in] layout to [in, out] for TRT matmul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Register NumPy bfloat16 support required by safetensors.
import ml_dtypes  # noqa: F401

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


# ---------------------------------------------------------------------------
# Safetensors I/O helpers
# ---------------------------------------------------------------------------
class _ReaderCollection(list):
    """Checkpoint readers with one exact tensor-to-reader index."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {name: reader for reader in readers for name in reader.keys()}
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    """Open the checkpoint's required NumPy safetensors readers."""
    import json

    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([safe_open(str(single), framework="numpy")])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        shard_files = sorted(set(str(value) for value in weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework="numpy") for shard in shard_files
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map={
                str(name): readers_by_file[str(shard)] for name, shard in weight_map.items()
            },
        )

    diff_single = model_dir / "diffusion_pytorch_model.safetensors"
    if diff_single.is_file():
        return _ReaderCollection([safe_open(str(diff_single), framework="numpy")])

    diff_index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if diff_index.is_file():
        index = json.loads(diff_index.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("diffusion safetensors index has no weight_map")
        shard_files = sorted(set(str(value) for value in weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework="numpy") for shard in shard_files
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map={
                str(name): readers_by_file[str(shard)] for name, shard in weight_map.items()
            },
        )

    raise FileNotFoundError(f"No supported safetensors checkpoint in {model_dir}")


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    """Copy a NumPy or ml_dtypes checkpoint tensor to float32."""
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))


def _load_projection(readers: _ReaderCollection, prefix: str, quantization: dict | None) -> np.ndarray:
    """Return [out, in] weights, retaining the values of a source AWQ checkpoint.

    Native fallback uses ordinary TensorRT matmuls, not an AWQ CUDA plugin. AWQ
    GEMM packs eight output columns per INT32 in order [0,2,4,6,1,3,5,7]. Unlike
    GPTQ, its packed zero points have no +1 offset. Reconstruct each group once
    at build time and round to the checkpoint scale dtype (W4A16 semantics).
    This expands weight storage; it does not promise packed INT4 residency.
    """
    if _has_tensor(readers, f"{prefix}.weight"):
        return _load_tensor(readers, f"{prefix}.weight")
    if not quantization or not (
        quantization.get("quant_method") == "awq"
        and quantization.get("bits") == 4
        and quantization.get("group_size") == 128
        and str(quantization.get("version", "")).lower() == "gemm"
        and quantization.get("zero_point") is True
    ):
        raise ValueError(f"{prefix}: native packed weights require group128 GEMM AWQ with zero points")

    def packed(suffix: str) -> np.ndarray:
        name = f"{prefix}.{suffix}"
        reader = readers.tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        value = np.asarray(reader.get_tensor(name))
        if value.ndim != 2 or value.dtype != np.int32:
            raise ValueError(f"{name}: expected a rank-2 INT32 packed tensor")
        # Work in unsigned integers so the high nibble survives sign extension.
        shifts = np.array([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.uint32)
        return ((value.astype(np.uint32)[..., None] >> shifts) & 15).reshape(value.shape[0], -1)

    values, zeros = packed("qweight"), packed("qzeros")
    scale_name = f"{prefix}.scales"
    if not _has_tensor(readers, scale_name):
        raise KeyError(f"Tensor not found: {scale_name}")
    scales = np.asarray(readers.tensor_map[scale_name].get_tensor(scale_name))
    group_size = int(quantization["group_size"])
    if (values.shape[0] % group_size or scales.shape != zeros.shape
            or scales.shape != (values.shape[0] // group_size, values.shape[1])):
        raise ValueError(f"{prefix}: inconsistent AWQ weights, zero points, or group scales")
    if scales.dtype not in (np.float16, np.float32) or not np.isfinite(scales).all():
        raise ValueError(f"{prefix}: AWQ scales must be finite floating-point values")
    grouped = values.reshape(-1, group_size, values.shape[1]).astype(np.float32)
    decoded = (grouped - zeros[:, None, :].astype(np.float32)) * scales[:, None, :]
    return np.ascontiguousarray(decoded.astype(scales.dtype).reshape(values.shape).T, dtype=np.float32)
