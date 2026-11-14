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

# Register NumPy bfloat16 support required by safetensors.
import ml_dtypes

from safetensors import safe_open

from .prepared_quantization import PreparedLinear


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision in ("fp16", "bf16"):
        return np.float16
    return np.float32


def _transpose_2d(arr: np.ndarray, name: str, precision: str = "fp32") -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous target dtype."""
    if isinstance(arr, PreparedLinear):
        return arr.transpose()
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


class PreparedSafetensorsReader:
    """Validated, read-only NumPy access without a training-framework dependency.

    safe_open validates the file layout first. Its NumPy backend does not expose
    FP8 on all supported NumPy versions, so map those bytes with the declared
    ml_dtypes dependency rather than modifying NumPy or unpacking whole shards.
    """

    def __init__(self, path: Path):
        self.path = path
        self.reader = safe_open(str(path), framework="numpy")
        with path.open("rb") as source:
            length = int.from_bytes(source.read(8), "little")
            self.header = json.loads(source.read(length))
        self.data_offset = 8 + length

    def keys(self):
        return self.reader.keys()

    def get_tensor(self, name: str):
        tensor = self.header[name]
        dtype = {"F8_E4M3": ml_dtypes.float8_e4m3fn, "BF16": ml_dtypes.bfloat16}.get(
            tensor["dtype"]
        )
        if dtype is None:
            return self.reader.get_tensor(name)
        # Layout, shape, dtype and data bounds were already checked by safe_open.
        return np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            shape=tuple(tensor["shape"]),
            offset=self.data_offset + tensor["data_offsets"][0],
        )


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    """Open the checkpoint's required NumPy safetensors readers."""
    import json

    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([PreparedSafetensorsReader(single)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        shard_files = sorted(set(str(value) for value in weight_map.values()))
        readers_by_file = {
            shard: PreparedSafetensorsReader(model_dir / shard) for shard in shard_files
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map={
                str(name): readers_by_file[str(shard)] for name, shard in weight_map.items()
            },
        )

    diff_single = model_dir / "diffusion_pytorch_model.safetensors"
    if diff_single.is_file():
        return _ReaderCollection([PreparedSafetensorsReader(diff_single)])

    diff_index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if diff_index.is_file():
        index = json.loads(diff_index.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("diffusion safetensors index has no weight_map")
        shard_files = sorted(set(str(value) for value in weight_map.values()))
        readers_by_file = {
            shard: PreparedSafetensorsReader(model_dir / shard) for shard in shard_files
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
