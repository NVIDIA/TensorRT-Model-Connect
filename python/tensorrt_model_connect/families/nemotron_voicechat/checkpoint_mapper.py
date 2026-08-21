# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy-only loading for the pinned FP32 VoiceChat checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from safetensors import safe_open


def _transpose_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Transpose [rows, cols] to contiguous FP32 [cols, rows]."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=np.float32)


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
    """Open the exact public VoiceChat safetensors with NumPy-only weight I/O."""
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"VoiceChat model.safetensors not found in {model_dir}")
    return _ReaderCollection([safe_open(str(single), framework="np")])


def _has_tensor(readers: list, name: str) -> bool:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        return name in tensor_map
    for r in readers:
        if name in r.keys():
            return True
    return False


def _to_numpy_fp32(t) -> np.ndarray:
    """Materialize a pinned checkpoint tensor as FP32."""
    return np.asarray(t, dtype=np.float32)


def _load_tensor(readers: list, name: str) -> np.ndarray:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return _to_numpy_fp32(reader.get_tensor(name))
    for r in readers:
        if name in r.keys():
            return _to_numpy_fp32(r.get_tensor(name))
    raise KeyError(f"Tensor not found: {name}")
