# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1:1 port of standard_checkpoint_mapper.cpp + tensor_math.cpp to Python.

Loads HF safetensors and maps keys to the flat weight dict expected by
standard_decoder_builder.py. All projections are transposed from HF
[out, in] layout to [in, out] for TRT matmul.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Register NumPy bfloat16 support required by safetensors.
import ml_dtypes  # noqa: F401

from safetensors import safe_open

from .config import ModelConfig


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision in ("fp16", "bf16"):
        return np.float16
    return np.float32


def _layer_key(layer_idx: int, suffix: str, model_prefix: str = "model") -> str:
    return f"{model_prefix}.layers.{layer_idx}.{suffix}"


def _transpose_2d(arr: np.ndarray, name: str, precision: str = "fp32") -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous target dtype."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=_target_np_dtype(precision))


def _repeat_head_norm(norm: np.ndarray, num_heads: int) -> np.ndarray:
    """Repeat per-head norm [head_dim] -> [num_heads * head_dim]."""
    return np.tile(norm, num_heads).astype(np.float32)


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


def load_standard_weights(
    model_dir: str | Path,
    config: ModelConfig,
    *,
    precision: str = "fp32",
    model_prefix: str = "model",
    embedding_key: str | None = None,
    final_norm_key: str | None = None,
    lm_head_key: str = "lm_head.weight",
) -> WeightDict:
    """Load HF safetensors and map to standard weight dict."""
    model_dir = Path(model_dir)
    readers = _open_safetensors(model_dir)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    target_dtype = _target_np_dtype(precision)

    weights = WeightDict()

    # Embedding
    if embedding_key is None:
        embedding_key = f"{model_prefix}.embed_tokens.weight"
    embedding = _load_tensor(readers, embedding_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(target_dtype)

    def _load_layer(layer_idx: int) -> tuple[int, WeightDict, int, int]:
        prefix = f"layer.{layer_idx}"
        layer = WeightDict()

        # Norms
        input_norm = _load_tensor(
            readers, _layer_key(layer_idx, "input_layernorm.weight", model_prefix)
        )
        post_norm = _load_tensor(
            readers, _layer_key(layer_idx, "post_attention_layernorm.weight", model_prefix)
        )
        layer[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        layer[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        q_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.q_proj.weight", model_prefix)
        )
        k_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.k_proj.weight", model_prefix)
        )
        v_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.v_proj.weight", model_prefix)
        )
        o_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.o_proj.weight", model_prefix)
        )

        q_hidden = q_raw.shape[0]
        gate_raw = _load_tensor(
            readers, _layer_key(layer_idx, "mlp.gate_proj.weight", model_prefix)
        )
        layer_mlp_size = gate_raw.shape[0]

        # Transpose all projections [out, in] -> [in, out]
        q_t = _transpose_2d(q_raw, "q_proj", precision=precision)
        k_t = _transpose_2d(k_raw, "k_proj", precision=precision)
        v_t = _transpose_2d(v_raw, "v_proj", precision=precision)
        o_t = _transpose_2d(o_raw, "o_proj", precision=precision)

        layer[f"{prefix}.w_q"] = q_t
        layer[f"{prefix}.w_k"] = k_t
        layer[f"{prefix}.w_v"] = v_t
        layer[f"{prefix}.w_o"] = o_t

        # Optional QKV biases (Qwen2 style)
        q_bias_key = _layer_key(layer_idx, "self_attn.q_proj.bias", model_prefix)
        k_bias_key = _layer_key(layer_idx, "self_attn.k_proj.bias", model_prefix)
        v_bias_key = _layer_key(layer_idx, "self_attn.v_proj.bias", model_prefix)
        if _has_tensor(readers, q_bias_key):
            layer[f"{prefix}.q_bias"] = _load_tensor(readers, q_bias_key).astype(target_dtype)
        if _has_tensor(readers, k_bias_key):
            layer[f"{prefix}.k_bias"] = _load_tensor(readers, k_bias_key).astype(target_dtype)
        if _has_tensor(readers, v_bias_key):
            layer[f"{prefix}.v_bias"] = _load_tensor(readers, v_bias_key).astype(target_dtype)

        # Optional per-head q/k norm (Qwen3 style)
        q_norm_key = _layer_key(layer_idx, "self_attn.q_norm.weight", model_prefix)
        k_norm_key = _layer_key(layer_idx, "self_attn.k_norm.weight", model_prefix)
        if _has_tensor(readers, q_norm_key):
            layer[f"{prefix}.q_norm"] = _repeat_head_norm(
                _load_tensor(readers, q_norm_key).astype(np.float32), num_heads
            )
        if _has_tensor(readers, k_norm_key):
            layer[f"{prefix}.k_norm"] = _repeat_head_norm(
                _load_tensor(readers, k_norm_key).astype(np.float32), num_kv_heads
            )

        # MLP projections
        up_raw = _load_tensor(readers, _layer_key(layer_idx, "mlp.up_proj.weight", model_prefix))
        down_raw = _load_tensor(
            readers, _layer_key(layer_idx, "mlp.down_proj.weight", model_prefix)
        )

        layer[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj", precision=precision)
        layer[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj", precision=precision)
        layer[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj", precision=precision)

        return layer_idx, layer, q_hidden, layer_mlp_size

    layer_results: list[tuple[int, WeightDict, int, int] | None] = [None] * num_layers
    max_workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_layer, i) for i in range(num_layers)]
        for future in as_completed(futures):
            layer_idx, layer, attention_size, mlp_size = future.result()
            layer_results[layer_idx] = (layer_idx, layer, attention_size, mlp_size)

    attention_size = 0
    kv_attention_size = 0
    mlp_size = 0
    for result in layer_results:
        if result is None:
            continue
        _layer_idx, layer, layer_attention_size, layer_mlp_size = result
        weights.update(layer)
        if attention_size == 0:
            attention_size = layer_attention_size
            first_k = layer[f"layer.{_layer_idx}.w_k"]
            kv_attention_size = int(first_k.shape[1])
        if mlp_size == 0:
            mlp_size = layer_mlp_size

    # Final norm
    if final_norm_key is None:
        final_norm_key = f"{model_prefix}.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    # LM head
    if _has_tensor(readers, lm_head_key):
        weights["w_out"] = _transpose_2d(
            _load_tensor(readers, lm_head_key), "lm_head", precision=precision
        )
    else:
        # Tied embeddings
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied", precision=precision)

    weights["_attention_size"] = attention_size  # type: ignore[assignment]
    weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
    weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

    return weights


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


def _is_float8_tensor(t) -> bool:
    """Return whether *t* stores a supported float8 checkpoint tensor."""
    dtype_name = str(getattr(t, "dtype", "")).lower()
    return "float8_e4m3" in dtype_name or "f8_e4m3" in dtype_name


def _get_raw_tensor(readers: list, name: str):
    """Look up a tensor without converting away its checkpoint dtype."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return reader.get_tensor(name)
    for reader in readers:
        if name in reader.keys():
            return reader.get_tensor(name)
    raise KeyError(f"Tensor not found: {name}")


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
