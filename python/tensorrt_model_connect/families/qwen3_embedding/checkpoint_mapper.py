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

# Register bfloat16 dtype with numpy (needed for safetensors without torch).
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

from safetensors import safe_open

from .config import ModelConfig


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision == "fp16":
        return np.float16
    return np.float32


def _model_key(model_prefix: str, suffix: str) -> str:
    return f"{model_prefix}.{suffix}" if model_prefix else suffix


def _layer_key(layer_idx: int, suffix: str, model_prefix: str = "model") -> str:
    return _model_key(model_prefix, f"layers.{layer_idx}.{suffix}")


def _transpose_2d(arr: np.ndarray, name: str, precision: str = "fp32") -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous target dtype."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=_target_np_dtype(precision))


def _copy_to_numpy(tensor, dtype: np.dtype, *, transpose_name: str | None = None) -> np.ndarray:
    """Copy a checkpoint tensor directly into an owned contiguous NumPy array."""
    if transpose_name is not None and tensor.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {transpose_name}")

    if hasattr(tensor, "numpy"):
        import torch

        source = tensor.transpose(0, 1) if transpose_name is not None else tensor
        torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
        output = torch.empty(tuple(source.shape), dtype=torch_dtype, device="cpu")
        output.copy_(source)
        return output.numpy()

    source = np.asarray(tensor)
    if source.dtype == np.uint16:
        if ml_dtypes is not None:
            source = source.view(ml_dtypes.bfloat16)
        else:
            source = (source.astype(np.uint32) << 16).view(np.float32)
    if transpose_name is not None:
        source = source.T
    return np.array(source, dtype=dtype, order="C", copy=True)


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
    include_lm_head: bool = True,
) -> WeightDict:
    """Load HF safetensors and map to standard weight dict."""
    model_dir = Path(model_dir)
    readers = _open_safetensors(model_dir)
    if (
        model_prefix == "model"
        and not _has_tensor(readers, "model.embed_tokens.weight")
        and _has_tensor(readers, "embed_tokens.weight")
    ):
        model_prefix = ""

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    target_dtype = _target_np_dtype(precision)

    weights = WeightDict()

    # Embedding
    if embedding_key is None:
        embedding_key = _model_key(model_prefix, "embed_tokens.weight")
    embedding = _load_tensor_as_dtype(readers, embedding_key, target_dtype)
    if embedding.shape != (vocab, hidden):
        raise ValueError(
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
    weights["embedding"] = embedding

    def _load_layer(layer_idx: int) -> tuple[int, WeightDict, int, int]:
        prefix = f"layer.{layer_idx}"
        layer = WeightDict()

        # Norms
        input_norm = _load_tensor(
            readers, _layer_key(layer_idx, "input_layernorm.weight", model_prefix))
        post_norm = _load_tensor(
            readers,
            _layer_key(layer_idx, "post_attention_layernorm.weight", model_prefix))
        layer[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        layer[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        # Transpose [out, in] -> [in, out] while copying directly to the
        # final storage dtype. In particular, FP16/BF16 builds must not stage
        # these model-sized tensors through FP32 first.
        q_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.q_proj.weight", model_prefix),
            "q_proj", target_dtype)
        k_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.k_proj.weight", model_prefix),
            "k_proj", target_dtype)
        v_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.v_proj.weight", model_prefix),
            "v_proj", target_dtype)
        o_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.o_proj.weight", model_prefix),
            "o_proj", target_dtype)

        q_hidden = q_t.shape[1]

        layer[f"{prefix}.w_q"] = q_t
        layer[f"{prefix}.w_k"] = k_t
        layer[f"{prefix}.w_v"] = v_t
        layer[f"{prefix}.w_o"] = o_t

        # Optional QKV biases (Qwen2 style)
        q_bias_key = _layer_key(layer_idx, "self_attn.q_proj.bias", model_prefix)
        k_bias_key = _layer_key(layer_idx, "self_attn.k_proj.bias", model_prefix)
        v_bias_key = _layer_key(layer_idx, "self_attn.v_proj.bias", model_prefix)
        if _has_tensor(readers, q_bias_key):
            layer[f"{prefix}.q_bias"] = _load_tensor(
                readers, q_bias_key).astype(target_dtype)
        if _has_tensor(readers, k_bias_key):
            layer[f"{prefix}.k_bias"] = _load_tensor(
                readers, k_bias_key).astype(target_dtype)
        if _has_tensor(readers, v_bias_key):
            layer[f"{prefix}.v_bias"] = _load_tensor(
                readers, v_bias_key).astype(target_dtype)

        # Optional per-head q/k norm (Qwen3 style)
        q_norm_key = _layer_key(layer_idx, "self_attn.q_norm.weight", model_prefix)
        k_norm_key = _layer_key(layer_idx, "self_attn.k_norm.weight", model_prefix)
        if _has_tensor(readers, q_norm_key):
            layer[f"{prefix}.q_norm"] = _repeat_head_norm(
                _load_tensor(readers, q_norm_key).astype(np.float32),
                num_heads)
        if _has_tensor(readers, k_norm_key):
            layer[f"{prefix}.k_norm"] = _repeat_head_norm(
                _load_tensor(readers, k_norm_key).astype(np.float32),
                num_kv_heads)

        # MLP projections
        layer[f"{prefix}.w_gate"] = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "mlp.gate_proj.weight", model_prefix),
            "gate_proj", target_dtype)
        layer[f"{prefix}.w_up"] = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "mlp.up_proj.weight", model_prefix),
            "up_proj", target_dtype)
        layer[f"{prefix}.w_down"] = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "mlp.down_proj.weight", model_prefix),
            "down_proj", target_dtype)
        layer_mlp_size = layer[f"{prefix}.w_gate"].shape[1]

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
        final_norm_key = _model_key(model_prefix, "norm.weight")
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(
            readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    # LM head
    if include_lm_head:
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _load_transposed_tensor(
                readers, lm_head_key, "lm_head", target_dtype)
        else:
            # Tied embeddings
            weights["w_out"] = _transpose_2d(
                embedding, "embedding_tied", precision=precision)

    weights["_attention_size"] = attention_size  # type: ignore[assignment]
    weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
    weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

    return weights


# ---------------------------------------------------------------------------
# Safetensors I/O helpers
# ---------------------------------------------------------------------------

def _detect_framework() -> str:
    """Use 'torch' if available (handles BF16 natively), else 'numpy'."""
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        return "numpy"


class _TorchBinReader:
    """Adapter that wraps a pytorch .bin state dict with the safetensors reader
    interface (keys() / get_tensor())."""

    def __init__(self, path: Path):
        import torch
        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self) -> list[str]:
        return list(self._state.keys())

    def get_tensor(self, name: str):
        return self._state[name]


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
    """Open all safetensor shards (or pytorch .bin) in a model directory."""
    fw = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework=fw)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework=fw)
            for shard in shard_files
        }
        tensor_map = {
            name: readers_by_file[shard]
            for name, shard in weight_map.items()
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    # Diffusers format: diffusion_pytorch_model.safetensors
    diff_single = model_dir / "diffusion_pytorch_model.safetensors"
    if diff_single.exists():
        return _ReaderCollection([safe_open(str(diff_single), framework=fw)])

    diff_index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if diff_index.exists():
        import json
        index = json.loads(diff_index.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework=fw)
            for shard in shard_files
        }
        tensor_map = {
            name: readers_by_file[shard]
            for name, shard in weight_map.items()
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    # Fallback: pytorch_model.bin (older HF models)
    bin_single = model_dir / "pytorch_model.bin"
    if bin_single.exists():
        return _ReaderCollection([_TorchBinReader(bin_single)])

    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}")


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
    if hasattr(t, "numpy"):
        dtype = getattr(t, "dtype", None)
        if str(dtype) == "torch.float32":
            return t.numpy()
        return t.float().numpy()

    dtype_str = str(t.dtype)
    if t.dtype == np.uint16 or dtype_str == "bfloat16":
        t = t.view(np.uint16).astype(np.uint32) << 16
        return t.view(np.float32)
    if dtype_str == "float16":
        return t.astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _get_tensor(readers: list, name: str):
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


def _load_tensor_as_dtype(readers: list, name: str, dtype: np.dtype) -> np.ndarray:
    return _copy_to_numpy(_get_tensor(readers, name), dtype)


def _load_transposed_tensor(
    readers: list,
    name: str,
    transpose_name: str,
    dtype: np.dtype,
) -> np.ndarray:
    return _copy_to_numpy(
        _get_tensor(readers, name), dtype, transpose_name=transpose_name)


def _load_tensor(readers: list, name: str) -> np.ndarray:
    return _to_numpy_fp32(_get_tensor(readers, name))
