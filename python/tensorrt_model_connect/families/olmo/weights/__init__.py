# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load OLMo checkpoints into the family-owned decoder weight layout."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Register bfloat16 dtype with numpy (needed for safetensors without torch).
try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open

from ..config import ModelConfig


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision in ("fp16", "bf16"):
        return np.float16
    return np.float32


def _layer_key(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


def _transpose_2d(arr: np.ndarray, name: str, precision: str = "fp32") -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous target dtype."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=_target_np_dtype(precision))


class WeightDict(dict):
    """OLMo builder weights keyed by embedding, projection, and norm name."""


def load_standard_weights(
    model_dir: str | Path,
    config: ModelConfig,
    *,
    precision: str = "fp32",
) -> WeightDict:
    """Load a standard Hugging Face OLMo checkpoint."""
    model_dir = Path(model_dir)
    readers = _open_safetensors(model_dir)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    target_dtype = _target_np_dtype(precision)

    weights = WeightDict()

    # Embedding
    embedding = _load_tensor(readers, "model.embed_tokens.weight")
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(target_dtype)

    def _load_layer(layer_idx: int) -> tuple[int, WeightDict, int, int]:
        prefix = f"layer.{layer_idx}"
        layer = WeightDict()

        # OLMo v1 norms have no parameters; accept weighted HF variants too.
        input_norm_key = _layer_key(layer_idx, "input_layernorm.weight")
        post_norm_key = _layer_key(layer_idx, "post_attention_layernorm.weight")
        layer[f"{prefix}.input_norm"] = (
            _load_tensor(readers, input_norm_key).astype(np.float32)
            if _has_tensor(readers, input_norm_key)
            else np.ones(hidden, dtype=np.float32)
        )
        layer[f"{prefix}.input_norm_beta"] = np.zeros(hidden, dtype=np.float32)
        layer[f"{prefix}.post_attn_norm"] = (
            _load_tensor(readers, post_norm_key).astype(np.float32)
            if _has_tensor(readers, post_norm_key)
            else np.ones(hidden, dtype=np.float32)
        )
        layer[f"{prefix}.post_attn_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        # Q/K/V/O projections
        q_raw = _load_tensor(readers, _layer_key(layer_idx, "self_attn.q_proj.weight"))
        k_raw = _load_tensor(readers, _layer_key(layer_idx, "self_attn.k_proj.weight"))
        v_raw = _load_tensor(readers, _layer_key(layer_idx, "self_attn.v_proj.weight"))
        o_raw = _load_tensor(readers, _layer_key(layer_idx, "self_attn.o_proj.weight"))

        q_hidden = q_raw.shape[0]
        gate_raw = _load_tensor(readers, _layer_key(layer_idx, "mlp.gate_proj.weight"))
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

        # MLP projections
        up_raw = _load_tensor(readers, _layer_key(layer_idx, "mlp.up_proj.weight"))
        down_raw = _load_tensor(readers, _layer_key(layer_idx, "mlp.down_proj.weight"))

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
    final_norm_key = "model.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)
    weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    # LM head
    if _has_tensor(readers, "lm_head.weight"):
        weights["w_out"] = _transpose_2d(
            _load_tensor(readers, "lm_head.weight"), "lm_head", precision=precision
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
            shard: safe_open(str(model_dir / shard), framework=fw) for shard in shard_files
        }
        tensor_map = {name: readers_by_file[shard] for name, shard in weight_map.items()}
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    # Fallback: pytorch_model.bin (older HF models)
    bin_single = model_dir / "pytorch_model.bin"
    if bin_single.exists():
        return _ReaderCollection([_TorchBinReader(bin_single)])

    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}"
    )


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
