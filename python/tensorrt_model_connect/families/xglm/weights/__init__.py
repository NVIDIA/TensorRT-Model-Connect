# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XGLM checkpoint readers and family-specific weight mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass


class WeightDict(dict):
    """Flat, family-owned weights consumed by the XGLM graph builders."""


class _TorchBinReader:
    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self):
        return self._state.keys()

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    def __init__(self, readers: list, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        self.tensor_map = (
            tensor_map
            if tensor_map is not None
            else {name: reader for reader in readers for name in reader.keys()}
        )


def _framework() -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy"
    return "torch"


def _open_checkpoint(model_dir: Path) -> _ReaderCollection:
    framework = _framework()
    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([safe_open(str(single), framework=framework)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        readers = {
            shard: safe_open(str(model_dir / shard), framework=framework)
            for shard in sorted(set(weight_map.values()))
        }
        return _ReaderCollection(
            list(readers.values()),
            {name: readers[shard] for name, shard in weight_map.items()},
        )

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return _ReaderCollection([_TorchBinReader(bin_path)])
    raise FileNotFoundError(f"No XGLM checkpoint found in {model_dir}")


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        if str(getattr(tensor, "dtype", "")) == "torch.float32":
            return tensor.numpy()
        return tensor.float().numpy()
    if tensor.dtype == np.uint16 or str(tensor.dtype) == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return np.ascontiguousarray(_to_numpy_fp32(reader.get_tensor(name)), dtype=np.float32)


def _transpose_2d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for {name}, got rank {array.ndim}")
    return np.ascontiguousarray(array.T, dtype=np.float32)


def _sinusoidal_positions(num_positions: int, hidden: int) -> np.ndarray:
    half = hidden // 2
    frequencies = np.exp(np.arange(half, dtype=np.float32) * -(np.log(10000.0) / (half - 1)))
    angles = np.arange(num_positions, dtype=np.float32)[:, None] * frequencies[None, :]
    table = np.zeros((num_positions, hidden), dtype=np.float32)
    table[:, :half] = np.sin(angles)
    table[:, half : 2 * half] = np.cos(angles)
    table[1] = 0.0
    return table


def load_standard_weights(model_dir: str, config) -> WeightDict:
    """Load the exact biased XGLM decoder layout from a HF checkpoint."""
    readers = _open_checkpoint(Path(model_dir))
    hidden = config.hidden_size
    if hidden % config.num_attention_heads:
        raise ValueError("XGLM hidden size must be divisible by attention heads")

    embedding = _load_tensor(readers, "model.embed_tokens.weight")
    expected_embedding = (config.vocab_size, hidden)
    if embedding.shape != expected_embedding:
        raise ValueError(
            f"XGLM token embedding must have shape {expected_embedding}, got {embedding.shape}"
        )

    weights = WeightDict()
    scale = np.float32(np.sqrt(hidden)) if config.raw.get("scale_embedding", False) else 1.0
    weights["embedding"] = np.ascontiguousarray(embedding * scale, dtype=np.float32)
    positions = _sinusoidal_positions(config.max_position_embeddings + 2, hidden)
    weights["position_embedding"] = np.ascontiguousarray(positions[2:])

    mlp_size = config.intermediate_size
    for layer in range(config.num_hidden_layers):
        prefix = f"layer.{layer}"
        hf = f"model.layers.{layer}"
        for logical, checkpoint in (
            ("input_norm", "self_attn_layer_norm.weight"),
            ("input_norm_beta", "self_attn_layer_norm.bias"),
            ("post_attn_norm", "final_layer_norm.weight"),
            ("post_attn_norm_beta", "final_layer_norm.bias"),
            ("q_bias", "self_attn.q_proj.bias"),
            ("k_bias", "self_attn.k_proj.bias"),
            ("v_bias", "self_attn.v_proj.bias"),
            ("o_bias", "self_attn.out_proj.bias"),
            ("fc1_bias", "fc1.bias"),
            ("fc2_bias", "fc2.bias"),
        ):
            weights[f"{prefix}.{logical}"] = _load_tensor(readers, f"{hf}.{checkpoint}")
        for logical, checkpoint in (
            ("w_q", "self_attn.q_proj.weight"),
            ("w_k", "self_attn.k_proj.weight"),
            ("w_v", "self_attn.v_proj.weight"),
            ("w_o", "self_attn.out_proj.weight"),
            ("w_fc1", "fc1.weight"),
            ("w_fc2", "fc2.weight"),
        ):
            weights[f"{prefix}.{logical}"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.{checkpoint}"), checkpoint
            )
        mlp_size = int(weights[f"{prefix}.w_fc1"].shape[1])

    final_norm = "model.layer_norm.weight"
    if _has_tensor(readers, final_norm):
        weights["final_norm"] = _load_tensor(readers, final_norm)
        final_beta = "model.layer_norm.bias"
        weights["final_norm_beta"] = (
            _load_tensor(readers, final_beta)
            if _has_tensor(readers, final_beta)
            else np.zeros(hidden, dtype=np.float32)
        )
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)
        weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    output = (
        _load_tensor(readers, "lm_head.weight")
        if _has_tensor(readers, "lm_head.weight")
        else embedding
    )
    weights["w_out"] = _transpose_2d(output, "lm_head.weight")
    weights["_attention_size"] = hidden
    weights["_kv_attention_size"] = hidden
    weights["_mlp_size"] = mlp_size
    return weights
