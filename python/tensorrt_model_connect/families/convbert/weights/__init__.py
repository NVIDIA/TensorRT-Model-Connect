# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Hugging Face ConvBERT checkpoints into family-owned graph weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

if TYPE_CHECKING:
    from ..config import ModelConfig


class WeightDict(dict):
    """ConvBERT embeddings, hybrid-attention, convolution, and FFN weights."""


class _TorchReader:
    def __init__(self, path: Path):
        import torch

        self.state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self):
        return self.state.keys()

    def get_tensor(self, name: str):
        return self.state[name]


class _Readers(list):
    def __init__(self, readers: list, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        self.tensor_map = tensor_map or {key: reader for reader in readers for key in reader.keys()}


def _open_checkpoint(model_dir: Path) -> _Readers:
    try:
        import torch  # noqa: F401

        framework = "torch"
    except ImportError:
        framework = "numpy"

    single = model_dir / "model.safetensors"
    if single.exists():
        return _Readers([safe_open(str(single), framework=framework)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        shards = {
            name: safe_open(str(model_dir / name), framework=framework)
            for name in sorted(set(weight_map.values()))
        }
        return _Readers(
            list(shards.values()),
            {key: shards[shard] for key, shard in weight_map.items()},
        )

    pytorch_bin = model_dir / "pytorch_model.bin"
    if pytorch_bin.exists():
        return _Readers([_TorchReader(pytorch_bin)])
    raise FileNotFoundError(f"No ConvBERT checkpoint found in {model_dir}")


def _to_float32(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy() if str(value.dtype) == "torch.float32" else value.float().numpy()
    if value.dtype == np.uint16 or str(value.dtype) == "bfloat16":
        bits = value.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(value, dtype=np.float32)


def _load(readers: _Readers, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_float32(reader.get_tensor(name))


def load_convbert_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Map a ConvBERT checkpoint to the tensors used by the TRT graph."""
    readers = _open_checkpoint(Path(model_dir))
    hidden = config.hidden_size
    embedding_size = int(config.raw.get("embedding_size", hidden))
    type_vocab_size = int(config.raw.get("type_vocab_size", 2))
    head_ratio = int(config.raw.get("head_ratio", 2))
    kernel_size = int(config.raw.get("conv_kernel_size", 9))
    new_num_heads = max(1, config.num_attention_heads // head_ratio)
    head_size = (hidden // new_num_heads) // 2
    all_head_size = new_num_heads * head_size

    if "convbert.embeddings.word_embeddings.weight" in readers.tensor_map:
        root = "convbert."
    elif "embeddings.word_embeddings.weight" in readers.tensor_map:
        root = ""
    else:
        root = "convbert."

    def tensor(name: str) -> np.ndarray:
        return _load(readers, root + name)

    weights = WeightDict(
        {
            "_convbert_new_num_heads": np.array([new_num_heads], dtype=np.int32),
            "_convbert_head_size": np.array([head_size], dtype=np.int32),
            "_convbert_all_head_size": np.array([all_head_size], dtype=np.int32),
            "_convbert_conv_kernel_size": np.array([kernel_size], dtype=np.int32),
            "embedding": tensor("embeddings.word_embeddings.weight"),
            "position_embedding": tensor("embeddings.position_embeddings.weight"),
            "token_type_embedding": tensor("embeddings.token_type_embeddings.weight"),
            "embed_norm": tensor("embeddings.LayerNorm.weight"),
            "embed_norm_beta": tensor("embeddings.LayerNorm.bias"),
        }
    )
    expected_shapes = {
        "embedding": (config.vocab_size, embedding_size),
        "position_embedding": (config.max_position_embeddings, embedding_size),
        "token_type_embedding": (type_vocab_size, embedding_size),
    }
    for key, expected in expected_shapes.items():
        if weights[key].shape != expected:
            raise ValueError(f"{key} shape {weights[key].shape} != {expected}")

    mappings = {
        "w_q": ("attention.self.query.weight", True),
        "w_k": ("attention.self.key.weight", True),
        "w_v": ("attention.self.value.weight", True),
        "q_bias": ("attention.self.query.bias", False),
        "k_bias": ("attention.self.key.bias", False),
        "v_bias": ("attention.self.value.bias", False),
        "sep_conv_dw": ("attention.self.key_conv_attn_layer.depthwise.weight", False),
        "sep_conv_pw": ("attention.self.key_conv_attn_layer.pointwise.weight", False),
        "sep_conv_bias": ("attention.self.key_conv_attn_layer.bias", False),
        "conv_kernel_w": ("attention.self.conv_kernel_layer.weight", True),
        "conv_kernel_bias": ("attention.self.conv_kernel_layer.bias", False),
        "conv_out_w": ("attention.self.conv_out_layer.weight", True),
        "conv_out_bias": ("attention.self.conv_out_layer.bias", False),
        "w_o": ("attention.output.dense.weight", True),
        "o_bias": ("attention.output.dense.bias", False),
        "post_attn_norm": ("attention.output.LayerNorm.weight", False),
        "post_attn_norm_beta": ("attention.output.LayerNorm.bias", False),
        "w_fc1": ("intermediate.dense.weight", True),
        "fc1_bias": ("intermediate.dense.bias", False),
        "w_fc2": ("output.dense.weight", True),
        "fc2_bias": ("output.dense.bias", False),
        "output_norm": ("output.LayerNorm.weight", False),
        "output_norm_beta": ("output.LayerNorm.bias", False),
    }
    for layer_idx in range(config.num_hidden_layers):
        source = f"encoder.layer.{layer_idx}."
        target = f"layer.{layer_idx}."
        for name, (checkpoint_name, transpose) in mappings.items():
            value = tensor(source + checkpoint_name)
            if name == "sep_conv_bias":
                value = value.squeeze(-1)
            if transpose:
                value = value.T
            weights[target + name] = np.ascontiguousarray(value, dtype=np.float32)
    return weights
