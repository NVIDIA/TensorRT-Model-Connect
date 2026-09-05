# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint readers and tensor container used by the BERT weight mapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Register NumPy bfloat16 support required by safetensors.
import ml_dtypes  # noqa: F401

from safetensors import safe_open

from ..config import ModelConfig


class WeightDict(dict):
    """BERT encoder weights keyed by the local graph builder's logical names."""


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


def _load_layer_norm(readers: _ReaderCollection, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one exact Hugging Face LayerNorm naming schema."""
    schemas = [
        (f"{prefix}.weight", f"{prefix}.bias"),
        (f"{prefix}.gamma", f"{prefix}.beta"),
    ]
    matches = [
        (weight, bias)
        for weight, bias in schemas
        if _has_tensor(readers, weight) and _has_tensor(readers, bias)
    ]
    if len(matches) != 1:
        raise KeyError(f"LayerNorm {prefix!r} must match exactly one naming schema")
    weight_name, bias_name = matches[0]
    weight = _load_tensor(readers, weight_name)
    bias = _load_tensor(readers, bias_name)
    return weight.astype(np.float32), bias.astype(np.float32)


def _detect_bert_prefix(readers: list) -> str:
    if _has_tensor(readers, "bert.embeddings.word_embeddings.weight"):
        return "bert"
    if _has_tensor(readers, "embeddings.word_embeddings.weight"):
        return ""
    return "bert"


def _prefixed(root: str, key: str) -> str:
    return f"{root}.{key}" if root else key


def load_bert_weights(model_dir: str | Path, config: ModelConfig) -> WeightDict:
    """Map Hugging Face BERT tensors to the local encoder graph contract."""
    readers = _open_safetensors(Path(model_dir))
    hidden = config.hidden_size
    root = _detect_bert_prefix(readers)
    type_vocab_size = int(config.raw.get("type_vocab_size", 2))
    weights = WeightDict()

    embedding = _load_tensor(readers, _prefixed(root, "embeddings.word_embeddings.weight"))
    assert embedding.shape == (config.vocab_size, hidden), (
        f"Embedding shape {embedding.shape} != ({config.vocab_size}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    position_embedding = _load_tensor(
        readers, _prefixed(root, "embeddings.position_embeddings.weight")
    )
    assert position_embedding.shape == (config.max_position_embeddings, hidden), (
        f"Position embedding shape {position_embedding.shape} != "
        f"({config.max_position_embeddings}, {hidden})"
    )
    weights["position_embedding"] = position_embedding.astype(np.float32)

    token_type_key = _prefixed(root, "embeddings.token_type_embeddings.weight")
    if _has_tensor(readers, token_type_key):
        token_type_embedding = _load_tensor(readers, token_type_key)
        assert token_type_embedding.shape == (type_vocab_size, hidden), (
            f"Token type embedding shape {token_type_embedding.shape} != "
            f"({type_vocab_size}, {hidden})"
        )
        weights["token_type_embedding"] = token_type_embedding.astype(np.float32)
    else:
        weights["token_type_embedding"] = np.zeros((type_vocab_size, hidden), dtype=np.float32)

    embed_norm, embed_norm_beta = _load_layer_norm(readers, _prefixed(root, "embeddings.LayerNorm"))
    weights["embed_norm"] = embed_norm
    weights["embed_norm_beta"] = embed_norm_beta

    projections = (("q", "query"), ("k", "key"), ("v", "value"))
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = _prefixed(root, f"encoder.layer.{layer_idx}")

        for logical, hf_name in projections:
            tensor = _load_tensor(readers, f"{hf_prefix}.attention.self.{hf_name}.weight")
            weights[f"{prefix}.w_{logical}"] = np.ascontiguousarray(tensor.T.astype(np.float32))
            weights[f"{prefix}.{logical}_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.{hf_name}.bias"
            ).astype(np.float32)

        output_weight = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
        weights[f"{prefix}.w_o"] = np.ascontiguousarray(output_weight.T.astype(np.float32))
        weights[f"{prefix}.o_bias"] = _load_tensor(
            readers, f"{hf_prefix}.attention.output.dense.bias"
        ).astype(np.float32)

        post_attn_norm, post_attn_norm_beta = _load_layer_norm(
            readers, f"{hf_prefix}.attention.output.LayerNorm"
        )
        weights[f"{prefix}.post_attn_norm"] = post_attn_norm
        weights[f"{prefix}.post_attn_norm_beta"] = post_attn_norm_beta

        for logical, hf_name in (("fc1", "intermediate"), ("fc2", "output")):
            tensor = _load_tensor(readers, f"{hf_prefix}.{hf_name}.dense.weight")
            weights[f"{prefix}.w_{logical}"] = np.ascontiguousarray(tensor.T.astype(np.float32))
            weights[f"{prefix}.{logical}_bias"] = _load_tensor(
                readers, f"{hf_prefix}.{hf_name}.dense.bias"
            ).astype(np.float32)

        output_norm, output_norm_beta = _load_layer_norm(readers, f"{hf_prefix}.output.LayerNorm")
        weights[f"{prefix}.output_norm"] = output_norm
        weights[f"{prefix}.output_norm_beta"] = output_norm_beta

    pooler_key = _prefixed(root, "pooler.dense.weight")
    if _has_tensor(readers, pooler_key):
        pooler_weight = _load_tensor(readers, pooler_key)
        weights["pooler_w"] = np.ascontiguousarray(pooler_weight.T.astype(np.float32))
        weights["pooler_bias"] = _load_tensor(readers, _prefixed(root, "pooler.dense.bias")).astype(
            np.float32
        )

    return weights
