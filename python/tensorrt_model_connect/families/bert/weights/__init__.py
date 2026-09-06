# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint readers and tensor container used by the BERT weight mapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Register bfloat16 dtype with numpy (needed for safetensors without torch).
try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open

from ..config import ModelConfig


class WeightDict(dict):
    """BERT encoder weights keyed by the local graph builder's logical names."""


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
    """Open a standard Hugging Face BERT checkpoint."""
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

    # Older BERT repositories may predate safetensors.
    bin_single = model_dir / "pytorch_model.bin"
    if bin_single.exists():
        return _ReaderCollection([_TorchBinReader(bin_single)])

    raise FileNotFoundError(
        f"No BERT model.safetensors, index, or pytorch_model.bin in {model_dir}"
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


def _load_layer_norm(readers: list, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load modern weight/bias or legacy gamma/beta LayerNorm tensors."""
    if _has_tensor(readers, f"{prefix}.weight"):
        weight = _load_tensor(readers, f"{prefix}.weight")
        bias = _load_tensor(readers, f"{prefix}.bias")
    else:
        weight = _load_tensor(readers, f"{prefix}.gamma")
        bias = _load_tensor(readers, f"{prefix}.beta")
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
    if embedding.shape != (config.vocab_size, hidden):
        raise ValueError(f'Embedding shape {embedding.shape} != ({config.vocab_size}, {hidden})')
    weights["embedding"] = embedding.astype(np.float32)

    position_embedding = _load_tensor(
        readers, _prefixed(root, "embeddings.position_embeddings.weight")
    )
    if position_embedding.shape != (config.max_position_embeddings, hidden):
        raise ValueError(f'Position embedding shape {position_embedding.shape} != ({config.max_position_embeddings}, {hidden})')
    weights["position_embedding"] = position_embedding.astype(np.float32)

    token_type_key = _prefixed(root, "embeddings.token_type_embeddings.weight")
    if _has_tensor(readers, token_type_key):
        token_type_embedding = _load_tensor(readers, token_type_key)
        if token_type_embedding.shape != (type_vocab_size, hidden):
            raise ValueError(f'Token type embedding shape {token_type_embedding.shape} != ({type_vocab_size}, {hidden})')
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
