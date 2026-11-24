# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint tensors of the privacy-filter token classifiers, as float32 graph operands.

Every tensor is upcast exactly from its stored dtype to float32. Projections are
transposed from the ``[out, in]`` layout of ``nn.Linear`` to ``[in, out]``.
The expert tensors keep the Transformers layout (``x @ gate_up_proj[e]``) with
the gate half first and the linear half second along the last axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Registers the NumPy bfloat16 dtype that safetensors returns for BF16 tensors.
import ml_dtypes  # noqa: F401
import numpy as np
from safetensors import safe_open

from .config import ModelConfig


@dataclass(frozen=True)
class LayerWeights:
    input_norm: np.ndarray  # [hidden]
    q_weight: np.ndarray  # [hidden, heads * head_dim]
    q_bias: np.ndarray  # [heads * head_dim]
    k_weight: np.ndarray  # [hidden, kv_heads * head_dim]
    k_bias: np.ndarray
    v_weight: np.ndarray
    v_bias: np.ndarray
    o_weight: np.ndarray  # [heads * head_dim, hidden]
    o_bias: np.ndarray  # [hidden]
    sinks: np.ndarray  # [heads]
    post_attention_norm: np.ndarray  # [hidden]
    router_weight: np.ndarray  # [hidden, experts]
    router_bias: np.ndarray  # [experts]
    gate_up_weight: np.ndarray  # [hidden, experts * 2 * intermediate], expert-major
    gate_up_bias: np.ndarray  # [experts, 2 * intermediate]
    down_weight: np.ndarray  # [experts, intermediate, hidden]
    down_bias: np.ndarray  # [experts, hidden]


@dataclass(frozen=True)
class ModelWeights:
    embedding: np.ndarray  # [vocab, hidden]
    layers: tuple[LayerWeights, ...]
    final_norm: np.ndarray  # [hidden]
    score_weight: np.ndarray  # [hidden, labels]
    score_bias: np.ndarray  # [labels]


class _Checkpoint:
    """Exact tensor-to-file index over one file or a sharded safetensors checkpoint."""

    def __init__(self, model_dir: Path):
        single = model_dir / "model.safetensors"
        index = model_dir / "model.safetensors.index.json"
        if single.is_file():
            reader = safe_open(str(single), framework="numpy")
            self._readers = {name: reader for name in reader.keys()}
        elif index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("model.safetensors.index.json has no weight_map")
            files = {
                shard: safe_open(str(model_dir / shard), framework="numpy")
                for shard in set(weight_map.values())
            }
            self._readers = {name: files[shard] for name, shard in weight_map.items()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint in {model_dir}")
        self._used: set[str] = set()

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        reader = self._readers.get(name)
        if reader is None:
            raise KeyError(f"checkpoint tensor is missing: {name}")
        value = np.asarray(reader.get_tensor(name), dtype=np.float32)
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"checkpoint tensor {name} has shape {tuple(value.shape)}, expected {tuple(shape)}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"checkpoint tensor {name} is not finite")
        self._used.add(name)
        return value

    def unused(self) -> list[str]:
        return sorted(set(self._readers) - self._used)


def _transposed(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.T, dtype=np.float32)


def load_weights(model_dir: Path, config: ModelConfig) -> ModelWeights:
    """Load every tensor of the token classifier; an unexpected or missing tensor is an error."""
    checkpoint = _Checkpoint(Path(model_dir))
    hidden = config.hidden_size
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    experts = config.num_local_experts
    inner = config.intermediate_size

    def bias(name: str, size: int) -> np.ndarray:
        if config.attention_bias:
            return checkpoint.get(name, (size,))
        return np.zeros((size,), dtype=np.float32)

    layers = []
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        attention = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        gate_up = checkpoint.get(f"{mlp}.experts.gate_up_proj", (experts, hidden, 2 * inner))
        layers.append(
            LayerWeights(
                input_norm=checkpoint.get(f"{prefix}.input_layernorm.weight", (hidden,)),
                q_weight=_transposed(
                    checkpoint.get(f"{attention}.q_proj.weight", (q_size, hidden))
                ),
                q_bias=bias(f"{attention}.q_proj.bias", q_size),
                k_weight=_transposed(
                    checkpoint.get(f"{attention}.k_proj.weight", (kv_size, hidden))
                ),
                k_bias=bias(f"{attention}.k_proj.bias", kv_size),
                v_weight=_transposed(
                    checkpoint.get(f"{attention}.v_proj.weight", (kv_size, hidden))
                ),
                v_bias=bias(f"{attention}.v_proj.bias", kv_size),
                o_weight=_transposed(
                    checkpoint.get(f"{attention}.o_proj.weight", (hidden, q_size))
                ),
                o_bias=bias(f"{attention}.o_proj.bias", hidden),
                sinks=checkpoint.get(f"{attention}.sinks", (config.num_attention_heads,)),
                post_attention_norm=checkpoint.get(
                    f"{prefix}.post_attention_layernorm.weight", (hidden,)
                ),
                router_weight=_transposed(
                    checkpoint.get(f"{mlp}.router.weight", (experts, hidden))
                ),
                router_bias=checkpoint.get(f"{mlp}.router.bias", (experts,)),
                # [experts, hidden, 2I] -> [hidden, experts * 2I]: one GEMM evaluates every expert.
                gate_up_weight=np.ascontiguousarray(
                    gate_up.transpose(1, 0, 2).reshape(hidden, experts * 2 * inner),
                    dtype=np.float32,
                ),
                gate_up_bias=checkpoint.get(
                    f"{mlp}.experts.gate_up_proj_bias", (experts, 2 * inner)
                ),
                down_weight=checkpoint.get(f"{mlp}.experts.down_proj", (experts, inner, hidden)),
                down_bias=checkpoint.get(f"{mlp}.experts.down_proj_bias", (experts, hidden)),
            )
        )
        del gate_up
    weights = ModelWeights(
        embedding=checkpoint.get("model.embed_tokens.weight", (config.vocab_size, hidden)),
        layers=tuple(layers),
        final_norm=checkpoint.get("model.norm.weight", (hidden,)),
        score_weight=_transposed(checkpoint.get("score.weight", (config.num_labels, hidden))),
        score_bias=checkpoint.get("score.bias", (config.num_labels,)),
    )
    unused = checkpoint.unused()
    if unused:
        raise ValueError(f"checkpoint has tensors this family does not map: {unused[:8]}")
    return weights
