# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint topology and exact supported mathematical contract."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    max_position_embeddings: int
    layer_types: tuple[str, ...]
    no_rope_layer_indices: tuple[int, ...]
    mlp_only_layers: tuple[int, ...]
    eos_token_id: tuple[int, ...]

    @classmethod
    def load(cls, directory: Path):
        raw = json.loads((directory / "config.json").read_text())
        required = {
            "model_type": "flint",
            "hidden_act": "silu",
            "attention_bias": False,
            "router_score_function": "sigmoid",
            "softmax_type": "learnable",
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
        }
        for key, expected in required.items():
            if raw.get(key) != expected:
                raise ValueError(f"unsupported Flint {key}: {raw.get(key)!r}")
        values = {key: raw[key] for key in cls.__dataclass_fields__}
        eos = values["eos_token_id"]
        if not isinstance(eos, list) or not eos or any(
            type(token) is not int or not 0 <= token < values["vocab_size"] for token in eos
        ):
            raise ValueError("Flint eos_token_id must be a non-empty list of vocabulary IDs")
        for key in ("layer_types", "no_rope_layer_indices", "mlp_only_layers", "eos_token_id"):
            values[key] = tuple(values[key])
        result = cls(**values)
        if len(result.layer_types) != result.num_hidden_layers:
            raise ValueError("Flint layer_types must describe every layer")
        if result.num_attention_heads % result.num_key_value_heads:
            raise ValueError("Flint query heads must be divisible by KV heads")
        if not 0 < result.num_experts_per_tok <= result.num_experts:
            raise ValueError("invalid Flint expert selection count")
        if "sliding_attention" in result.layer_types and (
            type(result.sliding_window) is not int or result.sliding_window < 2
        ):
            raise ValueError("Flint sliding_window must be at least two for sliding attention")
        for i, kind in enumerate(result.layer_types):
            if kind not in {"full_attention", "sliding_attention"}:
                raise ValueError(f"unsupported Flint attention kind: {kind}")
            if (kind == "full_attention") != (i in result.no_rope_layer_indices):
                raise ValueError("expected Flint full-attention layers without RoPE")
        return result
