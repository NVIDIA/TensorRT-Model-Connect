# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the topology implemented by this family before reading weights."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window: int
    layer_types: tuple[str, ...]
    layer_norm_eps: float
    rope_theta: float
    logit_scale: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool


def parse_config(raw: dict) -> Config:
    if raw.get("model_type") != "cohere2" or raw.get("architectures") != ["Cohere2ForCausalLM"]:
        raise ValueError("cohere2 requires Cohere2ForCausalLM")
    for name, expected in {
        "attention_bias": False,
        "hidden_act": "silu",
        "use_qk_norm": False,
        "use_parallel_block": True,
        "use_gated_activation": True,
        "use_parallel_embedding": False,
        "position_embedding_type": "rope_gptj",
        "rotary_pct": 1.0,
    }.items():
        if name in raw and raw[name] != expected:
            raise ValueError(f"cohere2 does not implement {name}={raw[name]!r}")
    if raw.get("rope_scaling") is not None:
        raise ValueError("cohere2 does not implement scaled RoPE")
    values = {}
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "sliding_window",
    ):
        value = raw.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"cohere2 {name} must be a positive integer")
        values[name] = value
    if values["hidden_size"] != values["head_dim"] * values["num_attention_heads"]:
        raise ValueError("cohere2 attention width must equal hidden_size")
    if values["num_attention_heads"] % values["num_key_value_heads"] or values["head_dim"] % 2:
        raise ValueError("cohere2 requires integral grouped-query heads and even rotary dimensions")
    layers = raw.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != values["num_hidden_layers"]
        or any(item not in ("sliding_attention", "full_attention") for item in layers)
    ):
        raise ValueError("cohere2 layer_types must describe every decoder layer")
    values["layer_types"] = tuple(layers)
    for name, default in (
        ("layer_norm_eps", 1e-5),
        ("rope_theta", 10000.0),
        ("logit_scale", 0.0625),
    ):
        value = raw.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"cohere2 {name} must be finite and positive")
        values[name] = float(value)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = raw.get(name)
        if type(value) is not int or not 0 <= value < values["vocab_size"]:
            raise ValueError(f"cohere2 {name} must be inside the vocabulary")
        values[name] = value
    tied = raw.get("tie_word_embeddings", raw.get("use_embedding_sharing", True))
    if type(tied) is not bool:
        raise ValueError("cohere2 tie_word_embeddings must be boolean")
    if "use_embedding_sharing" in raw and raw["use_embedding_sharing"] != tied:
        raise ValueError("cohere2 embedding-sharing settings disagree")
    values["tie_word_embeddings"] = tied
    return Config(**values)
