# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapped-weight validation for InternVL's Qwen2 native KV graph."""

from __future__ import annotations

from collections.abc import Mapping


def _require(weights: Mapping[str, object], name: str, shape: tuple[int, ...]) -> None:
    if name not in weights:
        raise ValueError(f"missing InternVL native KV weight {name}")
    actual = tuple(int(dim) for dim in weights[name].shape)
    if actual != shape:
        raise ValueError(
            f"InternVL native KV weight {name} must have shape {shape}, got {actual}")


def validate_native_kv_weights(config: object, weights: Mapping[str, object]) -> None:
    if not isinstance(weights, Mapping):
        raise ValueError("InternVL native KV weights must be a mapping")
    hidden = int(getattr(config, "hidden_size"))
    vocab = int(getattr(config, "vocab_size"))
    mlp = int(getattr(config, "intermediate_size"))
    layers = int(getattr(config, "num_hidden_layers"))
    heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = int(getattr(config, "head_dim"))
    attention = heads * head_dim
    kv_attention = kv_heads * head_dim

    for name, expected in (
        ("_attention_size", attention),
        ("_kv_attention_size", kv_attention),
        ("_mlp_size", mlp),
    ):
        if int(weights.get(name, expected)) != expected:
            raise ValueError(f"InternVL native KV metadata {name} must be {expected}")

    _require(weights, "embedding", (vocab, hidden))
    _require(weights, "final_norm", (hidden,))
    _require(weights, "w_out", (hidden, vocab))
    for layer in range(layers):
        prefix = f"layer.{layer}"
        for suffix, shape in (
            ("input_norm", (hidden,)),
            ("w_q", (hidden, attention)),
            ("w_k", (hidden, kv_attention)),
            ("w_v", (hidden, kv_attention)),
            ("w_o", (attention, hidden)),
            ("post_attn_norm", (hidden,)),
            ("w_gate", (hidden, mlp)),
            ("w_up", (hidden, mlp)),
            ("w_down", (mlp, hidden)),
        ):
            _require(weights, f"{prefix}.{suffix}", shape)
        for suffix, shape in (
            ("q_bias", (attention,)),
            ("k_bias", (kv_attention,)),
            ("v_bias", (kv_attention,)),
        ):
            name = f"{prefix}.{suffix}"
            if name in weights:
                _require(weights, name, shape)
