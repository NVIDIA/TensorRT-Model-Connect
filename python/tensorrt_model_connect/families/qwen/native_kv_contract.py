# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapped-weight contract for Qwen3's TensorRT native KV graph."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .build_routing import resolved_head_dim

_LAYER_KEY = re.compile(r"^layer\.(\d+)\.")
_BIAS_SUFFIXES = (
    "input_norm_beta",
    "q_bias",
    "k_bias",
    "v_bias",
    "o_bias",
    "post_attn_norm_beta",
    "gate_bias",
    "up_bias",
    "down_bias",
)


def _shape(value: object, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value.shape)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"native Qwen3 weight {name} has an invalid shape"
        ) from exc


def _require(
    weights: Mapping[str, object],
    name: str,
    expected: tuple[int, ...],
) -> None:
    if name not in weights:
        raise ValueError(f"missing native Qwen3 weight {name}")
    actual = _shape(weights[name], name)
    if actual != expected:
        raise ValueError(
            f"native Qwen3 weight {name} must have shape "
            f"{expected}, got {actual}"
        )


def validate_native_kv_weights(
    config: object,
    weights: Mapping[str, object],
) -> None:
    """Validate mapped tensors once, before TensorRT graph construction."""

    if not isinstance(weights, Mapping):
        raise ValueError("native Qwen3 weights must be a mapping")

    hidden = int(getattr(config, "hidden_size"))
    vocab = int(getattr(config, "vocab_size"))
    mlp = int(getattr(config, "intermediate_size"))
    layers = int(getattr(config, "num_hidden_layers"))
    heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = resolved_head_dim(config)
    attention = heads * head_dim
    kv_attention = kv_heads * head_dim

    layer_indices: set[int] = set()
    malformed: list[str] = []
    for name in weights:
        if not isinstance(name, str) or not name.startswith("layer."):
            continue
        match = _LAYER_KEY.match(name)
        if match is None:
            malformed.append(name)
        else:
            layer_indices.add(int(match.group(1)))
    if malformed or layer_indices != set(range(layers)):
        raise ValueError(
            "native Qwen3 weights require continuous layer indices; "
            f"malformed={sorted(malformed)}, found={sorted(layer_indices)}"
        )

    for name, expected in (
        ("_attention_size", attention),
        ("_kv_attention_size", kv_attention),
        ("_mlp_size", mlp),
    ):
        if name in weights and int(weights[name]) != expected:
            raise ValueError(
                f"native Qwen3 metadata {name} must be {expected}"
            )

    _require(weights, "embedding", (vocab, hidden))
    _require(weights, "final_norm", (hidden,))
    _require(weights, "w_out", (hidden, vocab))
    forbidden = [
        name
        for name in ("final_norm_beta", "lm_head_bias")
        if name in weights
    ]

    for layer in range(layers):
        prefix = f"layer.{layer}"
        for suffix, expected in (
            ("input_norm", (hidden,)),
            ("w_q", (hidden, attention)),
            ("w_k", (hidden, kv_attention)),
            ("w_v", (hidden, kv_attention)),
            ("w_o", (attention, hidden)),
            ("post_attn_norm", (hidden,)),
            ("w_gate", (hidden, mlp)),
            ("w_up", (hidden, mlp)),
            ("w_down", (mlp, hidden)),
            ("q_norm", (attention,)),
            ("k_norm", (kv_attention,)),
        ):
            _require(weights, f"{prefix}.{suffix}", expected)
        forbidden.extend(
            f"{prefix}.{suffix}"
            for suffix in _BIAS_SUFFIXES
            if f"{prefix}.{suffix}" in weights
        )

    if forbidden:
        raise ValueError(
            "native dense Qwen3 does not support bias weights: "
            + ", ".join(sorted(forbidden))
        )
