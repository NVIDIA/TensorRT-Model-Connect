# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapped-weight contract for Qwen-VL native KV graphs."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .build_routing import resolved_head_dim

_LAYER_KEY = re.compile(r"^layer\.(\d+)\.")


def _shape(value: object, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value.shape)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"native Qwen-VL weight {name} has an invalid shape") from exc


def _require(
    weights: Mapping[str, object], name: str, expected: tuple[int, ...],
) -> None:
    if name not in weights:
        raise ValueError(f"missing native Qwen-VL weight {name}")
    actual = _shape(weights[name], name)
    if actual != expected:
        raise ValueError(
            f"native Qwen-VL weight {name} must have shape {expected}, got {actual}"
        )


def _optional(
    weights: Mapping[str, object], name: str, expected: tuple[int, ...],
) -> None:
    if name in weights:
        _require(weights, name, expected)


def validate_native_kv_weights(
    config: object, weights: Mapping[str, object],
) -> None:
    """Validate generic Qwen2/2.5/3-VL mapped tensors before TRT build."""

    if not isinstance(weights, Mapping):
        raise ValueError("native Qwen-VL weights must be a mapping")

    hidden = int(getattr(config, "hidden_size"))
    vocab = int(getattr(config, "vocab_size"))
    mlp = int(getattr(config, "intermediate_size"))
    layers = int(getattr(config, "num_hidden_layers"))
    heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = resolved_head_dim(config)
    attention = heads * head_dim
    kv_attention = kv_heads * head_dim

    layer_indices = {
        int(match.group(1))
        for name in weights
        if isinstance(name, str)
        for match in [_LAYER_KEY.match(name)]
        if match is not None
    }
    if layer_indices != set(range(layers)):
        raise ValueError(
            "native Qwen-VL weights require continuous layer indices; "
            f"found={sorted(layer_indices)}"
        )

    for name, expected in (
        ("_attention_size", attention),
        ("_kv_attention_size", kv_attention),
        ("_mlp_size", mlp),
    ):
        if name in weights and int(weights[name]) != expected:
            raise ValueError(f"native Qwen-VL metadata {name} must be {expected}")

    _require(weights, "embedding", (vocab, hidden))
    _require(weights, "final_norm", (hidden,))
    _require(weights, "w_out", (hidden, vocab))

    qwen3 = str(getattr(config, "model_type", "")).lower() == "qwen3_vl"
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
        ):
            _require(weights, f"{prefix}.{suffix}", expected)
        for suffix, expected in (
            ("q_bias", (attention,)),
            ("k_bias", (kv_attention,)),
            ("v_bias", (kv_attention,)),
            ("o_bias", (hidden,)),
        ):
            _optional(weights, f"{prefix}.{suffix}", expected)
        if qwen3:
            _require(weights, f"{prefix}.q_norm", (attention,))
            _require(weights, f"{prefix}.k_norm", (kv_attention,))
        else:
            _optional(weights, f"{prefix}.q_norm", (attention,))
            _optional(weights, f"{prefix}.k_norm", (kv_attention,))
