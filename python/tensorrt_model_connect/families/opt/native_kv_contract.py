# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapped-weight contract for OPT's TensorRT native KV graph."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .build_routing import resolved_head_dim

_LAYER_KEY = re.compile(r"^layer\.(\d+)\.")


def _shape(value: object, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value.shape)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"native OPT weight {name} has an invalid shape") from exc


def _require(
    weights: Mapping[str, object],
    name: str,
    expected: tuple[int, ...],
) -> None:
    if name not in weights:
        raise ValueError(f"missing native OPT weight {name}")
    actual = _shape(weights[name], name)
    if actual != expected:
        raise ValueError(f"native OPT weight {name} must have shape {expected}, got {actual}")


def validate_native_kv_weights(
    config: object,
    weights: Mapping[str, object],
) -> None:
    """Validate mapped tensors once, before TensorRT graph construction."""

    if not isinstance(weights, Mapping):
        raise ValueError("native OPT weights must be a mapping")

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
            "native OPT weights require continuous layer indices; "
            f"malformed={sorted(malformed)}, found={sorted(layer_indices)}"
        )

    for name, expected in (
        ("_attention_size", attention),
        ("_kv_attention_size", kv_attention),
        ("_mlp_size", mlp),
    ):
        if name in weights and int(weights[name]) != expected:
            raise ValueError(f"native OPT metadata {name} must be {expected}")

    _require(weights, "embedding", (vocab, hidden))
    _require(
        weights,
        "position_embedding",
        (int(getattr(config, "max_position_embeddings")), hidden),
    )
    _require(weights, "final_norm", (hidden,))
    _require(weights, "final_norm_beta", (hidden,))
    _require(weights, "w_out", (hidden, vocab))

    for layer in range(layers):
        prefix = f"layer.{layer}"
        for suffix, expected in (
            ("input_norm", (hidden,)),
            ("input_norm_beta", (hidden,)),
            ("w_q", (hidden, attention)),
            ("w_k", (hidden, kv_attention)),
            ("w_v", (hidden, kv_attention)),
            ("q_bias", (attention,)),
            ("k_bias", (kv_attention,)),
            ("v_bias", (kv_attention,)),
            ("w_o", (attention, hidden)),
            ("o_bias", (hidden,)),
            ("post_attn_norm", (hidden,)),
            ("post_attn_norm_beta", (hidden,)),
            ("w_fc1", (hidden, mlp)),
            ("fc1_bias", (mlp,)),
            ("w_fc2", (mlp, hidden)),
            ("fc2_bias", (hidden,)),
        ):
            _require(weights, f"{prefix}.{suffix}", expected)

    forbidden = sorted(
        name
        for name in weights
        if isinstance(name, str)
        and (
            name.endswith(".q_norm")
            or name.endswith(".k_norm")
            or name.endswith(".w_gate")
            or name.endswith(".w_up")
            or name.endswith(".w_down")
            or name == "lm_head_bias"
        )
    )
    if forbidden:
        raise ValueError(
            "native OPT received unsupported mapped weights: " + ", ".join(forbidden)
        )
