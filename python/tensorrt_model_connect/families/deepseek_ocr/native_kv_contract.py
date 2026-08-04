# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint geometry validation for DeepSeek-OCR native KV."""

from __future__ import annotations

import numpy as np

from .build_routing import resolved_head_dim


def _shape(weights: dict, name: str, expected: tuple[int, ...]) -> None:
    value = weights.get(name)
    if not isinstance(value, np.ndarray) or value.shape != expected:
        actual = None if not isinstance(value, np.ndarray) else value.shape
        raise ValueError(
            f"DeepSeek-OCR native KV weight {name!r} has shape {actual}; "
            f"expected {expected}")


def validate_native_kv_weights(config: object, weights: dict) -> None:
    """Validate only tensors whose geometry defines the native attention ABI."""
    hidden = int(getattr(config, "hidden_size"))
    vocab = int(getattr(config, "vocab_size"))
    layers = int(getattr(config, "num_hidden_layers"))
    heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = resolved_head_dim(config)
    attention_size = heads * head_dim
    kv_size = kv_heads * head_dim

    _shape(weights, "embedding", (vocab, hidden))
    _shape(weights, "final_norm", (hidden,))
    _shape(weights, "w_out", (hidden, vocab))
    for layer in range(layers):
        prefix = f"layer.{layer}"
        _shape(weights, f"{prefix}.input_norm", (hidden,))
        _shape(weights, f"{prefix}.post_attn_norm", (hidden,))
        _shape(weights, f"{prefix}.w_q", (hidden, attention_size))
        _shape(weights, f"{prefix}.w_k", (hidden, kv_size))
        _shape(weights, f"{prefix}.w_v", (hidden, kv_size))
        _shape(weights, f"{prefix}.w_o", (attention_size, hidden))
