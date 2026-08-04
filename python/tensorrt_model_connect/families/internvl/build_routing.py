# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed build contract for InternVL TensorRT native KV cache."""

from __future__ import annotations

import math


def _positive(config: object, name: str) -> int:
    value = int(getattr(config, name, 0))
    if value <= 0:
        raise ValueError(f"InternVL native KV requires positive {name}")
    return value


def validate_native_kv_architecture(config: object) -> None:
    if str(getattr(config, "model_type", "")).lower() != "internvl":
        raise ValueError("InternVL native KV requires model_type='internvl'")
    raw = getattr(config, "raw", {})
    text = raw.get("text_config", {}) if isinstance(raw, dict) else {}
    if not isinstance(text, dict) or str(text.get("model_type", "")).lower() != "qwen2":
        raise ValueError("InternVL native KV requires a Qwen2 text_config")

    hidden = _positive(config, "hidden_size")
    heads = _positive(config, "num_attention_heads")
    kv_heads = _positive(config, "num_key_value_heads")
    for name in (
        "vocab_size", "intermediate_size", "num_hidden_layers",
        "max_position_embeddings",
    ):
        _positive(config, name)
    head_dim = int(getattr(config, "head_dim", 0))
    if hidden != heads * head_dim or head_dim != 128:
        raise ValueError(
            "InternVL native KV requires hidden_size == num_attention_heads * 128")
    if heads % kv_heads:
        raise ValueError(
            "InternVL native KV requires query heads divisible by KV heads")
    if str(getattr(config, "hidden_act", "")).lower() != "silu":
        raise ValueError("InternVL native KV requires hidden_act='silu'")
    for name in ("rms_norm_eps", "rope_theta"):
        value = float(getattr(config, name, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"InternVL native KV requires positive finite {name}")

    rope = text.get("rope_scaling")
    if rope is not None:
        if not isinstance(rope, dict):
            raise ValueError("InternVL native KV rope_scaling must be an object")
        rope_type = str(rope.get("rope_type", rope.get("type", "default"))).lower()
        if rope_type not in ("default", "dynamic"):
            raise ValueError(
                "InternVL native KV supports only default or dynamic-NTK RoPE")


def validate_native_kv_build(
    config: object,
    *,
    precision: str,
    max_cache_length: int,
    parallel: object,
    quantized: bool,
    debug_layer_outputs: bool,
) -> None:
    validate_native_kv_architecture(config)
    if str(precision).lower() != "bf16":
        raise ValueError("InternVL native KV requires BF16")
    context = int(getattr(config, "max_position_embeddings"))
    if int(max_cache_length) != context:
        raise ValueError(
            "InternVL native KV requires max_cache_length equal to the model "
            f"context ({context})")
    raw = getattr(config, "raw", {})
    if isinstance(raw, dict) and str(raw.get("_decoder_engine_layout", "split")) != "split":
        raise ValueError("InternVL native KV requires split prefill/decode engines")
    if quantized:
        raise ValueError("InternVL native KV does not support quantized builds")
    if debug_layer_outputs:
        raise ValueError("InternVL native KV does not support debug layer outputs")

    if bool(getattr(parallel, "enabled", False)):
        tp_size = int(getattr(parallel, "tp_size", 1))
        if tp_size not in (2, 4, 8):
            raise ValueError("InternVL native KV TP size must be 2, 4, or 8")
        if int(getattr(config, "num_attention_heads")) % tp_size:
            raise ValueError("InternVL query heads must be divisible by TP size")
        if int(getattr(config, "num_key_value_heads")) % tp_size:
            raise ValueError("InternVL KV heads must be divisible by TP size")
