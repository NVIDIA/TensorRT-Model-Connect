# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build contract for DeepSeek-OCR TensorRT native KV cache engines."""

from __future__ import annotations

import math

from ...parallel_config import ParallelConfig


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result > (1 << 31) - 1:
        raise ValueError(f"{name} must be a positive TensorRT int32 dimension")
    return result


def resolved_head_dim(config: object) -> int:
    head_dim = _positive(getattr(config, "head_dim", 0), "head_dim")
    hidden = _positive(getattr(config, "hidden_size", 0), "hidden_size")
    heads = _positive(
        getattr(config, "num_attention_heads", 0), "num_attention_heads")
    if hidden != heads * head_dim:
        raise ValueError(
            "DeepSeek-OCR native attention requires "
            "hidden_size == num_attention_heads * head_dim")
    return head_dim


def native_kv_cache_bytes(
    config: object,
    capacity: int,
    *,
    tp_size: int = 1,
    element_bytes: int = 2,
) -> int:
    """Return the full rank-local K+V allocation size."""
    capacity = _positive(capacity, "max_cache_length")
    context = _positive(
        getattr(config, "max_position_embeddings", 0),
        "max_position_embeddings",
    )
    if capacity != context:
        raise ValueError(
            "DeepSeek-OCR native KV requires max_cache_length == "
            f"max_position_embeddings ({context}), got {capacity}")
    tp_size = _positive(tp_size, "tp_size")
    kv_heads = _positive(
        getattr(config, "num_key_value_heads", 0), "num_key_value_heads")
    if kv_heads % tp_size:
        raise ValueError(
            "num_key_value_heads must be divisible by tensor parallel size")
    values = (
        2,
        _positive(getattr(config, "num_hidden_layers", 0), "num_hidden_layers"),
        capacity,
        kv_heads // tp_size,
        resolved_head_dim(config),
        _positive(element_bytes, "element_bytes"),
    )
    result = 1
    for value in values:
        result *= value
        if result > (1 << 64) - 1:
            raise ValueError("DeepSeek-OCR native KV byte size exceeds uint64")
    return result


def validate_native_kv_build(
    config: object,
    *,
    precision: str,
    max_cache_length: int,
    parallel: ParallelConfig,
    quantized: bool,
    debug_layer_outputs: bool,
) -> None:
    """Fail closed unless the complete native-KV contract is satisfied."""
    if str(getattr(config, "model_type", "")).lower() != "deepseek_vl_v2":
        raise ValueError("DeepSeek-OCR native KV requires model_type='deepseek_vl_v2'")
    raw = getattr(config, "raw", {})
    raw = raw if isinstance(raw, dict) else {}
    language = raw.get("language_config", {})
    language = language if isinstance(language, dict) else {}

    if str(precision).lower() != "bf16":
        raise ValueError("DeepSeek-OCR native KV requires BF16")
    if str(raw.get("_decoder_engine_layout", "split")) != "split":
        raise ValueError(
            "DeepSeek-OCR native KV requires split prefill/decode engines")
    if raw.get("_rtx_build_requested"):
        raise ValueError(
            "DeepSeek-OCR native KV requires the standard TensorRT backend")
    if raw.get("_runtime_dynamic_kv_requested") or raw.get("dynamic_kv_cache"):
        raise ValueError(
            "DeepSeek-OCR native KV uses one fixed full-context capacity")
    if quantized or raw.get("quantization_config"):
        raise ValueError("DeepSeek-OCR native KV does not support quantized builds")
    if debug_layer_outputs:
        raise ValueError("DeepSeek-OCR native KV does not support debug layer outputs")
    if raw.get("_fp32_layers"):
        raise ValueError("DeepSeek-OCR native KV does not support FP32 layer overrides")
    if bool(language.get("use_mla", raw.get("use_mla", False))):
        raise ValueError("DeepSeek-OCR native KV requires standard Q/K/V attention")

    head_dim = resolved_head_dim(config)
    if head_dim != 128:
        raise ValueError("DeepSeek-OCR native attention requires head_dim=128")
    heads = _positive(
        getattr(config, "num_attention_heads", 0), "num_attention_heads")
    kv_heads = _positive(
        getattr(config, "num_key_value_heads", 0), "num_key_value_heads")
    if heads % kv_heads:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads")
    for name in ("rms_norm_eps", "rope_theta"):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    if parallel.enabled:
        if heads % parallel.tp_size or kv_heads % parallel.tp_size:
            raise ValueError(
                "DeepSeek-OCR attention heads must be divisible by tp_size")
    native_kv_cache_bytes(
        config,
        max_cache_length,
        tp_size=parallel.tp_size if parallel.enabled else 1,
    )
