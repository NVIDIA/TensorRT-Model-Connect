# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing contract for Qwen2/2.5/3-VL TensorRT native KV cache."""

from __future__ import annotations

import math
import operator

_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1


class NativeKvCapability:
    """Loader-safe capability result without importing model dependencies."""

    __slots__ = ("applicable", "eligible", "reason")

    def __init__(self, applicable: bool, eligible: bool, reason: str) -> None:
        self.applicable = applicable
        self.eligible = eligible
        self.reason = reason


def _result(
    *, applicable: bool = True, reasons: list[str] | tuple[str, ...] = (),
) -> NativeKvCapability:
    return NativeKvCapability(
        applicable,
        applicable and not reasons,
        "; ".join(reasons) or "supported",
    )


def _raw(config: object) -> dict:
    value = getattr(config, "raw", {})
    return value if isinstance(value, dict) else {}


def _text_raw(raw: dict) -> dict:
    value = raw.get("text_config")
    return value if isinstance(value, dict) else raw


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _positive(config: object, name: str) -> int:
    value = _integer(getattr(config, name, None), name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT32_MAX:
        raise ValueError(f"{name} exceeds TensorRT's int32 dimension limit")
    return value


def resolved_head_dim(config: object) -> int:
    raw = _raw(config)
    text = _text_raw(raw)
    explicit = text.get(
        "head_dim", raw.get("head_dim", getattr(config, "_head_dim", 0))
    )
    if explicit not in (None, 0):
        head_dim = _integer(explicit, "head_dim")
    else:
        hidden = _positive(config, "hidden_size")
        heads = _positive(config, "num_attention_heads")
        if hidden % heads:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads when "
                "head_dim is absent"
            )
        head_dim = hidden // heads
    if not 0 < head_dim <= _INT32_MAX:
        raise ValueError("head_dim must be a positive TensorRT dimension")
    return head_dim


def _checked_product(label: str, *values: int) -> int:
    product = 1
    for value in values:
        if value <= 0 or product > _UINT64_MAX // value:
            raise ValueError(f"native Qwen-VL KV {label} exceeds uint64")
        product *= value
    return product


def native_kv_cache_geometry(
    config: object,
    capacity: int,
    *,
    element_bytes: int = 2,
    tp_size: int = 1,
) -> tuple[int, int]:
    """Return rank-local row bytes and full-context cache bytes."""

    capacity = _integer(capacity, "max_cache_length")
    context = _positive(config, "max_position_embeddings")
    if capacity != context:
        raise ValueError(
            "native Qwen-VL KV requires max_cache_length == "
            f"max_position_embeddings ({context}), got {capacity}"
        )
    tp_size = _integer(tp_size, "tp_size")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    kv_heads = _positive(config, "num_key_value_heads")
    if kv_heads % tp_size:
        raise ValueError("num_key_value_heads must be divisible by tp_size")
    row_bytes = _checked_product(
        "row size",
        2,
        _positive(config, "num_hidden_layers"),
        kv_heads // tp_size,
        resolved_head_dim(config),
        _integer(element_bytes, "element_bytes"),
    )
    return row_bytes, _checked_product("cache size", capacity, row_bytes)


def _mrope_config(raw: dict) -> tuple[dict, bool]:
    source = _text_raw(raw)
    rope = source.get("rope_scaling")
    if rope is None:
        rope = source.get("rope_parameters")
    return (rope if isinstance(rope, dict) else {}), source is not raw


def native_mrope_settings(config: object) -> tuple[tuple[int, int, int], bool]:
    """Return the official 3-axis mRoPE section and interleaving mode."""

    rope, _ = _mrope_config(_raw(config))
    raw_section = rope.get("mrope_section")
    if not isinstance(raw_section, (list, tuple)) or len(raw_section) != 3:
        raise ValueError("native Qwen-VL requires a three-axis mrope_section")
    try:
        section = tuple(int(value) for value in raw_section)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mrope_section values must be integers") from exc
    if any(value <= 0 for value in section):
        raise ValueError("mrope_section values must be positive")
    if sum(section) != resolved_head_dim(config) // 2:
        raise ValueError("mrope_section must sum to half of head_dim")
    interleaved = bool(rope.get("mrope_interleaved", False))
    return section, interleaved


_ARCHITECTURES = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
}


def native_kv_architecture_capability(config: object) -> NativeKvCapability:
    """Accept every dense official Qwen-VL size with the same graph contract."""

    model_type = str(getattr(config, "model_type", "")).lower()
    expected_architecture = _ARCHITECTURES.get(model_type)
    if expected_architecture is None:
        return _result(applicable=False)

    raw = _raw(config)
    text_raw = _text_raw(raw)
    reasons: list[str] = []
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != (expected_architecture,):
        reasons.append(f"architectures must contain exactly {expected_architecture}")

    try:
        dimensions = {
            name: _positive(config, name)
            for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
            )
        }
        head_dim = resolved_head_dim(config)
        if dimensions["num_attention_heads"] % dimensions["num_key_value_heads"]:
            reasons.append(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if head_dim != 128:
            reasons.append("native Qwen-VL attention requires head_dim=128")
        _, interleaved = native_mrope_settings(config)
        if interleaved != (model_type == "qwen3_vl"):
            reasons.append(
                "Qwen3-VL requires interleaved mRoPE while Qwen2/2.5-VL "
                "requires sectioned mRoPE"
            )
    except ValueError as exc:
        reasons.append(str(exc))

    if str(getattr(config, "hidden_act", "")).lower() != "silu":
        reasons.append("native Qwen-VL requires hidden_act='silu'")
    for name in ("rms_norm_eps", "rope_theta"):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value) or value <= 0:
            reasons.append(f"{name} must be finite and positive")

    if raw.get("is_encoder_decoder"):
        reasons.append("native Qwen-VL requires a decoder-only language model")
    vision = raw.get("vision_config")
    vision = vision if isinstance(vision, dict) else {}
    deepstack = vision.get("deepstack_visual_indexes")
    if model_type == "qwen3_vl":
        if not isinstance(deepstack, (list, tuple)) or not deepstack:
            reasons.append("Qwen3-VL requires DeepStack vision indexes")
    elif deepstack:
        reasons.append("Qwen2/2.5-VL does not use DeepStack vision indexes")
    if text_raw.get("use_sliding_window"):
        reasons.append("native Qwen-VL does not support sliding-window attention")
    try:
        if float(text_raw.get("partial_rotary_factor", 1.0)) != 1.0:
            reasons.append("native Qwen-VL requires full rotary embeddings")
    except (TypeError, ValueError, OverflowError):
        reasons.append("partial_rotary_factor must be numeric")
    for name in (
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ):
        if text_raw.get(name):
            reasons.append(f"native Qwen-VL does not support {name}")

    rope, _ = _mrope_config(raw)
    rope_type = str(rope.get("rope_type", "default")).lower()
    if rope_type not in ("", "default"):
        reasons.append("native Qwen-VL supports only default mRoPE scaling")
    for name in (
        "factor",
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "original_max_position_embeddings",
    ):
        if name in rope:
            reasons.append("native Qwen-VL does not support scaled mRoPE")
            break
    return _result(reasons=reasons)


def native_kv_build_capability(
    config: object,
    *,
    precision: str = "bf16",
    max_cache_length: int | None = None,
    tp_size: int = 1,
    dynamic_kv_cache: bool | None = None,
    quantized: bool | None = None,
    debug_layer_outputs: bool = False,
    lora_enabled: bool = False,
) -> NativeKvCapability:
    architecture = native_kv_architecture_capability(config)
    if not architecture.eligible:
        return architecture

    raw = _raw(config)
    reasons: list[str] = []
    if str(precision).lower() != "bf16":
        reasons.append("native Qwen-VL requires BF16")
    if str(raw.get("_decoder_engine_layout", "split")) != "split":
        reasons.append("native Qwen-VL requires split prefill/decode engines")
    if raw.get("_rtx_build_requested"):
        reasons.append("native Qwen-VL requires the standard TensorRT backend")
    if dynamic_kv_cache or raw.get("_runtime_dynamic_kv_requested"):
        reasons.append("native Qwen-VL uses one fixed physical KV capacity")
    if quantized or raw.get("quantization_config") or raw.get("_quantized_build_requested"):
        reasons.append("native Qwen-VL does not support quantized builds")
    if raw.get("_fp32_layers"):
        reasons.append("native Qwen-VL does not support FP32 layer overrides")
    if debug_layer_outputs:
        reasons.append("native Qwen-VL does not support debug layer outputs")
    if lora_enabled:
        reasons.append("native Qwen-VL does not support dynamic LoRA")
    try:
        tp_size = _integer(tp_size, "tp_size")
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if _positive(config, "num_attention_heads") % tp_size:
            reasons.append("num_attention_heads must be divisible by tp_size")
        if _positive(config, "intermediate_size") % tp_size:
            reasons.append("intermediate_size must be divisible by tp_size")
        native_kv_cache_geometry(
            config,
            (
                int(getattr(config, "max_position_embeddings"))
                if max_cache_length is None
                else max_cache_length
            ),
            tp_size=tp_size,
        )
    except ValueError as exc:
        reasons.append(str(exc))
    return _result(reasons=reasons)


def prefer_native_default(config: object) -> bool:
    """Choose full-context native KV without a user-facing build flag."""

    return native_kv_architecture_capability(config).eligible
