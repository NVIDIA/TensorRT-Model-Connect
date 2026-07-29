# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed routing contract for GLM's TensorRT native KV cache."""

from __future__ import annotations

import math
import operator

_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1


class NativeKvCapability:
    """Small, loader-safe capability result (no dataclass dependency)."""

    __slots__ = ("applicable", "eligible", "reason")

    def __init__(
        self,
        applicable: bool,
        eligible: bool,
        reason: str,
    ) -> None:
        self.applicable = applicable
        self.eligible = eligible
        self.reason = reason


def _result(
    *,
    applicable: bool = True,
    reasons: list[str] | tuple[str, ...] = (),
) -> NativeKvCapability:
    return NativeKvCapability(
        applicable,
        applicable and not reasons,
        "; ".join(reasons) or "supported",
    )


def _raw(config: object) -> dict:
    value = getattr(config, "raw", {})
    return value if isinstance(value, dict) else {}


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
    """Return the explicit HF head width, or derive it when absent."""

    raw = _raw(config)
    explicit = raw.get("head_dim", getattr(config, "_head_dim", 0))
    if "head_dim" in raw or explicit not in (None, 0):
        head_dim = _integer(explicit, "head_dim")
    else:
        hidden = _positive(config, "hidden_size")
        heads = _positive(config, "num_attention_heads")
        if hidden % heads:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads when head_dim is absent"
            )
        head_dim = hidden // heads
    if not 0 < head_dim <= _INT32_MAX:
        raise ValueError("head_dim must be a positive TensorRT dimension")
    return head_dim


def _checked_product(label: str, *values: int) -> int:
    product = 1
    for value in values:
        if value <= 0 or product > _UINT64_MAX // value:
            raise ValueError(f"native GLM KV {label} exceeds uint64")
        product *= value
    return product


def native_kv_cache_geometry(
    config: object,
    capacity: int,
    *,
    element_bytes: int = 2,
) -> tuple[int, int]:
    """Return byte geometry for GLM's required full-context cache."""

    capacity = _integer(capacity, "max_cache_length")
    context = _positive(config, "max_position_embeddings")
    if capacity != context:
        raise ValueError(
            "native GLM KV requires max_cache_length == "
            f"max_position_embeddings ({context}), got {capacity}"
        )
    row_bytes = _checked_product(
        "row size",
        2,
        _positive(config, "num_hidden_layers"),
        _positive(config, "num_key_value_heads"),
        resolved_head_dim(config),
        _integer(element_bytes, "element_bytes"),
    )
    return row_bytes, _checked_product("cache size", capacity, row_bytes)


def _enabled(value: object) -> bool:
    return value not in (None, False, 0, "", (), [], {})


def _rope_config(raw: dict) -> dict | None:
    parameters = raw.get("rope_parameters")
    scaling = raw.get("rope_scaling")
    if parameters is not None and scaling is not None:
        raise ValueError("RoPE configuration is ambiguous")
    rope = parameters if parameters is not None else scaling
    if rope is None:
        return None
    if not isinstance(rope, dict):
        raise ValueError("RoPE configuration must be an object")
    return rope


def resolved_partial_rotary_factor(config: object) -> float:
    """Resolve HF GLM's backward-compatible partial RoPE default."""

    raw = _raw(config)
    value = raw.get("partial_rotary_factor")
    rope = _rope_config(raw)
    if value is None and rope is not None:
        value = rope.get("partial_rotary_factor")
    if value is None:
        value = 0.5
    try:
        factor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("partial_rotary_factor must be numeric") from exc
    if not math.isfinite(factor):
        raise ValueError("partial_rotary_factor must be finite")
    return factor


def _validate_default_rope(raw: dict, reasons: list[str]) -> None:
    try:
        rope = _rope_config(raw)
    except ValueError as exc:
        reasons.append(str(exc))
        return
    if rope is None:
        return
    rope_type = str(rope.get("rope_type", rope.get("type", "default"))).lower()
    if rope_type not in ("", "default") or any(
        key in rope
        for key in (
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "factor",
            "original_max_position_embeddings",
        )
    ):
        reasons.append("native GLM supports only unscaled default RoPE")


def native_kv_architecture_capability(
    config: object,
) -> NativeKvCapability:
    """Accept any model size retaining the dense HF GLM graph contract."""

    if str(getattr(config, "model_type", "")).lower() != "glm":
        return _result(applicable=False)

    raw = _raw(config)
    reasons: list[str] = []
    if tuple(getattr(config, "architectures", ()) or ()) != ("GlmForCausalLM",):
        reasons.append("architectures must contain exactly GlmForCausalLM")

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
            reasons.append("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim != 128:
            reasons.append("native GLM attention requires head_dim=128")
    except ValueError as exc:
        reasons.append(str(exc))

    if str(getattr(config, "hidden_act", "")).lower() != "silu":
        reasons.append("native GLM requires hidden_act='silu'")
    for name in ("rms_norm_eps", "rope_theta"):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value) or value <= 0:
            reasons.append(f"{name} must be finite and positive")

    if raw.get("attention_bias") is not True:
        reasons.append("native GLM requires Q/K/V attention biases")
    unsupported_flags = (
        "mlp_bias",
        "is_encoder_decoder",
        "use_sliding_window",
        "sliding_window",
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "full_attention_interval",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
    )
    enabled = [name for name in unsupported_flags if _enabled(raw.get(name))]
    if enabled:
        reasons.append("unsupported GLM fields: " + ", ".join(enabled))

    for name in ("rope_interleaved", "interleaved_rope"):
        if name in raw and raw[name] is not True:
            reasons.append(f"native GLM requires {name}=true when specified")
    try:
        if resolved_partial_rotary_factor(config) != 0.5:
            reasons.append("native GLM requires partial_rotary_factor=0.5")
    except ValueError as exc:
        reasons.append(str(exc))
    _validate_default_rope(raw, reasons)
    return _result(reasons=reasons)


def native_kv_build_capability(
    config: object,
    *,
    precision: str = "bf16",
    max_cache_length: int | None = None,
    parallel_enabled: bool | None = None,
    dynamic_kv_cache: bool | None = None,
    quantized: bool | None = None,
    debug_layer_outputs: bool = False,
) -> NativeKvCapability:
    """Apply deployment constraints once, after architecture routing."""

    architecture = native_kv_architecture_capability(config)
    if not architecture.eligible:
        return architecture

    raw = _raw(config)
    reasons: list[str] = []
    if str(precision).lower() != "bf16":
        reasons.append("native GLM requires BF16")
    if str(raw.get("_decoder_engine_layout", "split")) != "split":
        reasons.append("native GLM requires split prefill/decode engines")
    if raw.get("_rtx_build_requested"):
        reasons.append("native GLM requires the standard TensorRT backend")
    if parallel_enabled or raw.get("_parallel_build_enabled"):
        reasons.append("native GLM does not support tensor parallel builds")
    if dynamic_kv_cache or raw.get("_runtime_dynamic_kv_requested") or raw.get("dynamic_kv_cache"):
        reasons.append("native GLM uses one fixed physical KV capacity")
    if quantized or raw.get("quantization_config") or raw.get("_quantized_build_requested"):
        reasons.append("native GLM does not support quantized builds")
    if raw.get("_fp32_layers"):
        reasons.append("native GLM does not support FP32 layer overrides")
    if debug_layer_outputs:
        reasons.append("native GLM does not support debug layer outputs")
    try:
        native_kv_cache_geometry(
            config,
            (
                int(getattr(config, "max_position_embeddings"))
                if max_cache_length is None
                else max_cache_length
            ),
        )
    except ValueError as exc:
        reasons.append(str(exc))
    return _result(reasons=reasons)


def prefer_native_default(config: object) -> bool:
    """Keep every ``glm`` build on the fail-closed native family path."""

    return native_kv_architecture_capability(config).applicable
