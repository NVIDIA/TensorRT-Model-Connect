# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed routing contract for GPT-NeoX's TensorRT native KV path."""

from __future__ import annotations

import math
import operator

_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1


class NativeKvCapability:
    """Loader-safe capability result without importing TensorRT."""

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
                "hidden_size must be divisible by num_attention_heads when "
                "head_dim is absent"
            )
        head_dim = hidden // heads
    if not 0 < head_dim <= _INT32_MAX:
        raise ValueError("head_dim must be a positive TensorRT dimension")
    return head_dim


def resolved_rotary_dim(config: object) -> int:
    """Return the partial-RoPE width used by GPT-NeoX."""

    head_dim = resolved_head_dim(config)
    try:
        rotary_pct = float(_raw(config).get("rotary_pct", 0.25))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rotary_pct must be numeric") from exc
    if not math.isfinite(rotary_pct) or not 0.0 < rotary_pct <= 1.0:
        raise ValueError("rotary_pct must be finite and in (0, 1]")
    rotary_dim = int(head_dim * rotary_pct)
    if rotary_dim < 2 or rotary_dim % 2:
        raise ValueError(
            "rotary_pct must produce an even rotary dimension of at least 2"
        )
    return rotary_dim


def _checked_product(label: str, *values: int) -> int:
    product = 1
    for value in values:
        if value <= 0 or product > _UINT64_MAX // value:
            raise ValueError(f"native GPT-NeoX KV {label} exceeds uint64")
        product *= value
    return product


def native_kv_cache_geometry(
    config: object,
    capacity: int,
    *,
    element_bytes: int = 2,
) -> tuple[int, int]:
    """Return byte geometry for the required full-context FP16 cache."""

    capacity = _integer(capacity, "max_cache_length")
    context = _positive(config, "max_position_embeddings")
    if capacity != context:
        raise ValueError(
            "native GPT-NeoX KV requires max_cache_length == "
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


def native_kv_architecture_capability(
    config: object,
) -> NativeKvCapability:
    """Accept GPT-NeoX/Pythia sizes that retain the qualified dense graph."""

    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type not in ("gpt_neox", "gptneox"):
        return _result(applicable=False)
    if model_type != "gpt_neox":
        return _result(reasons=["model_type must be exactly 'gpt_neox'"])

    raw = _raw(config)
    reasons: list[str] = []
    if tuple(getattr(config, "architectures", ()) or ()) != (
        "GPTNeoXForCausalLM",
    ):
        reasons.append(
            "architectures must contain exactly GPTNeoXForCausalLM"
        )

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
        if dimensions["hidden_size"] != (
            dimensions["num_attention_heads"] * head_dim
        ):
            reasons.append(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if dimensions["num_key_value_heads"] != dimensions[
            "num_attention_heads"
        ]:
            reasons.append(
                "native GPT-NeoX currently requires MHA "
                "(num_key_value_heads == num_attention_heads)"
            )
        if head_dim % 8 or head_dim > 128:
            reasons.append(
                "native GPT-NeoX attention requires head_dim to be an "
                "integer multiple of 8 no larger than 128"
            )
        resolved_rotary_dim(config)
    except ValueError as exc:
        reasons.append(str(exc))

    if str(getattr(config, "hidden_act", "")).lower() not in (
        "gelu",
        "gelu_new",
    ):
        reasons.append(
            "native GPT-NeoX requires hidden_act='gelu' or 'gelu_new'"
        )
    for name in ("rms_norm_eps", "rope_theta"):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value) or value <= 0:
            reasons.append(f"{name} must be finite and positive")

    if not isinstance(raw.get("use_parallel_residual", True), bool):
        reasons.append("use_parallel_residual must be boolean")
    unsupported_flags = (
        "is_encoder_decoder",
        "rope_scaling",
        "rope_parameters",
        "sliding_window",
        "use_sliding_window",
        "interleaved_rope",
        "rope_interleaved",
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
    )
    enabled = [name for name in unsupported_flags if _enabled(raw.get(name))]
    if enabled:
        reasons.append(
            "unsupported GPT-NeoX fields: " + ", ".join(enabled)
        )
    if bool(getattr(config, "tie_word_embeddings", False)):
        reasons.append(
            "native GPT-NeoX requires an explicit untied output projection"
        )
    return _result(reasons=reasons)


def native_kv_build_capability(
    config: object,
    *,
    precision: str = "fp16",
    max_cache_length: int | None = None,
    parallel_enabled: bool | None = None,
    dynamic_kv_cache: bool | None = None,
    quantized: bool | None = None,
    debug_layer_outputs: bool = False,
) -> NativeKvCapability:
    """Apply native-only deployment constraints after architecture routing."""

    architecture = native_kv_architecture_capability(config)
    if not architecture.eligible:
        return architecture

    raw = _raw(config)
    reasons: list[str] = []
    if str(precision).lower() != "fp16":
        reasons.append("native GPT-NeoX requires FP16")
    if str(raw.get("_decoder_engine_layout", "split")) != "split":
        reasons.append(
            "native GPT-NeoX requires split prefill/decode engines"
        )
    if raw.get("_rtx_build_requested"):
        reasons.append(
            "native GPT-NeoX requires the standard TensorRT backend"
        )
    if parallel_enabled or raw.get("_parallel_build_enabled"):
        reasons.append(
            "native GPT-NeoX does not support tensor parallel builds"
        )
    if (
        dynamic_kv_cache
        or raw.get("_runtime_dynamic_kv_requested")
        or raw.get("dynamic_kv_cache")
    ):
        reasons.append(
            "native GPT-NeoX uses one fixed physical KV capacity"
        )
    if (
        quantized
        or raw.get("quantization_config")
        or raw.get("_quantized_build_requested")
    ):
        reasons.append("native GPT-NeoX does not support quantized builds")
    if raw.get("_fp32_layers"):
        reasons.append(
            "native GPT-NeoX does not support FP32 layer overrides"
        )
    if debug_layer_outputs:
        reasons.append(
            "native GPT-NeoX does not support debug layer outputs"
        )
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
    """GPT-NeoX production routing has no legacy KV fallback."""

    return native_kv_architecture_capability(config).applicable
