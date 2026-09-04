# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing contract for dense SmolLM3 models using TensorRT native KV cache."""

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
            raise ValueError(f"native SmolLM3 KV {label} exceeds uint64")
        product *= value
    return product


def native_kv_cache_geometry(
    config: object,
    capacity: int,
    *,
    element_bytes: int = 2,
) -> tuple[int, int]:
    """Return runtime byte geometry for the required full-context cache."""

    capacity = _integer(capacity, "max_cache_length")
    context = _positive(config, "max_position_embeddings")
    if capacity != context:
        raise ValueError(
            "native SmolLM3 KV requires max_cache_length == "
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


def _validate_rope(raw: dict, reasons: list[str]) -> None:
    parameters = raw.get("rope_parameters")
    scaling = raw.get("rope_scaling")
    if parameters is not None and scaling is not None:
        reasons.append("RoPE configuration is ambiguous")
        return
    rope = parameters if parameters is not None else scaling
    if rope is None:
        return
    if not isinstance(rope, dict):
        reasons.append("RoPE configuration must be an object")
        return
    rope_type = str(rope.get("rope_type", rope.get("type", "default"))).lower()
    if rope_type in ("", "default"):
        if any(
            key in rope
            for key in (
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            )
        ):
            reasons.append("default RoPE must not contain scaling parameters")
        return
    if rope_type != "llama3":
        # The family still builds these: they route to the standard decoder,
        # which applies the scaling through an indexed cos/sin table. Only the
        # native KV graph is limited to unscaled and llama3 RoPE.
        reasons.append(
            f"native SmolLM3 KV does not support rope_type={rope_type!r}")
        return
    required = (
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    )
    if any(name not in rope for name in required):
        reasons.append("llama3 RoPE is missing required scaling parameters")
        return
    try:
        factor = float(rope["factor"])
        low = float(rope["low_freq_factor"])
        high = float(rope["high_freq_factor"])
        original = _integer(
            rope["original_max_position_embeddings"],
            "original_max_position_embeddings",
        )
    except (TypeError, ValueError, OverflowError):
        reasons.append("llama3 RoPE scaling parameters must be numeric")
        return
    if (
        not all(math.isfinite(value) for value in (factor, low, high))
        or factor < 1.0
        or low <= 0.0
        or high <= low
        or original <= 0
    ):
        reasons.append("llama3 RoPE scaling parameters are invalid")


def _validate_rope_layer_schedule(
    config: object,
    raw: dict,
    reasons: list[str],
) -> None:
    """Reject a malformed NoPE schedule at routing time.

    SmolLM3 marks NoPE layers through ``no_rope_layers`` (1 = the layer applies
    RoPE, 0 = NoPE) or, when that list is absent, ``no_rope_layer_interval``.
    Resolving it here keeps a bad schedule from surfacing as an exception deep
    in the graph builder.
    """
    if raw.get("no_rope_layers") is None and raw.get(
        "no_rope_layer_interval"
    ) is None:
        return
    # Imported here so this module keeps the import surface the other
    # native-KV routing modules have (math and operator only).
    from .config import resolve_rope_layer_schedule

    try:
        schedule = resolve_rope_layer_schedule(config)
    except ValueError as exc:
        reasons.append(str(exc))
        return
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if len(schedule) != num_layers:
        reasons.append(
            "NoPE schedule must cover every layer: got "
            f"{len(schedule)} entries for {num_layers} layers"
        )


def native_kv_architecture_capability(
    config: object,
) -> NativeKvCapability:
    """Accept any model size that retains the dense SmolLM3 graph contract."""

    model_type = str(getattr(config, "model_type", "")).lower()
    if not model_type.startswith("smollm3"):
        return _result(applicable=False)
    if model_type != "smollm3":
        return _result(reasons=["model_type must be exactly 'smollm3'"])

    raw = _raw(config)
    reasons: list[str] = []
    if tuple(getattr(config, "architectures", ()) or ()) != (
        "SmolLM3ForCausalLM",
    ):
        reasons.append("architectures must contain exactly SmolLM3ForCausalLM")

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
        if dimensions["num_attention_heads"] % dimensions[
            "num_key_value_heads"
        ]:
            reasons.append(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )
        if head_dim != 128:
            reasons.append("native SmolLM3 attention requires head_dim=128")
    except ValueError as exc:
        reasons.append(str(exc))

    if str(getattr(config, "hidden_act", "")).lower() != "silu":
        reasons.append("native SmolLM3 requires hidden_act='silu'")
    for name in ("rms_norm_eps", "rope_theta"):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value) or value <= 0:
            reasons.append(f"{name} must be finite and positive")

    unsupported_flags = (
        "attention_bias",
        "mlp_bias",
        "is_encoder_decoder",
        "use_sliding_window",
        "sliding_window",
        "rope_interleaved",
        "interleaved_rope",
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
    )
    enabled = [name for name in unsupported_flags if _enabled(raw.get(name))]
    if enabled:
        reasons.append("unsupported SmolLM3 fields: " + ", ".join(enabled))
    try:
        if _integer(raw.get("pretraining_tp", 1), "pretraining_tp") != 1:
            reasons.append("native SmolLM3 requires pretraining_tp=1")
    except ValueError as exc:
        reasons.append(str(exc))
    try:
        if float(raw.get("partial_rotary_factor", 1.0)) != 1.0:
            reasons.append("native SmolLM3 requires full rotary embeddings")
    except (TypeError, ValueError, OverflowError):
        reasons.append("partial_rotary_factor must be numeric")
    layer_types = raw.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, (list, tuple))
        or any(str(value).lower() != "full_attention" for value in layer_types)
    ):
        reasons.append("native SmolLM3 does not support hybrid layer types")
    _validate_rope(raw, reasons)
    _validate_rope_layer_schedule(config, raw, reasons)
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
        reasons.append("native SmolLM3 requires BF16")
    if str(raw.get("_decoder_engine_layout", "split")) != "split":
        reasons.append("native SmolLM3 requires split prefill/decode engines")
    if raw.get("_rtx_build_requested"):
        reasons.append("native SmolLM3 requires the standard TensorRT backend")
    if parallel_enabled or raw.get("_parallel_build_enabled"):
        reasons.append("native SmolLM3 does not support tensor parallel builds")
    if (
        dynamic_kv_cache
        or raw.get("_runtime_dynamic_kv_requested")
        or raw.get("dynamic_kv_cache")
    ):
        reasons.append("native SmolLM3 uses one fixed physical KV capacity")
    if (
        quantized
        or raw.get("quantization_config")
        or raw.get("_quantized_build_requested")
    ):
        reasons.append("native SmolLM3 does not support quantized builds")
    if raw.get("_fp32_layers"):
        reasons.append("native SmolLM3 does not support FP32 layer overrides")
    if debug_layer_outputs:
        reasons.append("native SmolLM3 does not support debug layer outputs")
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


def prefer_native_default(
    config: object,
) -> bool:
    """Route dense SmolLM3 to native KV without a user-facing build flag."""

    return native_kv_architecture_capability(config).eligible
