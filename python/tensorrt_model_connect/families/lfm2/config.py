# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated dense LFM2 configuration owned by the LFM2 family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


_DENSE_ARCHITECTURE = "Lfm2ForCausalLM"
_CONV_LAYER = "conv"
_ATTENTION_LAYER = "full_attention"
_MOE_FIELDS = (
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "expert_intermediate_size",
    "router_aux_loss_coef",
)
_VL_FIELDS = (
    "vision_config",
    "image_token_id",
    "image_token_index",
    "mm_projector_type",
)


def _raw_config(config: object) -> dict[str, Any]:
    raw = getattr(config, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("LFM2 config.raw must be a JSON object")
    return raw


def _first(raw: Mapping[str, Any], config: object, *names: str, default=None):
    for name in names:
        if name in raw and raw[name] is not None:
            return raw[name]
        value = getattr(config, name, None)
        if value is not None and value != 0 and value != "":
            return value
    return default


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _enabled(value: object) -> bool:
    return value not in (None, False, 0, "", (), [], {})


def _resolve_layer_types(raw: Mapping[str, Any], num_layers: int) -> tuple[str, ...]:
    layer_types = raw.get("layer_types")
    declared_attention_indices = raw.get("full_attn_idxs", raw.get("full_attention_indices"))
    if layer_types is None:
        indices = declared_attention_indices
        if not isinstance(indices, (list, tuple)):
            raise ValueError("dense LFM2 requires layer_types or full_attn_idxs")
        attention_indices: set[int] = set()
        for value in indices:
            if isinstance(value, bool):
                raise ValueError("full_attn_idxs must contain integer indices")
            try:
                index = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("full_attn_idxs must contain integer indices") from exc
            if not 0 <= index < num_layers:
                raise ValueError(f"full_attn_idxs contains out-of-range layer {index}")
            if index in attention_indices:
                raise ValueError(f"full_attn_idxs contains duplicate layer {index}")
            attention_indices.add(index)
        layer_types = [
            _ATTENTION_LAYER if index in attention_indices else _CONV_LAYER
            for index in range(num_layers)
        ]
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError("layer_types must be a list")
    normalized = tuple(str(value).lower() for value in layer_types)
    if len(normalized) != num_layers:
        raise ValueError(
            f"layer_types length {len(normalized)} does not match num_hidden_layers {num_layers}"
        )
    invalid = sorted(set(normalized) - {_CONV_LAYER, _ATTENTION_LAYER})
    if invalid:
        raise ValueError(f"unsupported dense LFM2 layer types: {invalid}")
    if _ATTENTION_LAYER not in normalized:
        raise ValueError("dense LFM2 requires at least one full_attention layer")
    if _CONV_LAYER not in normalized:
        raise ValueError("dense LFM2 requires at least one conv layer")
    if declared_attention_indices is not None and layer_types is not None:
        if not isinstance(declared_attention_indices, (list, tuple)):
            raise ValueError("full_attn_idxs must be a list")
        expected_indices = tuple(
            index for index, value in enumerate(normalized) if value == _ATTENTION_LAYER
        )
        try:
            actual_indices = tuple(int(value) for value in declared_attention_indices)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("full_attn_idxs must contain integer indices") from exc
        if actual_indices != expected_indices:
            raise ValueError("full_attn_idxs must agree with layer_types")
    return normalized


def _resolve_intermediate_size(raw: Mapping[str, Any], config: object) -> int:
    if "block_ff_dim" in raw and "intermediate_size" in raw:
        block_ff_dim = _positive_int(raw["block_ff_dim"], "block_ff_dim")
        intermediate_alias = _positive_int(raw["intermediate_size"], "intermediate_size")
        if intermediate_alias != block_ff_dim:
            raise ValueError("intermediate_size must equal the pre-adjust block_ff_dim")
    base = _positive_int(
        _first(raw, config, "block_ff_dim", "intermediate_size"),
        "block_ff_dim",
    )
    if not bool(raw.get("block_auto_adjust_ff_dim", True)):
        return base
    adjusted = int(2 * base / 3)
    multiplier = raw.get("block_ffn_dim_multiplier", 1.0)
    try:
        multiplier_value = float(multiplier)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("block_ffn_dim_multiplier must be numeric") from exc
    if not math.isfinite(multiplier_value) or multiplier_value <= 0:
        raise ValueError("block_ffn_dim_multiplier must be finite and positive")
    adjusted = int(multiplier_value * adjusted)
    multiple = _positive_int(raw.get("block_multiple_of", 256), "block_multiple_of")
    return multiple * ((adjusted + multiple - 1) // multiple)


@dataclass(frozen=True)
class DenseLfm2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    norm_eps: float
    rope_theta: float
    conv_l_cache: int
    conv_bias: bool
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    layer_types: tuple[str, ...]

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_attention_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def num_attention_layers(self) -> int:
        return self.layer_types.count(_ATTENTION_LAYER)

    @property
    def num_conv_layers(self) -> int:
        return self.layer_types.count(_CONV_LAYER)

    @property
    def default_cache_length(self) -> int:
        return min(self.max_position_embeddings, 32768)

    def bundle_overrides(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "layer_types": list(self.layer_types),
            "num_attention_layers": self.num_attention_layers,
            "num_conv_layers": self.num_conv_layers,
            "conv_L_cache": self.conv_l_cache,
            "conv_dim": self.hidden_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "native_kv_cache": True,
            "native_kv_contract_version": 1,
        }


def validate_dense_lfm2_config(config: object) -> DenseLfm2Config:
    """Normalize legacy LFM2 keys and reject non-dense architecture variants."""

    raw = _raw_config(config)
    model_type = str(_first(raw, config, "model_type", default="") or "").lower()
    if model_type != "lfm2":
        raise ValueError(f"LFM2 builder requires model_type='lfm2', got {model_type!r}")

    architectures = raw.get("architectures", getattr(config, "architectures", ())) or ()
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, (list, tuple)):
        raise ValueError("architectures must be a list")
    if architectures and tuple(str(value) for value in architectures) != (_DENSE_ARCHITECTURE,):
        raise ValueError("dense LFM2 supports exactly architectures=['Lfm2ForCausalLM']")

    unsupported = [name for name in (*_MOE_FIELDS, *_VL_FIELDS) if _enabled(raw.get(name))]
    if unsupported:
        raise ValueError("dense LFM2 does not support MoE/VL fields: " + ", ".join(unsupported))

    hidden = _positive_int(
        _first(raw, config, "hidden_size", "block_dim", "conv_dim"),
        "hidden_size",
    )
    if "block_dim" in raw and _positive_int(raw["block_dim"], "block_dim") != hidden:
        raise ValueError("block_dim must equal hidden_size")
    vocab = _positive_int(_first(raw, config, "vocab_size"), "vocab_size")
    num_layers = _positive_int(
        _first(raw, config, "num_hidden_layers", "num_layers"),
        "num_hidden_layers",
    )
    if "num_layers" in raw and _positive_int(raw["num_layers"], "num_layers") != num_layers:
        raise ValueError("num_layers must equal num_hidden_layers")
    num_heads = _positive_int(
        _first(raw, config, "num_attention_heads", "num_heads"),
        "num_attention_heads",
    )
    if "num_heads" in raw and _positive_int(raw["num_heads"], "num_heads") != num_heads:
        raise ValueError("num_heads must equal num_attention_heads")
    num_kv_heads = _positive_int(
        _first(raw, config, "num_key_value_heads", default=num_heads),
        "num_key_value_heads",
    )
    if hidden % num_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if num_heads % num_kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    head_dim = hidden // num_heads
    explicit_head_dim = raw.get("head_dim")
    if explicit_head_dim is not None and _positive_int(explicit_head_dim, "head_dim") != head_dim:
        raise ValueError("head_dim must equal hidden_size / num_attention_heads")

    conv_dim = _positive_int(raw.get("conv_dim", hidden), "conv_dim")
    conv_dim_out = _positive_int(raw.get("conv_dim_out", hidden), "conv_dim_out")
    if conv_dim != hidden or conv_dim_out != hidden:
        raise ValueError("dense LFM2 v1 requires conv_dim == conv_dim_out == hidden_size")
    if not bool(raw.get("block_use_swiglu", True)):
        raise ValueError("dense LFM2 requires block_use_swiglu=true")
    if not bool(raw.get("use_pos_enc", True)):
        raise ValueError("dense LFM2 requires use_pos_enc=true")

    conv_l_cache = _positive_int(
        raw.get("conv_L_cache", raw.get("conv_l_cache", 3)),
        "conv_L_cache",
    )
    if (
        "conv_L_cache" in raw
        and "conv_l_cache" in raw
        and _positive_int(raw["conv_l_cache"], "conv_l_cache") != conv_l_cache
    ):
        raise ValueError("conv_l_cache must equal conv_L_cache")
    max_positions = _positive_int(
        _first(raw, config, "max_position_embeddings", default=128000),
        "max_position_embeddings",
    )
    norm_eps = _positive_float(
        _first(raw, config, "norm_eps", "block_norm_eps", "rms_norm_eps", default=1e-5),
        "norm_eps",
    )
    if (
        "block_norm_eps" in raw
        and _positive_float(raw["block_norm_eps"], "block_norm_eps") != norm_eps
    ):
        raise ValueError("block_norm_eps must equal norm_eps")
    rope_theta = _positive_float(
        _first(raw, config, "rope_theta", "theta", default=1_000_000.0),
        "rope_theta",
    )
    if "theta" in raw and _positive_float(raw["theta"], "theta") != rope_theta:
        raise ValueError("theta must equal rope_theta")
    if (
        "tie_word_embeddings" in raw
        and "tie_embedding" in raw
        and bool(raw["tie_word_embeddings"]) != bool(raw["tie_embedding"])
    ):
        raise ValueError("tie_embedding must equal tie_word_embeddings")
    tie_embeddings = bool(raw.get("tie_word_embeddings", raw.get("tie_embedding", True)))

    return DenseLfm2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=_resolve_intermediate_size(raw, config),
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_positions,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        conv_l_cache=conv_l_cache,
        conv_bias=bool(raw.get("conv_bias", False)),
        tie_word_embeddings=tie_embeddings,
        bos_token_id=int(_first(raw, config, "bos_token_id", default=-1)),
        eos_token_id=int(_first(raw, config, "eos_token_id", default=-1)),
        pad_token_id=int(_first(raw, config, "pad_token_id", default=-1)),
        layer_types=_resolve_layer_types(raw, num_layers),
    )
