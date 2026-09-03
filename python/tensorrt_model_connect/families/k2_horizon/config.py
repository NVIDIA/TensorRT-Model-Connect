# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader-safe validation for the qualified K2-Horizon graph contract."""

from __future__ import annotations

from dataclasses import dataclass
import math


_ARCHITECTURE = "K2HorizonForCausalLM"
_MODEL_TYPE = "k2_horizon"


@dataclass(frozen=True)
class K2HorizonConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    layernorm_num_groups: int

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_attention_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def _raw(config: object) -> dict:
    value = getattr(config, "raw", {})
    return value if isinstance(value, dict) else {}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"K2-Horizon {name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"K2-Horizon {name} must be finite and positive") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"K2-Horizon {name} must be finite and positive")
    return resolved


def _enabled(value: object) -> bool:
    return value not in (None, False, 0, "", (), [], {})


def validate_config(config: object) -> K2HorizonConfig:
    """Fail closed unless the checkpoint matches the qualified dense graph."""

    raw = _raw(config)
    if str(getattr(config, "model_type", "")).lower() != _MODEL_TYPE:
        raise ValueError("K2-Horizon family requires model_type='k2_horizon'")
    if tuple(getattr(config, "architectures", ()) or ()) != (_ARCHITECTURE,):
        raise ValueError("K2-Horizon architectures must contain exactly K2HorizonForCausalLM")

    values = {
        name: _positive_int(getattr(config, name, None), name)
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
    head_dim = _positive_int(raw.get("head_dim", getattr(config, "head_dim", None)), "head_dim")
    if values["hidden_size"] != values["num_attention_heads"] * head_dim:
        raise ValueError("K2-Horizon hidden_size must equal num_attention_heads * head_dim")
    if values["num_attention_heads"] % values["num_key_value_heads"]:
        raise ValueError("K2-Horizon num_attention_heads must be divisible by num_key_value_heads")
    if head_dim != 128:
        raise ValueError("K2-Horizon native attention requires head_dim=128")

    groups = _positive_int(raw.get("layernorm_num_groups"), "layernorm_num_groups")
    if groups != 4:
        raise ValueError("K2-Horizon-7B requires layernorm_num_groups=4")
    if values["hidden_size"] % groups:
        raise ValueError("K2-Horizon hidden_size must be divisible by layernorm_num_groups")

    unsupported: list[str] = []
    if str(raw.get("hidden_act", getattr(config, "hidden_act", ""))).lower() != "silu":
        unsupported.append("hidden_act")
    for name in (
        "attention_bias",
        "mlp_bias",
        "query_key_norm",
        "attention_gate_func",
        "use_sliding_window",
        "sliding_window",
        "dynamic_kv_cache",
        "quantization_config",
        "is_encoder_decoder",
        "num_experts",
        "num_local_experts",
        "mova_num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "rope_interleaved",
        "interleaved_rope",
    ):
        if _enabled(raw.get(name)):
            unsupported.append(name)
    if bool(raw.get("tie_word_embeddings", getattr(config, "tie_word_embeddings", False))):
        unsupported.append("tie_word_embeddings")
    rope_head_dim = raw.get("rope_head_dim", head_dim)
    if (
        isinstance(rope_head_dim, bool)
        or not isinstance(rope_head_dim, int)
        or rope_head_dim != head_dim
    ):
        unsupported.append("rope_head_dim")
    for name in ("rope_parameters", "rope_scaling"):
        rope = raw.get(name)
        if rope is None:
            continue
        if not isinstance(rope, dict) or str(
            rope.get("rope_type", rope.get("type", "default"))
        ).lower() not in ("", "default"):
            unsupported.append(name)
        elif any(
            key in rope
            for key in (
                "attention_factor",
                "beta_fast",
                "beta_slow",
                "factor",
                "original_max_position_embeddings",
            )
        ):
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            "K2-Horizon support requires the dense BF16 full-RoPE graph; "
            "unsupported fields: " + ", ".join(sorted(set(unsupported)))
        )

    return K2HorizonConfig(
        **values,
        head_dim=head_dim,
        rms_norm_eps=_positive_float(
            getattr(config, "rms_norm_eps", raw.get("rms_norm_eps")),
            "rms_norm_eps",
        ),
        rope_theta=_positive_float(
            getattr(config, "rope_theta", raw.get("rope_theta")),
            "rope_theta",
        ),
        layernorm_num_groups=groups,
    )
