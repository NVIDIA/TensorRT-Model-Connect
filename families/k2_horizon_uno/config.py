# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the qualified K2-Horizon-7B-Uno checkpoint recipe."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


BASE_ARCHITECTURE = "K2HorizonForCausalLM"
BASE_MODEL_ID = "IFM/K2-Horizon-7B"
BASE_REVISION = "586b03f0fd1fbbf2f13eeafc33749e95ae34dd10"
ADAPTER_FILENAME = "adapter_model.safetensors"
LORA_RANK = 128
LORA_ALPHA = 8192.0
LORA_SCALE = 64.0
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
)
DEFAULT_MAX_BLOCK_SIZE = 8


@dataclass(frozen=True)
class K2HorizonUnoConfig:
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
    lora_rank: int = LORA_RANK
    lora_scale: float = LORA_SCALE
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_attention_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


_EXACT_INTS = {
    "vocab_size": 250_624,
    "hidden_size": 4096,
    "intermediate_size": 12_288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 524_288,
    "layernorm_num_groups": 4,
}


def _raw(config: object) -> dict:
    value = getattr(config, "raw", {})
    return value if isinstance(value, dict) else {}


def _enabled(value: object) -> bool:
    return value not in (None, False, 0, "", (), [], {})


def _exact_positive_float(value: object, expected: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"K2-Horizon-Uno {name} must be {expected}") from error
    if not math.isfinite(resolved) or resolved != expected:
        raise ValueError(f"K2-Horizon-Uno {name} must be {expected}")
    return resolved


def load_and_validate_adapter_config(path: str | Path) -> dict:
    """Return the exact public PEFT recipe or fail closed."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Uno adapter config: {config_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("Uno adapter_config.json must contain a JSON object")

    expected = {
        "base_model_name_or_path": BASE_MODEL_ID,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": LORA_RANK,
        "revision": None,
        "target_modules": list(LORA_TARGET_MODULES),
        "task_type": "CAUSAL_LM",
    }
    if raw != expected:
        changed = sorted(
            key for key in set(raw) | set(expected) if raw.get(key) != expected.get(key)
        )
        raise ValueError(
            "Uno adapter_config.json does not match the qualified recipe; changed fields: "
            + ", ".join(changed)
        )
    return raw


def validate_config(config: object) -> K2HorizonUnoConfig:
    """Accept only the dense BF16 K2-Horizon-7B base graph."""

    raw = _raw(config)
    if str(getattr(config, "model_type", "")).lower() != "k2_horizon":
        raise ValueError("K2-Horizon-Uno base requires model_type='k2_horizon'")
    if tuple(getattr(config, "architectures", ()) or ()) != (BASE_ARCHITECTURE,):
        raise ValueError(
            f"K2-Horizon-Uno base architectures must contain exactly {BASE_ARCHITECTURE}"
        )

    resolved_ints: dict[str, int] = {}
    for name, expected in _EXACT_INTS.items():
        value = raw.get(name, getattr(config, name, None))
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"K2-Horizon-Uno {name} must be {expected}")
        resolved_ints[name] = value

    unsupported: list[str] = []
    if str(raw.get("dtype", "")).lower() != "bfloat16":
        unsupported.append("dtype")
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
        "mova_num_experts_per_tok",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "rope_interleaved",
        "interleaved_rope",
    ):
        if _enabled(raw.get(name)):
            unsupported.append(name)
    if bool(raw.get("tie_word_embeddings", getattr(config, "tie_word_embeddings", False))):
        unsupported.append("tie_word_embeddings")
    if raw.get("rope_head_dim", resolved_ints["head_dim"]) != resolved_ints["head_dim"]:
        unsupported.append("rope_head_dim")
    rope = raw.get("rope_parameters")
    if not isinstance(rope, dict) or rope != {
        "rope_theta": 10_000_000.0,
        "rope_type": "default",
    }:
        unsupported.append("rope_parameters")
    if raw.get("rope_scaling") is not None:
        unsupported.append("rope_scaling")
    if unsupported:
        raise ValueError(
            "K2-Horizon-Uno supports only the dense BF16 full-RoPE base graph; "
            "unsupported fields: " + ", ".join(sorted(set(unsupported)))
        )

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope.get("rope_theta") if isinstance(rope, dict) else None
    return K2HorizonUnoConfig(
        **resolved_ints,
        rms_norm_eps=_exact_positive_float(
            getattr(config, "rms_norm_eps", raw.get("rms_norm_eps")),
            1e-6,
            "rms_norm_eps",
        ),
        rope_theta=_exact_positive_float(rope_theta, 10_000_000.0, "rope_theta"),
    )
