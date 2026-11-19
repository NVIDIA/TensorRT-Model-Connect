# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apertus checkpoint configuration, validated against the graph this family builds.

Every field that changes the decoder's computation is either consumed here or
rejected. Hugging Face ``ApertusForCausalLM`` always builds the per-head query
and key RMSNorms and the pre-norm residual blocks, so ``qk_norm`` and
``post_norm`` are accepted only with the values that match that graph.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ARCHITECTURE = "ApertusForCausalLM"
_INT32_MAX = (1 << 31) - 1
_ROPE_TYPES = ("default", "llama3")


@dataclass(frozen=True)
class RopeConfig:
    """Rotary position parameters; the scaling fields are used only by ``llama3``."""

    theta: float
    rope_type: str
    factor: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 1.0
    original_max_position_embeddings: int = 0


@dataclass(frozen=True)
class ApertusConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope: RopeConfig
    bos_token_id: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def _integer(raw: dict, name: str, *, minimum: int = 1) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"Apertus config field {name!r} must be an integer >= {minimum}")
    return value


def _positive_float(raw: dict, name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Apertus config field {name!r} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Apertus config field {name!r} must be finite and positive")
    return value


def _require_flag(raw: dict, name: str, expected: bool) -> None:
    value = raw.get(name, expected)
    if not isinstance(value, bool) or value is not expected:
        raise ValueError(f"Apertus supports only {name}={expected}, got {value!r}")


def _rope(raw: dict) -> RopeConfig:
    parameters = raw.get("rope_parameters")
    scaling = raw.get("rope_scaling")
    if parameters is not None and scaling is not None and parameters != scaling:
        raise ValueError("Apertus config has conflicting rope_parameters and rope_scaling")
    rope = parameters if parameters is not None else scaling
    if not rope:
        # transformers substitutes defaults for a missing or empty dict (llama3 or plain RoPE,
        # depending on the key); require the checkpoint to state its schedule instead.
        raise ValueError("Apertus config must declare rope_parameters or rope_scaling")
    if not isinstance(rope, dict):
        raise ValueError("Apertus RoPE configuration must be an object")
    if "rope_theta" in rope and "rope_theta" in raw and rope["rope_theta"] != raw["rope_theta"]:
        raise ValueError("Apertus config has conflicting rope_theta values")
    merged = {"rope_theta": raw.get("rope_theta"), **rope}
    rope_type = merged.get("rope_type", merged.get("type", "default"))
    if "type" in merged and merged.get("type") != rope_type:
        raise ValueError("Apertus RoPE configuration has conflicting type fields")
    if rope_type not in _ROPE_TYPES:
        raise ValueError(f"Apertus supports RoPE types {_ROPE_TYPES}, got {rope_type!r}")
    partial = merged.get("partial_rotary_factor", 1.0)
    if partial != 1.0:
        raise ValueError("Apertus supports only full rotary embeddings")
    theta = _positive_float(merged, "rope_theta")
    if rope_type == "default":
        return RopeConfig(theta=theta, rope_type=rope_type)
    low = _positive_float(merged, "low_freq_factor")
    high = _positive_float(merged, "high_freq_factor")
    if high <= low:
        raise ValueError("Apertus llama3 RoPE requires high_freq_factor > low_freq_factor")
    return RopeConfig(
        theta=theta,
        rope_type=rope_type,
        factor=_positive_float(merged, "factor"),
        low_freq_factor=low,
        high_freq_factor=high,
        original_max_position_embeddings=_integer(merged, "original_max_position_embeddings"),
    )


def _eos_token_ids(raw: dict, generation: dict) -> tuple[int, ...]:
    value = generation.get("eos_token_id", raw.get("eos_token_id"))
    values = value if isinstance(value, list) else [value]
    if not values or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values
    ):
        raise ValueError("Apertus eos_token_id must be a non-negative integer or a list of them")
    return tuple(dict.fromkeys(values))


def parse_config(raw: dict, generation: dict) -> ApertusConfig:
    """Validate ``config.json`` and ``generation_config.json`` for the native graph."""
    if raw.get("model_type") != "apertus":
        raise ValueError(f"Apertus requires model_type='apertus', got {raw.get('model_type')!r}")
    if raw.get("architectures") != [ARCHITECTURE]:
        raise ValueError(f"Apertus requires architectures=[{ARCHITECTURE!r}]")
    if raw.get("hidden_act") != "xielu":
        raise ValueError(f"Apertus requires hidden_act='xielu', got {raw.get('hidden_act')!r}")
    for name, expected in (
        ("attention_bias", False),
        ("mlp_bias", False),
        ("qk_norm", True),
        ("post_norm", False),
        ("tie_word_embeddings", False),
    ):
        _require_flag(raw, name, expected)
    hidden = _integer(raw, "hidden_size")
    heads = _integer(raw, "num_attention_heads")
    kv_heads = _integer(raw, "num_key_value_heads")
    if heads % kv_heads:
        raise ValueError("Apertus num_attention_heads must be divisible by num_key_value_heads")
    if raw.get("head_dim") is None:
        if hidden % heads:
            raise ValueError("Apertus hidden_size must be divisible by num_attention_heads")
        head_dim = hidden // heads
    else:
        head_dim = _integer(raw, "head_dim")
    if head_dim % 2:
        raise ValueError("Apertus rotary embeddings require an even head_dim")
    vocab = _integer(raw, "vocab_size")
    bos = _integer(generation if "bos_token_id" in generation else raw, "bos_token_id", minimum=0)
    eos = _eos_token_ids(raw, generation)
    pad = raw.get("pad_token_id", -1)
    if pad is None:
        pad = -1
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < -1:
        raise ValueError("Apertus pad_token_id must be an integer")
    if max((bos, pad, *eos)) >= vocab:
        raise ValueError("Apertus special token IDs must be inside the vocabulary")
    return ApertusConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=_integer(raw, "intermediate_size"),
        num_hidden_layers=_integer(raw, "num_hidden_layers"),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rms_norm_eps=_positive_float(raw, "rms_norm_eps"),
        max_position_embeddings=_integer(raw, "max_position_embeddings"),
        rope=_rope(raw),
        bos_token_id=bos,
        eos_token_ids=eos,
        pad_token_id=pad,
    )


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def load_config(model_dir: Path) -> ApertusConfig:
    generation_path = model_dir / "generation_config.json"
    generation = _read_object(generation_path) if generation_path.is_file() else {}
    return parse_config(_read_object(model_dir / "config.json"), generation)
