# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Laguna ``config.json`` contract.

Missing optional keys take the defaults of the Hugging Face ``LagunaConfig``;
every value the family cannot represent is rejected instead of ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_LAYER_TYPES = (FULL_ATTENTION, SLIDING_ATTENTION)
_MLP_TYPES = ("dense", "sparse")
_DEFAULT_ROPE = {
    FULL_ATTENTION: {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
    SLIDING_ATTENTION: {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
    },
}


@dataclass(frozen=True)
class RopeSpec:
    """One layer type's rotary embedding (``default`` or ``yarn``)."""

    rope_type: str
    theta: float
    rotary_dim: int
    factor: float = 1.0
    original_max_position_embeddings: int = 0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None
    truncate: bool = True


@dataclass(frozen=True)
class LagunaConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    heads_per_layer: tuple[int, ...]
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    intermediate_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    router_logit_softcapping: float
    rms_norm_eps: float
    sliding_window: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    rope: dict[str, RopeSpec]

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


def _positive_int(raw: dict, name: str, default: int | None = None) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Laguna config {name} must be a positive integer, got {value!r}")
    return value


def _number(raw: dict, name: str, default: float) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Laguna config {name} must be a number, got {value!r}")
    return float(value)


def _token_id(raw: dict, name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Laguna config {name} must be a non-negative integer, got {value!r}")
    return value


def _eos_ids(raw: dict) -> tuple[int, ...]:
    value = raw.get("eos_token_id")
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError(f"Laguna config eos_token_id is invalid: {value!r}")
    return tuple(values)


def _layer_list(raw: dict, name: str, default: list, allowed: tuple, layers: int) -> tuple:
    value = raw.get(name)
    value = default if value is None else value
    if not isinstance(value, list) or len(value) != layers:
        raise ValueError(f"Laguna config {name} must list one entry per layer")
    for entry in value:
        if entry not in allowed:
            raise ValueError(f"Laguna config {name} has unsupported entry {entry!r}")
    return tuple(value)


def _layer_rope_parameters(raw: dict, layer_type: str) -> dict:
    """One layer type's rope parameters with the reference defaults applied.

    As transformers ``standardize_rope_params`` does for per-layer-type rope
    parameters, a top-level ``rope_theta`` / ``partial_rotary_factor`` fills a
    missing per-layer value and YaRN's original context defaults to
    ``max_position_embeddings``. Flat (not per-layer-type) rope parameters are
    rejected, as the reference rotary embedding indexes them by layer type.
    """
    rope = raw.get("rope_parameters") or _DEFAULT_ROPE
    if not isinstance(rope, dict) or not isinstance(rope.get(layer_type), dict):
        raise ValueError(f"Laguna rope_parameters must hold an object for {layer_type!r}")
    params = dict(rope[layer_type])
    params.setdefault("rope_type", params.get("type", "default"))
    if raw.get("rope_theta") is not None:
        params.setdefault("rope_theta", raw["rope_theta"])
    if raw.get("partial_rotary_factor") is not None:
        params.setdefault("partial_rotary_factor", raw["partial_rotary_factor"])
    if params["rope_type"] == "yarn" and raw.get("max_position_embeddings") is not None:
        params.setdefault("original_max_position_embeddings", raw["max_position_embeddings"])
    return params


def _rope_spec(params: dict, head_dim: int, layer_type: str) -> RopeSpec:
    rope_type = params["rope_type"]
    theta = _number(params, "rope_theta", 0.0)
    partial = _number(params, "partial_rotary_factor", 1.0)
    rotary_dim = int(head_dim * partial)
    if theta <= 0.0 or rotary_dim < 2 or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError(f"Laguna rope_parameters[{layer_type!r}] is invalid: {params!r}")
    if rope_type == "default":
        return RopeSpec("default", theta, rotary_dim)
    if rope_type != "yarn":
        raise ValueError(f"Laguna does not support rope_type={rope_type!r}")
    if params.get("mscale") or params.get("mscale_all_dim"):
        raise ValueError("Laguna YaRN with mscale / mscale_all_dim is not supported")
    attention_factor = params.get("attention_factor")
    factor = _number(params, "factor", 0.0)
    if factor <= 0.0:
        raise ValueError(f"Laguna YaRN rope_parameters[{layer_type!r}] needs factor > 0")
    return RopeSpec(
        "yarn",
        theta,
        rotary_dim,
        factor=factor,
        original_max_position_embeddings=_positive_int(params, "original_max_position_embeddings"),
        beta_fast=float(params.get("beta_fast") or 32.0),
        beta_slow=float(params.get("beta_slow") or 1.0),
        attention_factor=None if attention_factor is None else float(attention_factor),
    )


def parse_config(raw: dict) -> LagunaConfig:
    """Validate one Laguna ``config.json`` object."""
    if raw.get("model_type") != "laguna":
        raise ValueError(f"Laguna does not support model_type={raw.get('model_type')!r}")
    if raw.get("architectures") not in (None, ["LagunaForCausalLM"]):
        raise ValueError(f"Laguna does not support architectures={raw.get('architectures')!r}")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("Laguna supports only hidden_act=silu")
    if raw.get("attention_bias", False):
        raise ValueError("Laguna attention_bias=True is not supported")
    if raw.get("gating", True) not in (True, "per-head"):
        raise ValueError("Laguna supports only the per-head softplus attention gate")
    if raw.get("moe_apply_router_weight_on_input", False):
        raise ValueError("Laguna moe_apply_router_weight_on_input=True is not supported")

    layers = _positive_int(raw, "num_hidden_layers", 40)
    head_dim = _positive_int(raw, "head_dim", 128)
    num_heads = _positive_int(raw, "num_attention_heads", 48)
    kv_heads = _positive_int(raw, "num_key_value_heads", 8)
    heads = raw.get("num_attention_heads_per_layer") or [num_heads] * layers
    if not isinstance(heads, list) or len(heads) != layers:
        raise ValueError("Laguna num_attention_heads_per_layer must list one entry per layer")
    for count in heads:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count % kv_heads:
            raise ValueError(f"Laguna heads per layer must be multiples of {kv_heads}: {heads}")
    layer_types = _layer_list(raw, "layer_types", [FULL_ATTENTION] * layers, _LAYER_TYPES, layers)
    mlp_types = _layer_list(
        raw, "mlp_layer_types", ["dense"] + ["sparse"] * (layers - 1), _MLP_TYPES, layers
    )

    rope = {
        kind: _rope_spec(_layer_rope_parameters(raw, kind), head_dim, kind)
        for kind in set(layer_types)
    }

    config = LagunaConfig(
        vocab_size=_positive_int(raw, "vocab_size", 100352),
        hidden_size=_positive_int(raw, "hidden_size", 2048),
        num_hidden_layers=layers,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        heads_per_layer=tuple(heads),
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
        intermediate_size=_positive_int(raw, "intermediate_size", 8192),
        moe_intermediate_size=_positive_int(raw, "moe_intermediate_size", 512),
        shared_expert_intermediate_size=_positive_int(raw, "shared_expert_intermediate_size", 512),
        num_experts=_positive_int(raw, "num_experts", 256),
        num_experts_per_tok=_positive_int(raw, "num_experts_per_tok", 8),
        routed_scaling_factor=_number(raw, "moe_routed_scaling_factor", 1.0),
        router_logit_softcapping=_number(raw, "moe_router_logit_softcapping", 0.0),
        rms_norm_eps=_number(raw, "rms_norm_eps", 1e-6),
        sliding_window=_positive_int(raw, "sliding_window", 512),
        max_position_embeddings=_positive_int(raw, "max_position_embeddings", 131072),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        bos_token_id=_token_id(raw, "bos_token_id"),
        eos_token_ids=_eos_ids(raw),
        pad_token_id=_token_id(raw, "pad_token_id"),
        rope=rope,
    )
    if config.num_experts_per_tok > config.num_experts:
        raise ValueError("Laguna num_experts_per_tok exceeds num_experts")
    if config.router_logit_softcapping < 0.0:
        raise ValueError("Laguna moe_router_logit_softcapping must be non-negative")
    return config


def load_config(model_dir: str | Path) -> LagunaConfig:
    path = Path(model_dir) / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return parse_config(raw)
