# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned Qwen3-Next topology, independent of the executing platform."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    full_attention_interval: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    max_position_embeddings: int
    eos_token_id: int

    @property
    def layer_types(self):
        return tuple(
            "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        )

    @property
    def conv_dim(self):
        return (
            2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )

    @classmethod
    def load(cls, directory: Path):
        raw = json.loads((directory / "config.json").read_text())
        required = {
            "model_type": "qwen3_next",
            "hidden_act": "silu",
            "attention_bias": False,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
            "use_sliding_window": False,
            "rope_scaling": None,
        }
        for field, expected in required.items():
            if raw.get(field) != expected:
                raise ValueError(f"unsupported Qwen3-Next {field}: {raw.get(field)!r}")
        config = cls(**{field: raw[field] for field in cls.__dataclass_fields__})
        for field in cls.__dataclass_fields__:
            if getattr(config, field) <= 0:
                raise ValueError(f"Qwen3-Next {field} must be positive")
        if (
            config.num_attention_heads % config.num_key_value_heads
            or config.linear_num_value_heads % config.linear_num_key_heads
        ):
            raise ValueError("Qwen3-Next head groups must divide evenly")
        rotary = config.head_dim * config.partial_rotary_factor
        if rotary != int(rotary) or int(rotary) % 2:
            raise ValueError("Qwen3-Next rotary dimension must be an even integer")
        if "layer_types" in raw and tuple(raw["layer_types"]) != config.layer_types:
            raise ValueError("Qwen3-Next layer pattern disagrees with full_attention_interval")
        return config
