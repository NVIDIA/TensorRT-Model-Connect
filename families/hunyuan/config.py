# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact dense Hunyuan mathematical configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool
    head_dim: int
    hidden_act: str = "silu"
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        raw = json.loads(text)
        if not isinstance(raw, dict) or raw.get("model_type") != "hunyuan_v1_dense":
            raise ValueError("Hunyuan requires model_type=hunyuan_v1_dense")
        if raw.get("architectures") != ["HunYuanDenseV1ForCausalLM"]:
            raise ValueError("Hunyuan requires the dense causal-language architecture")
        if raw.get("use_cla", False) or not raw.get("use_qk_norm", True):
            raise ValueError("Hunyuan requires independent KV layers and post-RoPE Q/K normalization")
        if raw.get("attention_bias", False) or raw.get("mlp_bias", False):
            raise ValueError("Hunyuan bias-bearing projections are not supported")
        if raw.get("norm_type", "rms") != "rms" or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("Hunyuan requires RMSNorm and SwiGLU")
        dims = {}
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
            value = raw.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Hunyuan {name} must be a positive integer")
            dims[name] = value
        head_dim = raw.get("head_dim", dims["hidden_size"] // dims["num_attention_heads"])
        if (type(head_dim) is not int or head_dim <= 2 or head_dim % 2 or
                dims["hidden_size"] != dims["num_attention_heads"] * head_dim or
                dims["num_attention_heads"] % dims["num_key_value_heads"]):
            raise ValueError("Hunyuan has invalid grouped-attention dimensions")
        theta = float(raw.get("rope_theta", 10000))
        scaling = raw.get("rope_scaling") or raw.get("rope_parameters") or {}
        if scaling:
            kind = scaling.get("rope_type", scaling.get("type", "default"))
            if kind == "dynamic" and scaling.get("alpha"):
                alpha = float(scaling["alpha"])
                if not math.isfinite(alpha) or alpha <= 0 or scaling.get("factor", 1) != 1:
                    raise ValueError("Hunyuan has unsupported DynamicNTKAlpha parameters")
                # A fixed alpha-adjusted base, not length-dependent NTK.
                theta *= alpha ** (head_dim / (head_dim - 2))
            elif kind != "default":
                raise ValueError(f"Hunyuan RoPE mode is unsupported: {kind}")
        eps = float(raw.get("rms_norm_eps", 1e-5))
        if not math.isfinite(theta) or theta <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Hunyuan requires finite positive RoPE base and norm epsilon")
        tokens = {}
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = raw.get(name, -1)
            if type(value) is not int or value < -1 or value >= dims["vocab_size"]:
                raise ValueError(f"Hunyuan {name} is invalid")
            tokens[name] = value
        return ModelConfig(model_type=raw["model_type"], **dims, **tokens,
                           head_dim=head_dim, rms_norm_eps=eps, rope_theta=theta,
                           tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
                           raw=raw)

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_json((Path(model_dir) / "config.json").read_text())
