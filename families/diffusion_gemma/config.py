# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma's encoder, denoiser, and tied text-weight contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"diffusion_gemma {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    text_config: dict[str, Any]
    canvas_length: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        if not isinstance(raw, dict) or raw.get("model_type") != "diffusion_gemma":
            raise ValueError("diffusion_gemma requires a diffusion_gemma checkpoint")
        if raw.get("architectures") != ["DiffusionGemmaForBlockDiffusion"]:
            raise ValueError("diffusion_gemma requires DiffusionGemmaForBlockDiffusion")
        text = raw.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") != "diffusion_gemma_text":
            raise ValueError("diffusion_gemma requires a diffusion_gemma_text configuration")
        text = dict(text)
        for name in (
            "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "num_global_key_value_heads", "head_dim", "global_head_dim",
            "intermediate_size", "moe_intermediate_size", "num_experts", "top_k_experts",
            "max_position_embeddings", "sliding_window",
        ):
            _positive_int(text.get(name), name)
        if text["top_k_experts"] > text["num_experts"]:
            raise ValueError("diffusion_gemma top_k_experts exceeds num_experts")
        for name in ("num_key_value_heads", "num_global_key_value_heads"):
            if text["num_attention_heads"] % text[name]:
                raise ValueError(f"diffusion_gemma num_attention_heads must be divisible by {name}")
        layers = text.get("layer_types")
        if not isinstance(layers, list) or len(layers) != text["num_hidden_layers"]:
            raise ValueError("diffusion_gemma requires one layer_type per transformer layer")
        if any(kind not in {"full_attention", "sliding_attention"} for kind in layers):
            raise ValueError("diffusion_gemma has an unsupported attention layer type")
        if text.get("attention_bias", False):
            raise ValueError("diffusion_gemma attention projection biases are not supported")
        if text.get("hidden_activation") != "gelu_pytorch_tanh":
            raise ValueError("diffusion_gemma requires the checkpoint's tanh-approximate GELU")
        if text.get("use_bidirectional_attention") not in {"vision", "none", None}:
            raise ValueError("diffusion_gemma requires a causal text encoder")
        if not raw.get("tie_word_embeddings") or not text.get("tie_word_embeddings"):
            raise ValueError("diffusion_gemma requires tied text and LM-head embeddings")
        for name in ("rms_norm_eps", "final_logit_softcapping"):
            value = text.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"diffusion_gemma {name} must be finite and positive")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"diffusion_gemma {name} must be finite and positive")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("diffusion_gemma requires per-attention-type RoPE parameters")
        for kind in set(layers):
            parameters = rope.get(kind)
            if not isinstance(parameters, dict):
                raise ValueError(f"diffusion_gemma missing {kind} RoPE parameters")
            expected = "proportional" if kind == "full_attention" else "default"
            if parameters.get("rope_type") != expected:
                raise ValueError(f"diffusion_gemma unsupported {kind} RoPE type")
            head_dim = text["global_head_dim"] if kind == "full_attention" else text["head_dim"]
            fraction = parameters.get("partial_rotary_factor", 1.0)
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ValueError("diffusion_gemma partial_rotary_factor must be numeric")
            rotary_dim = int(head_dim * fraction)
            if not 0 < rotary_dim <= head_dim or rotary_dim % 2:
                raise ValueError("diffusion_gemma rotary dimension must be positive and even")
            theta = parameters.get("rope_theta")
            if not isinstance(theta, (int, float)) or isinstance(theta, bool):
                raise ValueError("diffusion_gemma rope_theta must be finite and positive")
            if not math.isfinite(theta) or theta <= 0:
                raise ValueError("diffusion_gemma rope_theta must be finite and positive")
        canvas = _positive_int(raw.get("canvas_length"), "canvas_length")
        if canvas > text["max_position_embeddings"]:
            raise ValueError("diffusion_gemma canvas exceeds checkpoint position capacity")
        for name in ("bos_token_id", "pad_token_id"):
            value = text.get(name)
            if type(value) is not int or not 0 <= value < text["vocab_size"]:
                raise ValueError(f"diffusion_gemma invalid {name}")
        return cls(raw=dict(raw), text_config=text, canvas_length=canvas)

    @classmethod
    def from_dir(cls, directory: str | Path) -> Config:
        return cls.from_dict(json.loads((Path(directory) / "config.json").read_text()))

    def layer_dimensions(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.text_config["num_hidden_layers"]:
            raise IndexError("diffusion_gemma layer index is out of range")
        full = self.text_config["layer_types"][index] == "full_attention"
        return (
            self.text_config["num_global_key_value_heads"] if full
            else self.text_config["num_key_value_heads"],
            self.text_config["global_head_dim"] if full else self.text_config["head_dim"],
        )

    def layer_weight_shapes(self, index: int) -> dict[str, tuple[int, ...]]:
        text = self.text_config
        hidden = text["hidden_size"]
        intermediate = text["intermediate_size"]
        experts = text["num_experts"]
        expert_width = text["moe_intermediate_size"]
        kv_heads, head_dim = self.layer_dimensions(index)
        shapes = {
            "self_attn.q_proj.weight": (text["num_attention_heads"] * head_dim, hidden),
            "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
            "self_attn.o_proj.weight": (hidden, text["num_attention_heads"] * head_dim),
            "self_attn.q_norm.weight": (head_dim,),
            "self_attn.k_norm.weight": (head_dim,),
            "mlp.gate_proj.weight": (intermediate, hidden),
            "mlp.up_proj.weight": (intermediate, hidden),
            "mlp.down_proj.weight": (hidden, intermediate),
            "router.proj.weight": (experts, hidden),
            "router.scale": (hidden,),
            "router.per_expert_scale": (experts,),
            "experts.gate_up_proj": (experts, 2 * expert_width, hidden),
            "experts.down_proj": (experts, hidden, expert_width),
            "layer_scalar": (1,),
        }
        for name in (
            "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
            "post_feedforward_layernorm", "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2",
        ):
            shapes[f"{name}.weight"] = (hidden,)
        if text["layer_types"][index] == "sliding_attention":
            shapes["self_attn.v_proj.weight"] = (kv_heads * head_dim, hidden)
        return shapes

    def embedding_weight_shapes(self) -> dict[str, tuple[int, ...]]:
        text = self.text_config
        hidden, intermediate = text["hidden_size"], text["intermediate_size"]
        return {
            "embed_tokens.weight": (text["vocab_size"], hidden),
            "self_conditioning.pre_norm.weight": (hidden,),
            "self_conditioning.gate_proj.weight": (intermediate, hidden),
            "self_conditioning.up_proj.weight": (intermediate, hidden),
            "self_conditioning.down_proj.weight": (hidden, intermediate),
        }

    def output_weight_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "embed_tokens.weight": (self.text_config["vocab_size"], self.text_config["hidden_size"]),
            "norm.weight": (self.text_config["hidden_size"],),
        }
