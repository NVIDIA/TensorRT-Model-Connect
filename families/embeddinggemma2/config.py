# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-backbone dimensions and checkpoint tensor contracts."""

from dataclasses import dataclass
import math


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"EmbeddingGemma2 {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Config:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_size_per_layer_input: int
    embedding_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    layers: tuple

    @classmethod
    def from_dict(cls, raw):
        if raw.get("model_type") != "embedding_gemma2":
            raise ValueError("Expected embedding_gemma2 checkpoint metadata")
        text = raw.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("EmbeddingGemma2 requires text_config")
        names = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "hidden_size_per_layer_input",
            "embedding_dim",
            "max_position_embeddings",
        )
        values = {name: _positive(text.get(name), name) for name in names}
        eps = text.get("rms_norm_eps")
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("EmbeddingGemma2 rms_norm_eps must be finite and positive")
        if (
            text.get("hidden_activation") != "gelu_pytorch_tanh"
            or text.get("attention_bias") is not False
        ):
            raise ValueError("EmbeddingGemma2 requires bias-free attention and tanh GELU")
        layer_types = text.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != values["num_hidden_layers"]:
            raise ValueError("EmbeddingGemma2 layer_types does not match layer count")
        overrides = {}
        per_layer = text.get("per_layer_config", {})
        if "per_layer_config" not in text:
            per_layer = {
                str(index): {
                    "head_dim": text.get("global_head_dim", 512),
                    "num_key_value_heads": text.get("num_global_key_value_heads", 1),
                }
                for index, kind in enumerate(layer_types)
                if kind == "full_attention"
            }
        if not isinstance(per_layer, dict):
            raise ValueError("EmbeddingGemma2 per_layer_config must be an object")
        for key, fields in per_layer.items():
            index = int(key)
            if (
                index in overrides
                or not 0 <= index < len(layer_types)
                or not isinstance(fields, dict)
            ):
                raise ValueError("EmbeddingGemma2 invalid per-layer override")
            if set(fields) - {"head_dim", "num_key_value_heads"}:
                raise ValueError("EmbeddingGemma2 unsupported per-layer override")
            overrides[index] = fields
        layers = []
        for index, kind in enumerate(layer_types):
            if kind not in ("sliding_attention", "full_attention"):
                raise ValueError("EmbeddingGemma2 unsupported attention type")
            fields = overrides.get(index, {})
            dim = _positive(fields.get("head_dim", text.get("head_dim")), "head_dim")
            kv = _positive(
                fields.get("num_key_value_heads", text.get("num_key_value_heads")),
                "num_key_value_heads",
            )
            if dim % 2 or values["num_attention_heads"] % kv:
                raise ValueError("EmbeddingGemma2 invalid RoPE or grouped-query dimensions")
            rope = text.get("rope_parameters", {}).get(kind, {})
            theta = rope.get("rope_theta")
            if (
                rope.get("rope_type") != "default"
                or type(theta) not in (int, float)
                or not math.isfinite(theta)
                or theta <= 0
            ):
                raise ValueError("EmbeddingGemma2 supports only finite default RoPE")
            radius = (
                _positive(text.get("sliding_window"), "sliding_window")
                if kind == "sliding_attention"
                else None
            )
            layers.append((dim, kv, float(theta), radius))
        return cls(**values, rms_norm_eps=float(eps), layers=tuple(layers))

    def layer(self, index):
        return self.layers[index]

    def weight_shapes(self):
        hidden, ple = self.hidden_size, self.hidden_size_per_layer_input
        shapes = {
            "embed_tokens.weight": (self.vocab_size, hidden),
            "ple.per_layer_model_projection.weight": (self.num_hidden_layers * ple, hidden),
            "ple.per_layer_projection_norm.weight": (ple,),
            "norm.weight": (hidden,),
            "embedding_projection.weight": (self.embedding_dim, hidden),
        }
        for index in range(self.num_hidden_layers):
            dim, kv, _, _ = self.layer(index)
            heads = self.num_attention_heads
            local = {
                "self_attn.q_proj.weight": (heads * dim, hidden),
                "self_attn.k_proj.weight": (kv * dim, hidden),
                "self_attn.v_proj.weight": (kv * dim, hidden),
                "self_attn.o_proj.weight": (hidden, heads * dim),
                "self_attn.q_norm.weight": (dim,),
                "self_attn.k_norm.weight": (dim,),
                "mlp.gate_proj.weight": (self.intermediate_size, hidden),
                "mlp.up_proj.weight": (self.intermediate_size, hidden),
                "mlp.down_proj.weight": (hidden, self.intermediate_size),
                "ple_block.per_layer_input_gate.weight": (ple, hidden),
                "ple_block.per_layer_projection.weight": (hidden, ple),
                "ple_block.post_per_layer_input_norm.weight": (hidden,),
                "layer_scalar": (1,),
            }
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                local[name + ".weight"] = (hidden,)
            shapes.update({f"layers.{index}.{key}": shape for key, shape in local.items()})
        return shapes
