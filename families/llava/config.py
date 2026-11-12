# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LLaVA 1.5 CLIP/Llama configuration contract."""

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    bos_token_id: int = 1
    eos_token_id: int = 2
    raw: dict = field(default_factory=dict)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self):
        return self.hidden_size

    @classmethod
    def from_dir(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        text = raw.get("text_config", {})
        if raw.get("model_type") != "llava" or text.get("model_type", "llama") != "llama":
            raise ValueError("LLaVA requires a llava checkpoint with a Llama decoder")
        if raw.get("quantization_config") or text.get("quantization_config"):
            raise ValueError("LLaVA requires an unquantized checkpoint")
        rope = text.get("rope_parameters") or {}
        if text.get("rope_scaling") or rope.get("rope_type", "default") != "default":
            raise ValueError("LLaVA requires unscaled RoPE")
        if text.get("attention_bias", False) or text.get("mlp_bias", False) or text.get("hidden_act", "silu") != "silu":
            raise ValueError("LLaVA requires bias-free Llama attention and a SwiGLU MLP")
        from transformers import LlamaConfig
        # The original checkpoint omits the standard Llama 7B dimensions.
        # Resolve the same serialized defaults as the upstream reference.
        text = LlamaConfig(**text).to_dict()
        vision = raw.get("vision_config", {})
        if vision.get("model_type", "clip_vision_model") != "clip_vision_model":
            raise ValueError("LLaVA requires the CLIP vision tower")
        if raw.get("vision_feature_select_strategy", "default") != "default":
            raise ValueError("LLaVA supports patch features with the CLS token removed")
        if raw.get("vision_feature_layer", -2) != -2:
            raise ValueError("LLaVA requires the penultimate CLIP hidden state")
        if vision.get("hidden_act", "quick_gelu") != "quick_gelu":
            raise ValueError("LLaVA requires CLIP quick_gelu")
        if raw.get("projector_hidden_act", "gelu") != "gelu":
            raise ValueError("LLaVA requires the GELU multimodal projector")
        if text.get("rope_scaling") or text.get("tie_word_embeddings", False):
            raise ValueError("LLaVA requires unscaled RoPE and an untied LM head")
        fields = {name: text[name] for name in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads")}
        fields["num_key_value_heads"] = text.get("num_key_value_heads", fields["num_attention_heads"])
        fields["vocab_size"] = text.get("vocab_size", raw.get("vocab_size", 32064))
        for name in ("rms_norm_eps", "rope_theta", "max_position_embeddings", "bos_token_id", "eos_token_id"):
            if name in text:
                fields[name] = text[name]
        return cls(**fields, raw=raw)
