# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V 4.7 checkpoint configuration with the exact geometry this family implements."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

_LAYER_TYPES = {"linear_attention", "full_attention"}


def _int(raw: dict, name: str, *, minimum: int = 1) -> int:
    value = raw.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"MiniCPM-V 4.7 config field {name!r} must be an integer >= {minimum}")
    return value


def _float(raw: dict, name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"MiniCPM-V 4.7 config field {name!r} must be a positive number")
    return float(value)


def _require(raw: dict, name: str, expected) -> None:
    if raw.get(name, expected) != expected:
        raise ValueError(f"MiniCPM-V 4.7 family supports {name}={expected!r}, got {raw.get(name)!r}")


@dataclass(frozen=True)
class TextConfig:
    """Qwen3.5-MoE hybrid decoder: Gated DeltaNet and gated full attention, routed experts."""

    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]
    max_position_embeddings: int

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def num_linear_layers(self) -> int:
        return sum(kind == "linear_attention" for kind in self.layer_types)

    @property
    def num_full_layers(self) -> int:
        return sum(kind == "full_attention" for kind in self.layer_types)

    @staticmethod
    def from_dict(raw: dict) -> TextConfig:
        if raw.get("model_type") != "qwen3_5_moe_text":
            raise ValueError("MiniCPM-V 4.7 requires a qwen3_5_moe_text language model")
        for name, expected in (("hidden_act", "silu"), ("attention_bias", False), ("attn_output_gate", True),
                               ("mamba_ssm_dtype", "float32"), ("tie_word_embeddings", False)):
            _require(raw, name, expected)
        rope = raw.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_type", "default") != "default":
            raise ValueError("MiniCPM-V 4.7 requires default multimodal RoPE parameters")
        if rope.get("mrope_interleaved") is not True:
            raise ValueError("MiniCPM-V 4.7 requires interleaved multimodal RoPE")
        section = rope.get("mrope_section")
        if (not isinstance(section, list) or len(section) != 3
                or any(type(value) is not int or value < 0 for value in section)):
            raise ValueError("MiniCPM-V 4.7 requires three multimodal RoPE sections")
        layer_types = raw.get("layer_types")
        if not isinstance(layer_types, list) or any(kind not in _LAYER_TYPES for kind in layer_types):
            raise ValueError("MiniCPM-V 4.7 layer_types must name linear or full attention layers")
        config = TextConfig(
            hidden_size=_int(raw, "hidden_size"),
            num_hidden_layers=_int(raw, "num_hidden_layers"),
            layer_types=tuple(layer_types),
            num_attention_heads=_int(raw, "num_attention_heads"),
            num_key_value_heads=_int(raw, "num_key_value_heads"),
            head_dim=_int(raw, "head_dim"),
            linear_num_key_heads=_int(raw, "linear_num_key_heads"),
            linear_num_value_heads=_int(raw, "linear_num_value_heads"),
            linear_key_head_dim=_int(raw, "linear_key_head_dim"),
            linear_value_head_dim=_int(raw, "linear_value_head_dim"),
            linear_conv_kernel_dim=_int(raw, "linear_conv_kernel_dim"),
            num_experts=_int(raw, "num_experts"),
            num_experts_per_tok=_int(raw, "num_experts_per_tok"),
            moe_intermediate_size=_int(raw, "moe_intermediate_size"),
            shared_expert_intermediate_size=_int(raw, "shared_expert_intermediate_size"),
            vocab_size=_int(raw, "vocab_size"),
            rms_norm_eps=_float(raw, "rms_norm_eps"),
            rope_theta=_float(rope, "rope_theta"),
            partial_rotary_factor=_float(rope, "partial_rotary_factor"),
            mrope_section=tuple(section),
            max_position_embeddings=_int(raw, "max_position_embeddings"),
        )
        if len(config.layer_types) != config.num_hidden_layers or not config.num_full_layers \
                or not config.num_linear_layers:
            raise ValueError("MiniCPM-V 4.7 layer_types must cover every decoder layer with both mixers")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("MiniCPM-V 4.7 attention heads must group evenly over KV heads")
        if (config.linear_num_value_heads % config.linear_num_key_heads
                or config.linear_key_head_dim != config.linear_value_head_dim):
            raise ValueError("MiniCPM-V 4.7 DeltaNet heads must group evenly with square states")
        rotary = config.rotary_dim
        if rotary < 2 or rotary % 2 or rotary > config.head_dim or sum(section) != rotary // 2:
            raise ValueError("MiniCPM-V 4.7 multimodal RoPE sections must cover the rotary half dimension")
        if raw.get("partial_rotary_factor") not in (None, config.partial_rotary_factor):
            raise ValueError("MiniCPM-V 4.7 partial rotary factors disagree")
        if not 1 <= config.num_experts_per_tok <= config.num_experts:
            raise ValueError("MiniCPM-V 4.7 routes to an invalid number of experts")
        return config


@dataclass(frozen=True)
class VisionConfig:
    """SigLIP-style tower with one window-attention 2x2 merger after layer insert_layer_id."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    insert_layer_id: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def patch_width(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @staticmethod
    def from_dict(raw: dict, insert_layer_id: int) -> VisionConfig:
        if raw.get("model_type") != "minicpmv4_7_vision":
            raise ValueError("MiniCPM-V 4.7 requires its minicpmv4_7_vision tower")
        _require(raw, "hidden_act", "gelu_pytorch_tanh")
        for name in ("window_kernel_size",):
            if list(raw.get(name, [2, 2])) != [2, 2]:
                raise ValueError(f"MiniCPM-V 4.7 family supports {name}=[2, 2]")
        config = VisionConfig(
            hidden_size=_int(raw, "hidden_size"),
            intermediate_size=_int(raw, "intermediate_size"),
            num_hidden_layers=_int(raw, "num_hidden_layers"),
            num_attention_heads=_int(raw, "num_attention_heads"),
            num_channels=_int(raw, "num_channels"),
            image_size=_int(raw, "image_size"),
            patch_size=_int(raw, "patch_size"),
            layer_norm_eps=_float(raw, "layer_norm_eps"),
            insert_layer_id=insert_layer_id,
        )
        if config.hidden_size % config.num_attention_heads or config.num_channels != 3:
            raise ValueError("MiniCPM-V 4.7 vision tower must have RGB input and whole attention heads")
        if not 0 <= insert_layer_id < config.num_hidden_layers:
            raise ValueError("MiniCPM-V 4.7 insert_layer_id must name a vision encoder layer")
        if config.image_size % config.patch_size:
            raise ValueError("MiniCPM-V 4.7 position grid must be a whole number of patches")
        return config


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    vision: VisionConfig
    image_token_id: int
    image_start_id: int
    image_end_id: int
    slice_start_id: int
    slice_end_id: int
    newline_id: int
    raw: dict

    @staticmethod
    def from_dict(raw: dict) -> ModelConfig:
        if raw.get("model_type") != "minicpmv4_7":
            raise ValueError(f"MiniCPM-V 4.7 does not support model_type={raw.get('model_type')!r}")
        if raw.get("architectures") != ["MiniCPMV4_7ForConditionalGeneration"]:
            raise ValueError("MiniCPM-V 4.7 requires MiniCPMV4_7ForConditionalGeneration")
        for name, expected in (("downsample_mode", "16x"), ("merger_times", 1), ("mrope_mode", "canvas"),
                               ("tie_word_embeddings", False)):
            _require(raw, name, expected)
        if list(raw.get("merge_kernel_size", [2, 2])) != [2, 2]:
            raise ValueError("MiniCPM-V 4.7 family supports merge_kernel_size=[2, 2]")
        text = raw.get("text_config")
        vision = raw.get("vision_config")
        if not isinstance(text, dict) or not isinstance(vision, dict):
            raise ValueError("MiniCPM-V 4.7 config requires text_config and vision_config objects")
        insert = raw.get("insert_layer_id", 6)
        if type(insert) is not int:
            raise ValueError("MiniCPM-V 4.7 insert_layer_id must be an integer")
        return ModelConfig(
            text=TextConfig.from_dict(text),
            vision=VisionConfig.from_dict(vision, insert),
            image_token_id=_int(raw, "image_token_id", minimum=0),
            image_start_id=_int(raw, "image_start_id", minimum=0),
            image_end_id=_int(raw, "image_end_id", minimum=0),
            slice_start_id=_int(raw, "slice_start_id", minimum=0),
            slice_end_id=_int(raw, "slice_end_id", minimum=0),
            newline_id=_int(raw, "newline_id", minimum=0),
            raw=raw,
        )

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        return ModelConfig.from_dict(json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")))
