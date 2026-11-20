# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint configuration for the LightOn OCR image-to-text lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _positive(config: dict, name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"lighton_ocr requires a positive integer {name}")
    return value


def _positive_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"lighton_ocr requires a finite positive {name}")


@dataclass(frozen=True)
class Config:
    raw: dict
    text_config: dict
    vision_config: dict
    spatial_merge_size: int
    image_token_id: int
    eos_token_id: int
    pad_token_id: int

    @classmethod
    def from_dir(cls, directory: str | Path) -> "Config":
        return cls.from_dict(json.loads((Path(directory) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        architectures = raw.get("architectures", [])
        if not set(architectures) & {"LightOnOCRForConditionalGeneration", "LightOnOcrForConditionalGeneration"}:
            raise ValueError("lighton_ocr requires a LightOn OCR checkpoint architecture")
        text, vision = raw.get("text_config", {}), raw.get("vision_config", {})
        if text.get("model_type") != "qwen3" or vision.get("model_type") != "pixtral":
            raise ValueError("lighton_ocr requires its Qwen3 decoder and Pixtral vision encoder")
        for value in (text, vision):
            for name in ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "head_dim"):
                _positive(value, name)
            if value.get("hidden_act") != "silu":
                raise ValueError("lighton_ocr requires the checkpoint's SiLU feedforward activation")
            parameters = value.get("rope_parameters", {})
            if parameters.get("rope_type", "default") != "default":
                raise ValueError("lighton_ocr supports the checkpoint's default rotary embedding")
            _positive_scalar(parameters.get("rope_theta", value.get("rope_theta", 0)), "rotary embedding base")
        for name in ("num_key_value_heads", "max_position_embeddings", "vocab_size"):
            _positive(text, name)
        if text["num_attention_heads"] % text["num_key_value_heads"] or text["head_dim"] % 2:
            raise ValueError("lighton_ocr requires divisible grouped-query heads and an even decoder head dimension")
        if text.get("attention_bias", False) or not text.get("tie_word_embeddings", True):
            raise ValueError("lighton_ocr requires bias-free attention and tied language embeddings")
        if text.get("use_qk_norm", True) is not True or text.get("use_sliding_window", False):
            raise ValueError("lighton_ocr requires Q/K normalization and full decoder attention")
        layers = text.get("layer_types", ["full_attention"] * text["num_hidden_layers"])
        if len(layers) != text["num_hidden_layers"] or set(layers) != {"full_attention"}:
            raise ValueError("lighton_ocr requires full attention in every decoder layer")
        _positive_scalar(text.get("rms_norm_eps", 0), "RMS normalization epsilon")
        for name in ("patch_size", "image_size", "num_channels"):
            _positive(vision, name)
        if vision["num_channels"] != 3 or vision["hidden_size"] != vision["num_attention_heads"] * vision["head_dim"]:
            raise ValueError("lighton_ocr requires RGB input and the declared Pixtral head geometry")
        if vision["head_dim"] % 4 or vision["image_size"] % vision["patch_size"]:
            raise ValueError("lighton_ocr requires axial rotary heads and patch-aligned maximum image size")
        merge = _positive(raw, "spatial_merge_size")
        if vision["image_size"] % (vision["patch_size"] * merge):
            raise ValueError("lighton_ocr maximum image profile must align to complete merged patches")
        if raw.get("vision_feature_layer", -1) != -1:
            raise ValueError("lighton_ocr consumes the final vision encoder layer")
        if raw.get("projector_hidden_act", "gelu") != "gelu" or raw.get("multimodal_projector_bias", False):
            raise ValueError("lighton_ocr requires the bias-free exact GELU projector")
        ids = {}
        for name in ("image_token_id", "eos_token_id", "pad_token_id"):
            value = raw.get(name)
            if type(value) is not int or not 0 <= value < text["vocab_size"]:
                raise ValueError(f"lighton_ocr requires an in-vocabulary integer {name}")
            ids[name] = value
        return cls(raw, text, vision, merge, **ids)

    @property
    def maximum_image_tokens(self) -> int:
        side = self.vision_config["image_size"] // (self.vision_config["patch_size"] * self.spatial_merge_size)
        return side * side
