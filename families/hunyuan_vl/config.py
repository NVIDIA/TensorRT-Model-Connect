# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunYuanVL checkpoint metadata: exact text, vision, RoPE and image-processor contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

ARCHITECTURE = "HunYuanVLForConditionalGeneration"


def _positive(document: dict, name: str, owner: str) -> int:
    value = document.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"HunYuanVL {owner} {name} must be a positive integer")
    return value


def _finite_positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"HunYuanVL {name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"HunYuanVL {name} must be finite and positive")
    return value


def _token(raw: dict, *names: str) -> int:
    values = {raw[name] for name in names if name in raw}
    if len(values) != 1 or type(next(iter(values))) is not int or next(iter(values)) < 0:
        raise ValueError(f"HunYuanVL requires one consistent token id for {names[0]}")
    return next(iter(values))


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_alpha: float | None
    rope_sections: tuple[int, ...]
    image_token_id: int
    image_start_token_id: int
    image_end_token_id: int
    vision_hidden_size: int
    vision_intermediate_size: int
    vision_layers: int
    vision_heads: int
    vision_eps: float
    patch_size: int
    merge_size: int
    position_edge: int

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelConfig":
        return cls.from_dict(
            json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        )

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        if not isinstance(raw, dict) or raw.get("model_type") != "hunyuan_vl":
            raise ValueError("HunYuanVL requires model_type=hunyuan_vl")
        if raw.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"HunYuanVL requires architectures=[{ARCHITECTURE!r}]")
        text = raw.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") != "hunyuan_vl_text":
            raise ValueError("HunYuanVL requires a nested hunyuan_vl_text text_config")
        if text.get("hidden_act") != "silu":
            raise ValueError("HunYuanVL text MLP requires the gated SiLU activation")
        if text.get("attention_bias", False) or text.get("mlp_bias", False):
            raise ValueError("HunYuanVL text attention and MLP must be bias-free")
        if text.get("use_qk_norm", raw.get("use_qk_norm", True)) is not True:
            raise ValueError("HunYuanVL attention requires the per-head query/key RMSNorm")
        if raw.get("tie_word_embeddings", text.get("tie_word_embeddings")) is not True:
            raise ValueError("HunYuanVL requires the language head tied to the token embedding")
        names = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        )
        values = {name: _positive(text, name, "text") for name in names}
        if values["num_attention_heads"] % values["num_key_value_heads"]:
            raise ValueError("HunYuanVL attention heads must be divisible by key/value heads")
        if values["head_dim"] % 2:
            raise ValueError("HunYuanVL head_dim must be even")
        eps = _finite_positive(text.get("rms_norm_eps"), "rms_norm_eps")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("HunYuanVL requires text_config.rope_parameters")
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        sections = rope.get("mrope_section", rope.get("xdrope_section"))
        if (
            "mrope_section" in rope
            and "xdrope_section" in rope
            and list(rope["mrope_section"]) != list(rope["xdrope_section"])
        ):
            raise ValueError("HunYuanVL mrope_section and xdrope_section disagree")
        if (
            not isinstance(sections, list)
            or len(sections) < 3
            or any(type(n) is not int or n <= 0 for n in sections)
            or sum(sections) * 2 != values["head_dim"]
        ):
            raise ValueError(
                "HunYuanVL needs at least three positive RoPE sections covering head_dim / 2"
            )
        theta = _finite_positive(rope.get("rope_theta"), "rope_theta")
        alpha = rope.get("alpha")
        if rope_type in ("xdrope", "dynamic") and alpha:
            alpha = _finite_positive(alpha, "rope alpha")
        elif rope_type == "default" and not alpha:
            alpha = None
        else:
            raise ValueError("HunYuanVL supports default RoPE or fixed-alpha xdrope/dynamic RoPE")
        vision = raw.get("vision_config")
        if not isinstance(vision, dict) or vision.get("model_type") != "hunyuan_vl_vision":
            raise ValueError("HunYuanVL requires a hunyuan_vl_vision vision_config")
        if vision.get("hidden_act") != "gelu":
            raise ValueError("HunYuanVL vision requires the exact (erf) GELU")
        if vision.get("interpolate_mode", "bilinear") != "bilinear":
            raise ValueError("HunYuanVL vision positions require bilinear interpolation")
        if vision.get("num_channels", 3) != 3 or vision.get("temporal_patch_size", 1) != 1:
            raise ValueError("HunYuanVL vision requires RGB images with temporal_patch_size=1")
        width = _positive(vision, "hidden_size", "vision")
        heads = _positive(vision, "num_attention_heads", "vision")
        if width % heads or vision.get("num_key_value_heads", heads) != heads:
            raise ValueError("HunYuanVL vision attention requires equal query/key/value heads")
        patch = _positive(vision, "patch_size", "vision")
        merge = _positive(vision, "spatial_merge_size", "vision")
        max_image = _positive(vision, "max_image_size", "vision")
        if max_image % patch:
            raise ValueError("HunYuanVL max_image_size must be a multiple of patch_size")
        if vision.get("text_hidden_size", values["hidden_size"]) != values["hidden_size"]:
            raise ValueError("HunYuanVL vision projector width must equal the text hidden size")
        return cls(
            raw=raw,
            **values,
            rms_norm_eps=eps,
            rope_theta=theta,
            rope_alpha=alpha,
            rope_sections=tuple(sections),
            image_token_id=_token(raw, "image_token_id"),
            image_start_token_id=_token(raw, "im_start_id", "image_start_token_id"),
            image_end_token_id=_token(raw, "im_end_id", "image_end_token_id"),
            vision_hidden_size=width,
            vision_intermediate_size=_positive(vision, "intermediate_size", "vision"),
            vision_layers=_positive(vision, "num_hidden_layers", "vision"),
            vision_heads=heads,
            vision_eps=_finite_positive(
                vision.get("rms_norm_eps", vision.get("layer_norm_eps")), "vision rms_norm_eps"
            ),
            patch_size=patch,
            merge_size=merge,
            position_edge=max_image // patch,
        )

    def inverse_frequencies(self) -> list[float]:
        """RoPE inverse frequencies, evaluated as the reference does (FP32 tensor arithmetic)."""
        import torch

        dim = self.head_dim
        base = self.rope_theta
        if self.rope_alpha is not None:
            base = self.rope_theta * self.rope_alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return [float(value) for value in inv_freq.tolist()]


@dataclass(frozen=True)
class ImageProcessor:
    """The effective HunYuanVL image processor settings (legacy pixel keys override ``size``)."""

    min_pixels: int
    max_pixels: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale_factor: float

    @classmethod
    def from_dir(cls, model_dir: Path, config: ModelConfig) -> "ImageProcessor":
        raw = json.loads((Path(model_dir) / "preprocessor_config.json").read_text(encoding="utf-8"))
        if raw.get("image_processor_type") not in (None, "HunYuanVLImageProcessor"):
            raise ValueError("HunYuanVL requires HunYuanVLImageProcessor preprocessing")
        for name in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            if raw.get(name, True) is not True:
                raise ValueError(f"HunYuanVL preprocessing requires {name}=true")
        if (
            raw.get("patch_size", config.patch_size) != config.patch_size
            or raw.get("merge_size", config.merge_size) != config.merge_size
            or raw.get("temporal_patch_size", 1) != 1
        ):
            raise ValueError("HunYuanVL image processor geometry must match the vision tower")
        size = raw.get("size") or {}
        min_pixels = raw.get("min_pixels", size.get("shortest_edge"))
        max_pixels = raw.get("max_pixels", size.get("longest_edge"))
        if (
            type(min_pixels) is not int
            or type(max_pixels) is not int
            or not 0 < min_pixels <= max_pixels
        ):
            raise ValueError("HunYuanVL image processor needs 0 < min_pixels <= max_pixels")
        mean, std = raw.get("image_mean"), raw.get("image_std")
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or len(mean) != 3
            or len(std) != 3
        ):
            raise ValueError("HunYuanVL image processor needs RGB mean and std")
        return cls(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            mean=tuple(float(v) for v in mean),
            std=tuple(_finite_positive(v, "image_std") for v in std),
            rescale_factor=_finite_positive(raw.get("rescale_factor", 1 / 255), "rescale_factor"),
        )
