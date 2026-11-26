# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunYuanVL checkpoint metadata: exact text, vision, RoPE and image-processor contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

ARCHITECTURE = "HunYuanVLForConditionalGeneration"
# Bounds the runtime's 64-bit pixel arithmetic (a 2^20 x 2^20 image).
MAX_PIXELS = 1 << 40


def load_json(path: Path):
    """Read a JSON file and reject an object that repeats a key (json.loads keeps the last)."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"HunYuanVL {Path(path).name} repeats the key {key!r}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def _positive(document: dict, name: str, owner: str) -> int:
    value = document.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"HunYuanVL {owner} {name} must be a positive integer")
    return value


def same_json(a, b) -> bool:
    """JSON equality that also compares types: false is not 0, and 3.0 is not 3."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_json(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return a == b


def _exactly(document: dict, name: str, expected, owner: str) -> None:
    """An optional setting that, when present, must be `expected` with the same JSON type."""
    if name in document and not same_json(document[name], expected):
        raise ValueError(f"HunYuanVL {owner} {name} must be {expected!r}")


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"HunYuanVL {name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"HunYuanVL {name} must be finite")
    return value


def _finite_positive(value, name: str) -> float:
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"HunYuanVL {name} must be finite and positive")
    return value


def _float32(value, name: str) -> float:
    """A number stored as float32 (the runtime's rule): finite and within float32's range."""
    value = _finite(value, name)
    if abs(value) > float(np.finfo(np.float32).max):
        raise ValueError(f"HunYuanVL {name} is outside the float32 range")
    return value


def _float32_positive(value, name: str) -> float:
    """A positive number stored as float32: it must stay a normal float32 value there."""
    value = _float32(value, name)
    with np.errstate(under="ignore"):
        stored = np.float32(value)
    if not stored >= np.finfo(np.float32).tiny:
        raise ValueError(f"HunYuanVL {name} must be a positive, normal float32 value")
    return value


def _bfloat16(value) -> np.float32:
    """A float32 rounded to bfloat16 (nearest, ties to even) as the runtime packs activations."""
    bits = int(np.float32(value).view(np.uint32))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return np.uint32(bits).view(np.float32)


def check_normalization(mean, std, rescale_factor: float, bfloat16: bool = False) -> None:
    """Fold rescaling into normalization the way the runtime does (float32 settings, a double
    reciprocal, float32 folded values, then (pixel - mean) / std in float32) and require finite
    results over the whole uint8 range, also once packed as bfloat16 when `bfloat16` is set."""
    inverse = 1.0 / rescale_factor
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for m, s in zip(mean, std):
            m32, s32 = np.float32(m), np.float32(s)
            folded_mean = np.float32(float(m32) * inverse)
            folded_std = np.float32(float(s32) * inverse)
            ends = [(np.float32(pixel) - folded_mean) / folded_std for pixel in (0.0, 255.0)]
            if (
                not math.isfinite(inverse)
                or not np.isfinite(m32)
                or not s32 >= np.finfo(np.float32).tiny
                or not np.isfinite(folded_mean)
                or not np.isfinite(folded_std)
                or not folded_std >= np.finfo(np.float32).tiny
                or not all(np.isfinite(end) for end in ends)
            ):
                raise ValueError(
                    "HunYuanVL image mean, std and rescale_factor overflow or underflow float32"
                )
            if bfloat16 and not all(np.isfinite(_bfloat16(end)) for end in ends):
                raise ValueError("HunYuanVL image mean, std and rescale_factor overflow bfloat16")


def _token(raw: dict, *names: str) -> int:
    values = [raw[name] for name in names if name in raw]
    if not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != 1:
        raise ValueError(f"HunYuanVL requires one consistent token id for {names[0]}")
    return values[0]


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
        return cls.from_dict(load_json(Path(model_dir) / "config.json"))

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
        # Attention and MLP are bias-free.
        _exactly(text, "attention_bias", False, "text")
        _exactly(text, "mlp_bias", False, "text")
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
        eps = _float32_positive(text.get("rms_norm_eps"), "rms_norm_eps")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("HunYuanVL requires text_config.rope_parameters")
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        sections = rope.get("mrope_section", rope.get("xdrope_section"))
        if (
            "mrope_section" in rope
            and "xdrope_section" in rope
            and not same_json(rope["mrope_section"], rope["xdrope_section"])
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
        if rope_type in ("xdrope", "dynamic"):
            alpha = _finite_positive(alpha, "rope alpha")
        elif rope_type != "default" or alpha is not None:
            raise ValueError("HunYuanVL supports default RoPE or fixed-alpha xdrope/dynamic RoPE")
        vision = raw.get("vision_config")
        if not isinstance(vision, dict) or vision.get("model_type") != "hunyuan_vl_vision":
            raise ValueError("HunYuanVL requires a hunyuan_vl_vision vision_config")
        if vision.get("hidden_act") != "gelu":
            raise ValueError("HunYuanVL vision requires the exact (erf) GELU")
        _exactly(vision, "interpolate_mode", "bilinear", "vision")
        _exactly(vision, "num_channels", 3, "vision")
        _exactly(vision, "temporal_patch_size", 1, "vision")
        width = _positive(vision, "hidden_size", "vision")
        heads = _positive(vision, "num_attention_heads", "vision")
        if width % heads:
            raise ValueError("HunYuanVL vision width must be divisible by its heads")
        _exactly(vision, "num_key_value_heads", heads, "vision")
        patch = _positive(vision, "patch_size", "vision")
        merge = _positive(vision, "spatial_merge_size", "vision")
        if merge != 2:
            raise ValueError(
                "HunYuanVL supports only spatial_merge_size 2 (the runtime's 2x2 merge)"
            )
        max_image = _positive(vision, "max_image_size", "vision")
        if max_image % patch:
            raise ValueError("HunYuanVL max_image_size must be a multiple of patch_size")
        _exactly(vision, "text_hidden_size", values["hidden_size"], "vision")
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
            vision_eps=_float32_positive(
                vision.get("rms_norm_eps", vision.get("layer_norm_eps")), "vision rms_norm_eps"
            ),
            patch_size=patch,
            merge_size=merge,
            position_edge=max_image // patch,
        )

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Every tensor the builder reads, with the shape this configuration implies."""
        h, inner, vocab, d = (
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
            self.head_dim,
        )
        q, kv = self.num_attention_heads * d, self.num_key_value_heads * d
        shapes = {
            "model.embed_tokens.weight": (vocab, h),
            "lm_head.weight": (vocab, h),
            "model.norm.weight": (h,),
        }
        for layer in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            for name, shape in (
                ("input_layernorm.weight", (h,)),
                ("post_attention_layernorm.weight", (h,)),
                ("self_attn.q_proj.weight", (q, h)),
                ("self_attn.k_proj.weight", (kv, h)),
                ("self_attn.v_proj.weight", (kv, h)),
                ("self_attn.o_proj.weight", (h, q)),
                ("self_attn.query_layernorm.weight", (d,)),
                ("self_attn.key_layernorm.weight", (d,)),
                ("mlp.gate_proj.weight", (inner, h)),
                ("mlp.up_proj.weight", (inner, h)),
                ("mlp.down_proj.weight", (h, inner)),
            ):
                shapes[prefix + name] = shape
        width, wide, m = self.vision_hidden_size, self.vision_intermediate_size, self.merge_size
        shapes.update(
            {
                "vit.embeddings.patch_embedding.weight": (
                    width,
                    3,
                    self.patch_size,
                    self.patch_size,
                ),
                "vit.embeddings.patch_embedding.bias": (width,),
                "vit.embeddings.position_embedding.weight": (1 + self.position_edge**2, width),
                "vit.perceive.before_rms.weight": (width,),
                "vit.perceive.proj.0.weight": (width * m, width, m, m),
                "vit.perceive.proj.0.bias": (width * m,),
                "vit.perceive.proj.2.weight": (width * m * m, width * m, 1, 1),
                "vit.perceive.proj.2.bias": (width * m * m,),
                "vit.perceive.image_newline": (width * m * m,),
                "vit.perceive.mlp.weight": (h, width * m * m),
                "vit.perceive.mlp.bias": (h,),
                "vit.perceive.image_begin": (h,),
                "vit.perceive.image_end": (h,),
                "vit.perceive.after_rms.weight": (h,),
            }
        )
        for layer in range(self.vision_layers):
            prefix = f"vit.layers.{layer}."
            for name, out, size in (
                ("input_layernorm", width, None),
                ("post_attention_layernorm", width, None),
                ("self_attn.q_proj", width, width),
                ("self_attn.k_proj", width, width),
                ("self_attn.v_proj", width, width),
                ("self_attn.o_proj", width, width),
                ("mlp.dense_h_to_4h", wide, width),
                ("mlp.dense_4h_to_h", width, wide),
            ):
                shapes[prefix + name + ".weight"] = (out,) if size is None else (out, size)
                shapes[prefix + name + ".bias"] = (out,)
        return shapes

    def inverse_frequencies(self) -> list[float]:
        """RoPE inverse frequencies, evaluated as the reference does (FP32 tensor arithmetic);
        each must be a normal float32 value for the runtime's angles to be finite and nonzero."""
        import torch

        dim = self.head_dim
        base = self.rope_theta
        try:
            if self.rope_alpha is not None:
                base = self.rope_theta * self.rope_alpha ** (dim / (dim - 2))
        except OverflowError:
            base = math.inf
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        values = [float(value) for value in inv_freq.tolist()]
        tiny = float(np.finfo(np.float32).tiny)
        if not all(math.isfinite(value) and value >= tiny for value in values):
            raise ValueError(
                "HunYuanVL rope_theta and alpha give RoPE frequencies outside the float32 range"
            )
        return values


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
        raw = load_json(Path(model_dir) / "preprocessor_config.json")
        _exactly(raw, "image_processor_type", "HunYuanVLImageProcessor", "image processor")
        for name in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            if raw.get(name, True) is not True:
                raise ValueError(f"HunYuanVL preprocessing requires {name}=true")
        # The geometry matches the vision tower.
        _exactly(raw, "patch_size", config.patch_size, "image processor")
        _exactly(raw, "merge_size", config.merge_size, "image processor")
        _exactly(raw, "temporal_patch_size", 1, "image processor")
        # `resample` is not read: the reference processor resizes bicubic whatever it says (the
        # released configuration has 1, LANCZOS), and so does the runtime.
        size = raw.get("size", {})
        if not isinstance(size, dict):
            raise ValueError("HunYuanVL image processor size must be an object")
        min_pixels = raw.get("min_pixels", size.get("shortest_edge"))
        max_pixels = raw.get("max_pixels", size.get("longest_edge"))
        if (
            type(min_pixels) is not int
            or type(max_pixels) is not int
            or not 0 < min_pixels <= max_pixels <= MAX_PIXELS
        ):
            raise ValueError(
                f"HunYuanVL image processor needs 0 < min_pixels <= max_pixels <= {MAX_PIXELS}"
            )
        mean, std = raw.get("image_mean"), raw.get("image_std")
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or len(mean) != 3
            or len(std) != 3
        ):
            raise ValueError("HunYuanVL image processor needs RGB mean and std")
        mean = tuple(_float32(v, "image_mean") for v in mean)
        std = tuple(_float32_positive(v, "image_std") for v in std)
        rescale_factor = _float32_positive(raw.get("rescale_factor", 1 / 255), "rescale_factor")
        check_normalization(mean, std, rescale_factor)
        return cls(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            mean=mean,
            std=std,
            rescale_factor=rescale_factor,
        )
