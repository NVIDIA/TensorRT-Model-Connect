# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma checkpoint configuration as the transformers 5.2.0 reference resolves it.

A key the checkpoint omits takes the constructor default of the reference
``GemmaConfig``, ``SiglipVisionConfig`` or ``SiglipImageProcessor``. A value the
family graph does not implement is rejected, never ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

_GEMMA_DEFAULTS = {
    "vocab_size": 256000,
    "hidden_size": 3072,
    "intermediate_size": 24576,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
    "attention_bias": False,
}
_SIGLIP_DEFAULTS = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
}
_DEFAULT_ROPE_THETA = 10_000.0
_BICUBIC = 3
# Generation settings that leave greedy decoding to the EOS ids unchanged, and
# decoding settings that are neutral at the listed value.
_NEUTRAL_GENERATION_KEYS = frozenset({
    "_from_model_config", "bos_token_id", "eos_token_id", "pad_token_id",
    "transformers_version", "max_length", "cache_implementation", "use_cache",
    "output_attentions", "output_hidden_states",
})
_GREEDY_GENERATION_VALUES = {
    "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
    "min_length": 0, "min_new_tokens": None,
}


def _read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"PaliGemma {path.name} must be a JSON object")
    return document


def _positive(document: dict, defaults: dict, name: str, owner: str) -> int:
    value = document.get(name, defaults.get(name))
    if type(value) is not int or value <= 0:
        raise ValueError(f"PaliGemma {owner} {name} must be a positive integer")
    return value


def _positive_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"PaliGemma {name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"PaliGemma {name} must be finite and positive")
    return value


def _token_ids(value, name: str) -> tuple[int, ...]:
    values = value if isinstance(value, list) else [value]
    if not values or any(type(item) is not int or item < 0 for item in values):
        raise ValueError(f"PaliGemma {name} must be a non-negative id or a list of them")
    return tuple(values)


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid * self.grid


@dataclass(frozen=True)
class TextConfig:
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


@dataclass(frozen=True)
class ImageProcessing:
    """SiglipImageProcessor: Pillow bicubic squash, rescale and normalize; no crop."""

    height: int
    width: int
    rescale_factor: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    vision: VisionConfig
    text: TextConfig
    image_token_id: int
    num_image_tokens: int

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelConfig":
        return cls.from_dict(_read_json(Path(model_dir) / "config.json"))

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        if not isinstance(raw, dict) or raw.get("model_type") != "paligemma":
            raise ValueError("expected a paligemma checkpoint configuration")
        if raw.get("tie_word_embeddings", True) is not True:
            raise ValueError("PaliGemma requires the language-model head tied to the token embedding")
        return cls(raw=raw, vision=_vision(raw), text=_text(raw),
                   image_token_id=_image_token(raw), num_image_tokens=_image_tokens(raw))


def _vision(raw: dict) -> VisionConfig:
    vision = raw.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("PaliGemma requires a vision_config")
    if vision.get("model_type", "siglip_vision_model") != "siglip_vision_model":
        raise ValueError("PaliGemma requires the SigLIP vision tower")
    if vision.get("hidden_act", _SIGLIP_DEFAULTS["hidden_act"]) != "gelu_pytorch_tanh":
        raise ValueError("PaliGemma vision MLP requires tanh-approximated GELU")
    names = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_channels", "image_size", "patch_size")
    values = {name: _positive(vision, _SIGLIP_DEFAULTS, name, "vision") for name in names}
    if values["hidden_size"] % values["num_attention_heads"]:
        raise ValueError("PaliGemma vision heads must divide the vision width")
    if values["image_size"] % values["patch_size"]:
        raise ValueError("PaliGemma vision patches must tile the image")
    if values["num_channels"] != 3:
        raise ValueError("PaliGemma vision tower must take RGB images")
    eps = _positive_float(vision.get("layer_norm_eps", _SIGLIP_DEFAULTS["layer_norm_eps"]),
                          "vision layer_norm_eps")
    return VisionConfig(**values, layer_norm_eps=eps)


def _text(raw: dict) -> TextConfig:
    text = raw.get("text_config")
    if not isinstance(text, dict) or text.get("model_type", "gemma") != "gemma":
        raise ValueError("PaliGemma requires the Gemma text_config")
    if text.get("hidden_act", _GEMMA_DEFAULTS["hidden_act"]) != "gelu_pytorch_tanh":
        raise ValueError("PaliGemma Gemma MLP requires tanh-approximated GELU")
    if text.get("attention_bias", False) is not False:
        raise ValueError("PaliGemma Gemma attention is bias-free")
    names = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings")
    values = {name: _positive(text, _GEMMA_DEFAULTS, name, "text") for name in names}
    if values["num_attention_heads"] % values["num_key_value_heads"]:
        raise ValueError("PaliGemma attention heads must be divisible by KV heads")
    if values["head_dim"] % 2:
        raise ValueError("PaliGemma rotary head dimension must be even")
    eps = _positive_float(text.get("rms_norm_eps", _GEMMA_DEFAULTS["rms_norm_eps"]),
                          "text rms_norm_eps")
    return TextConfig(**values, rms_norm_eps=eps, rope_theta=_rope_theta(text))


def _rope_theta(text: dict) -> float:
    rope = text.get("rope_parameters")
    if rope is None:
        if text.get("rope_scaling") is not None:
            raise ValueError("PaliGemma Gemma uses unscaled RoPE")
        return _positive_float(text.get("rope_theta", _DEFAULT_ROPE_THETA), "rope_theta")
    if not isinstance(rope, dict) or rope.get("rope_type", "default") != "default":
        raise ValueError("PaliGemma Gemma uses unscaled default RoPE")
    if set(rope) - {"rope_type", "rope_theta"}:
        raise ValueError("PaliGemma Gemma RoPE carries no scaling parameters")
    return _positive_float(rope.get("rope_theta", text.get("rope_theta", _DEFAULT_ROPE_THETA)),
                           "rope_theta")


def _image_token(raw: dict) -> int:
    value = raw.get("image_token_index", raw.get("image_token_id"))
    if type(value) is not int or value < 0:
        raise ValueError("PaliGemma requires a non-negative image_token_index")
    return value


def _image_tokens(raw: dict) -> int:
    vision = _vision(raw)
    text = _text(raw)
    projection = raw.get("projection_dim", raw["vision_config"].get("projection_dim", 2048))
    if projection != text.hidden_size:
        raise ValueError("PaliGemma projector width must equal the Gemma width")
    stored = raw["text_config"].get("num_image_tokens", vision.num_patches)
    if stored != vision.num_patches:
        raise ValueError("PaliGemma num_image_tokens must equal the vision patch count")
    if _image_token(raw) >= text.vocab_size:
        raise ValueError("PaliGemma image token must lie inside the embedding table")
    return vision.num_patches


def image_processing(model_dir: Path, vision: VisionConfig, num_image_tokens: int) -> ImageProcessing:
    """Resolve the image processor; reject anything other than the SigLIP square squash.

    transformers 4.x saves it as preprocessor_config.json, 5.x inside processor_config.json.
    """
    legacy = Path(model_dir) / "preprocessor_config.json"
    if legacy.is_file():
        document = _read_json(legacy)
    else:
        document = _read_json(Path(model_dir) / "processor_config.json").get("image_processor")
        if not isinstance(document, dict):
            raise ValueError("PaliGemma processor_config.json has no image_processor")
    if document.get("image_processor_type", "SiglipImageProcessor") != "SiglipImageProcessor":
        raise ValueError("PaliGemma requires the SigLIP image processor")
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if document.get(flag, True) is not True:
            raise ValueError(f"PaliGemma preprocessing requires {flag}")
    if document.get("resample", _BICUBIC) != _BICUBIC:
        raise ValueError("PaliGemma preprocessing is qualified for bicubic resampling only")
    size = document.get("size", {"height": 224, "width": 224})
    if (not isinstance(size, dict) or set(size) != {"height", "width"}
            or size["height"] != vision.image_size or size["width"] != vision.image_size):
        raise ValueError("PaliGemma preprocessing must resize to the vision image size")
    if document.get("image_seq_length", num_image_tokens) != num_image_tokens:
        raise ValueError("PaliGemma image_seq_length must equal the vision patch count")
    scale = _positive_float(document.get("rescale_factor", 1 / 255), "rescale_factor")
    channels = []
    for name in ("image_mean", "image_std"):
        values = document.get(name, [0.5, 0.5, 0.5])
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"PaliGemma {name} must list three channels")
        channels.append(tuple(float(value) for value in values))
    if any(value == 0 or not math.isfinite(value) for value in channels[1]):
        raise ValueError("PaliGemma image_std must be finite and non-zero")
    return ImageProcessing(height=size["height"], width=size["width"], rescale_factor=scale,
                           mean=channels[0], std=channels[1])


def eos_token_ids(model_dir: Path, raw: dict) -> tuple[int, ...]:
    """EOS ids greedy generation stops at, as the reference generation config resolves them."""
    path = Path(model_dir) / "generation_config.json"
    generation = _read_json(path) if path.is_file() else {}
    unsupported = sorted(
        key for key, value in generation.items() if key not in _NEUTRAL_GENERATION_KEYS
        and (key not in _GREEDY_GENERATION_VALUES or value != _GREEDY_GENERATION_VALUES[key]))
    if unsupported:
        raise ValueError(f"PaliGemma generation_config.json changes greedy decoding: {unsupported}")
    value = generation.get("eos_token_id")
    if value is None:
        value = raw.get("eos_token_id", raw.get("text_config", {}).get("eos_token_id", 1))
    return _token_ids(value, "eos_token_id")
