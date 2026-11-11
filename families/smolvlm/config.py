# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLM checkpoint metadata: model, processor, chat template and generation files.

Absent keys take the defaults of the Hugging Face ``SmolVLMVisionConfig`` and
``LlamaConfig`` classes, so the graph is built from the same values the reference
model uses. Anything the native runtime does not implement is rejected here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

# The checkpoint's chat template. The native runtime renders exactly this template
# (role names, the ':' / ': ' rule after the role, one '<end_of_utterance>\n' per
# message and the 'Assistant:' generation prompt) instead of evaluating Jinja.
CHAT_TEMPLATE = (
    "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}"
    "{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}"
    "{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}"
    "<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

# PIL.Image.Resampling.LANCZOS, the only filter the native image path implements.
LANCZOS = 1

SPECIAL_TOKENS = {
    "image_token": "<image>",
    "fake_image_token": "<fake_token_around_image>",
    "global_image_token": "<global-img>",
    "end_of_utterance_token": "<end_of_utterance>",
    "bos_token": "<|im_start|>",
}


def _read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"SmolVLM {path.name} must contain a JSON object")
    return document


def _positive(document: dict, name: str, default=None) -> int:
    value = document.get(name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"SmolVLM {name} must be a positive integer")
    return value


def _finite_positive(document: dict, name: str, default=None) -> float:
    value = document.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SmolVLM {name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"SmolVLM {name} must be finite and positive")
    return value


def _rope_theta(text: dict) -> float:
    parameters = text.get("rope_parameters")
    if parameters is not None:
        if not isinstance(parameters, dict) or parameters.get("rope_type", "default") != "default":
            raise ValueError("SmolVLM requires default (unscaled) RoPE")
        if parameters.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("SmolVLM requires full-width RoPE")
        return _finite_positive(parameters, "rope_theta", 10000.0)
    if text.get("rope_scaling") is not None:
        raise ValueError("SmolVLM requires default (unscaled) RoPE")
    if text.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("SmolVLM requires full-width RoPE")
    return _finite_positive(text, "rope_theta", 10000.0)


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
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size


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
    longest_edge: int
    tile_size: int
    rescale_factor: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    image_seq_len: int

    def as_json(self) -> dict:
        return {
            "longest_edge": self.longest_edge,
            "tile_size": self.tile_size,
            "rescale_factor": self.rescale_factor,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "image_seq_len": self.image_seq_len,
        }


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    vision: VisionConfig
    text: TextConfig
    scale_factor: int
    image_token_id: int
    processing: ImageProcessing
    eos_token_id: int

    @property
    def image_tokens_per_tile(self) -> int:
        return self.vision.patches_per_side ** 2 // self.scale_factor ** 2

    @property
    def connector_width(self) -> int:
        return self.vision.hidden_size * self.scale_factor ** 2

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelConfig":
        root = Path(model_dir)
        raw = _read_json(root / "config.json")
        preprocessor = _read_json(root / "preprocessor_config.json")
        processor = _read_json(root / "processor_config.json")
        generation = _read_json(root / "generation_config.json")
        template = _read_json(root / "chat_template.json").get("chat_template")
        if template != CHAT_TEMPLATE:
            raise ValueError("SmolVLM chat_template.json differs from the template the runtime renders")
        tokenizer = _read_json(root / "tokenizer_config.json")
        for key, token in SPECIAL_TOKENS.items():
            if tokenizer.get(key, token) != token:
                raise ValueError(f"SmolVLM tokenizer {key} must be {token!r}")
        return cls.from_dicts(raw, preprocessor, processor, generation)

    @classmethod
    def from_dicts(cls, raw: dict, preprocessor: dict, processor: dict, generation: dict) -> "ModelConfig":
        if raw.get("model_type") != "smolvlm":
            raise ValueError("expected a SmolVLM checkpoint (model_type 'smolvlm')")
        if raw.get("architectures", ["SmolVLMForConditionalGeneration"]) != ["SmolVLMForConditionalGeneration"]:
            raise ValueError("SmolVLM requires the SmolVLMForConditionalGeneration head")
        if raw.get("tie_word_embeddings", False) is not False:
            raise ValueError("SmolVLM requires its separate lm_head weight (tie_word_embeddings false)")
        if raw.get("quantization_config") is not None:
            raise ValueError("SmolVLM expects original, unquantized checkpoint weights")
        vision_raw, text_raw = raw.get("vision_config"), raw.get("text_config")
        if not isinstance(vision_raw, dict) or not isinstance(text_raw, dict):
            raise ValueError("SmolVLM requires vision_config and text_config")

        if vision_raw.get("model_type", "smolvlm_vision") != "smolvlm_vision":
            raise ValueError("SmolVLM requires the smolvlm_vision tower")
        if vision_raw.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("SmolVLM vision MLP requires tanh GELU")
        vision = VisionConfig(
            hidden_size=_positive(vision_raw, "hidden_size", 1152),
            intermediate_size=_positive(vision_raw, "intermediate_size", 3072),
            num_hidden_layers=_positive(vision_raw, "num_hidden_layers", 12),
            num_attention_heads=_positive(vision_raw, "num_attention_heads", 16),
            num_channels=_positive(vision_raw, "num_channels", 3),
            image_size=_positive(vision_raw, "image_size", 224),
            patch_size=_positive(vision_raw, "patch_size", 32),
            layer_norm_eps=_finite_positive(vision_raw, "layer_norm_eps", 1e-6),
        )
        if vision.num_channels != 3:
            raise ValueError("SmolVLM vision tower must take RGB input")
        if vision.hidden_size % vision.num_attention_heads:
            raise ValueError("SmolVLM vision heads must divide the vision width")
        if vision.image_size % vision.patch_size:
            raise ValueError("SmolVLM vision image_size must be a multiple of patch_size")

        if text_raw.get("model_type", "llama") != "llama":
            raise ValueError("SmolVLM requires a Llama text decoder")
        if text_raw.get("hidden_act", "silu") != "silu":
            raise ValueError("SmolVLM text MLP requires SiLU")
        for flag in ("attention_bias", "mlp_bias", "use_resampler", "qk_layer_norms", "rope_interleaved"):
            if text_raw.get(flag, False):
                raise ValueError(f"SmolVLM text decoder does not support {flag}")
        if text_raw.get("pretraining_tp", 1) != 1:
            raise ValueError("SmolVLM text decoder requires pretraining_tp 1")
        hidden = _positive(text_raw, "hidden_size", 4096)
        heads = _positive(text_raw, "num_attention_heads", 32)
        text = TextConfig(
            hidden_size=hidden,
            intermediate_size=_positive(text_raw, "intermediate_size", 11008),
            num_hidden_layers=_positive(text_raw, "num_hidden_layers", 32),
            num_attention_heads=heads,
            num_key_value_heads=_positive(text_raw, "num_key_value_heads", heads),
            head_dim=_positive(text_raw, "head_dim", hidden // heads),
            vocab_size=_positive(text_raw, "vocab_size", 32000),
            max_position_embeddings=_positive(text_raw, "max_position_embeddings", 2048),
            rms_norm_eps=_finite_positive(text_raw, "rms_norm_eps", 1e-6),
            rope_theta=_rope_theta(text_raw),
        )
        if text.num_attention_heads % text.num_key_value_heads:
            raise ValueError("SmolVLM attention heads must be a multiple of the KV heads")
        if text.head_dim % 2:
            raise ValueError("SmolVLM RoPE requires an even head_dim")

        scale_factor = _positive(raw, "scale_factor", 2)
        if vision.patches_per_side % scale_factor:
            raise ValueError("SmolVLM pixel shuffle must divide the patch grid")
        image_token_id = raw.get("image_token_id")
        if type(image_token_id) is not int or not 0 <= image_token_id < text.vocab_size:
            raise ValueError("SmolVLM image_token_id must be a token of the text vocabulary")
        eos = generation.get("eos_token_id")
        if type(eos) is not int or not 0 <= eos < text.vocab_size:
            raise ValueError("SmolVLM generation_config.json needs one integer eos_token_id")
        for key in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty"):
            if key in generation:
                raise ValueError(f"SmolVLM generation default {key} is not implemented by the runtime")
        processing = _processing(preprocessor, processor, vision, scale_factor)
        return cls(raw=raw, vision=vision, text=text, scale_factor=scale_factor,
                   image_token_id=image_token_id, processing=processing, eos_token_id=eos)


def _processing(preprocessor: dict, processor: dict, vision: VisionConfig, scale_factor: int) -> ImageProcessing:
    if preprocessor.get("image_processor_type") != "SmolVLMImageProcessor":
        raise ValueError("SmolVLM requires the SmolVLMImageProcessor contract")
    for flag in ("do_convert_rgb", "do_resize", "do_image_splitting", "do_rescale", "do_normalize"):
        if preprocessor.get(flag) is not True:
            raise ValueError(f"SmolVLM image preprocessing requires {flag} true")
    if preprocessor.get("resample") != LANCZOS:
        raise ValueError("SmolVLM image preprocessing requires Lanczos resampling")
    size, tile = preprocessor.get("size"), preprocessor.get("max_image_size")
    if not isinstance(size, dict) or set(size) != {"longest_edge"}:
        raise ValueError("SmolVLM preprocessing size must be {'longest_edge': N}")
    if not isinstance(tile, dict) or set(tile) != {"longest_edge"}:
        raise ValueError("SmolVLM max_image_size must be {'longest_edge': N}")
    longest_edge, tile_size = _positive(size, "longest_edge"), _positive(tile, "longest_edge")
    if tile_size != vision.image_size:
        raise ValueError("SmolVLM tiles must match the vision tower's image_size")
    if longest_edge > 4096:
        raise ValueError("SmolVLM longest_edge exceeds the 4096-pixel processor bound")
    mean, std = preprocessor.get("image_mean"), preprocessor.get("image_std")
    for name, values in (("image_mean", mean), ("image_std", std)):
        if (not isinstance(values, list) or len(values) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in values)):
            raise ValueError(f"SmolVLM {name} must hold three finite numbers")
    if any(v <= 0 for v in std):
        raise ValueError("SmolVLM image_std must be positive")
    image_seq_len = _positive(processor, "image_seq_len")
    if image_seq_len != vision.patches_per_side ** 2 // scale_factor ** 2:
        raise ValueError("SmolVLM image_seq_len does not match the vision grid and pixel shuffle")
    return ImageProcessing(
        longest_edge=longest_edge,
        tile_size=tile_size,
        rescale_factor=_finite_positive(preprocessor, "rescale_factor"),
        image_mean=tuple(float(v) for v in mean),
        image_std=tuple(float(v) for v in std),
        image_seq_len=image_seq_len,
    )
