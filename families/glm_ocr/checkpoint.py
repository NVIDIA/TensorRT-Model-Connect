# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-OCR checkpoint contract: configuration, processor, tokenizer facts and weights.

Every value the graphs or the native runtime depend on is read here and checked
against the one layout this family implements. A checkpoint that differs (another
rotary scheme, tied embeddings, attention biases on the decoder, a different
image processor) is rejected instead of being built with assumptions that no
longer hold.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 with NumPy for safetensors)
import numpy as np
from safetensors import safe_open

_MODEL_TYPE = "glm_ocr"
_ARCHITECTURE = "GlmOcrForConditionalGeneration"
_IMAGE_PROCESSOR = "Glm46VImageProcessor"
_BICUBIC = 3

# Token strings the native prompt renderer emits; their ids come from tokenizer.json.
PROMPT_TOKENS = (
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|image|>",
)


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    out_hidden_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    in_channels: int
    rms_norm_eps: float
    rope_theta: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merge_area(self) -> int:
        return self.spatial_merge_size * self.spatial_merge_size


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: tuple[int, ...]
    num_nextn_predict_layers: int


@dataclass(frozen=True)
class ProcessorConfig:
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float


@dataclass(frozen=True)
class GlmOcrConfig:
    text: TextConfig
    vision: VisionConfig
    processor: ProcessorConfig
    image_token_id: int
    image_start_token_id: int
    image_end_token_id: int
    eos_token_ids: tuple[int, ...]
    prompt_token_ids: dict[str, int]
    special_token_ids: tuple[int, ...]

    @property
    def max_image_patches(self) -> int:
        """Patches of the largest image the processor emits (its pixel budget covers both frames)."""
        p = self.processor
        return p.max_pixels // (p.temporal_patch_size * p.patch_size * p.patch_size)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"glm_ocr checkpoint is missing {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"glm_ocr expects a JSON object in {path.name}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"glm_ocr: {message}")


def _positive_int(section: dict, key: str, where: str) -> int:
    value = section.get(key)
    _require(isinstance(value, int) and value > 0, f"{where}.{key} must be a positive integer")
    return int(value)


def _vision_config(raw: dict) -> VisionConfig:
    v = raw.get("vision_config")
    _require(isinstance(v, dict), "config.json has no vision_config")
    rope = v.get("rope_parameters") or {}
    _require(v.get("model_type") == "glm_ocr_vision", "vision_config.model_type must be glm_ocr_vision")
    _require(v.get("hidden_act") == "silu", "vision MLP must use silu")
    _require(v.get("attention_bias") is True, "vision attention must carry biases")
    _require(rope.get("rope_type") == "axial", "vision rotary embedding must be axial")
    config = VisionConfig(
        depth=_positive_int(v, "depth", "vision_config"),
        hidden_size=_positive_int(v, "hidden_size", "vision_config"),
        num_heads=_positive_int(v, "num_heads", "vision_config"),
        intermediate_size=_positive_int(v, "intermediate_size", "vision_config"),
        out_hidden_size=_positive_int(v, "out_hidden_size", "vision_config"),
        patch_size=_positive_int(v, "patch_size", "vision_config"),
        temporal_patch_size=_positive_int(v, "temporal_patch_size", "vision_config"),
        spatial_merge_size=_positive_int(v, "spatial_merge_size", "vision_config"),
        in_channels=_positive_int(v, "in_channels", "vision_config"),
        rms_norm_eps=float(v.get("rms_norm_eps", 0.0)),
        rope_theta=float(rope.get("rope_theta", 0.0)),
    )
    _require(config.in_channels == 3, "vision_config.in_channels must be 3")
    _require(config.hidden_size % config.num_heads == 0, "vision heads must divide hidden_size")
    _require(config.head_dim % 4 == 0, "vision head_dim must split into four axial rotary parts")
    _require(config.rms_norm_eps > 0.0 and config.rope_theta > 0.0, "vision eps and rope_theta")
    return config


def _text_config(raw: dict) -> TextConfig:
    t = raw.get("text_config")
    _require(isinstance(t, dict), "config.json has no text_config")
    rope = t.get("rope_parameters") or {}
    _require(t.get("model_type") == "glm_ocr_text", "text_config.model_type must be glm_ocr_text")
    _require(t.get("hidden_act") == "silu", "decoder MLP must use silu")
    _require(t.get("attention_bias") is False, "decoder attention must not carry biases")
    _require(rope.get("rope_type") == "default", "decoder rotary embedding must be the default type")
    _require(float(rope.get("partial_rotary_factor", 1.0)) == 1.0, "decoder rotary must be full")
    _require(not t.get("tie_word_embeddings", False) and not raw.get("tie_word_embeddings", False),
             "lm_head must not be tied to the token embedding")
    section = rope.get("mrope_section")
    _require(isinstance(section, list) and len(section) == 3 and all(
        isinstance(item, int) and item > 0 for item in section), "mrope_section must be three sizes")
    config = TextConfig(
        hidden_size=_positive_int(t, "hidden_size", "text_config"),
        intermediate_size=_positive_int(t, "intermediate_size", "text_config"),
        num_hidden_layers=_positive_int(t, "num_hidden_layers", "text_config"),
        num_attention_heads=_positive_int(t, "num_attention_heads", "text_config"),
        num_key_value_heads=_positive_int(t, "num_key_value_heads", "text_config"),
        head_dim=_positive_int(t, "head_dim", "text_config"),
        vocab_size=_positive_int(t, "vocab_size", "text_config"),
        rms_norm_eps=float(t.get("rms_norm_eps", 0.0)),
        rope_theta=float(rope.get("rope_theta", 0.0)),
        mrope_section=tuple(int(item) for item in section),
        num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
    )
    _require(config.num_attention_heads % config.num_key_value_heads == 0,
             "key/value heads must divide the attention heads")
    _require(sum(config.mrope_section) * 2 == config.head_dim,
             "mrope_section must cover every rotary pair of head_dim")
    _require(config.rms_norm_eps > 0.0 and config.rope_theta > 0.0, "decoder eps and rope_theta")
    return config


def _processor_config(model_dir: Path, vision: VisionConfig) -> ProcessorConfig:
    combined = model_dir / "processor_config.json"
    if combined.is_file():
        p = _read_json(combined).get("image_processor")
        _require(isinstance(p, dict), "processor_config.json has no image_processor")
    else:
        p = _read_json(model_dir / "preprocessor_config.json")
    _require(p.get("image_processor_type") == _IMAGE_PROCESSOR,
             f"image processor must be {_IMAGE_PROCESSOR}")
    for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
        _require(p.get(flag, True) is True, f"image processor must keep {flag}=true")
    _require(int(p.get("resample", _BICUBIC)) == _BICUBIC, "image processor must resize bicubic")
    rescale = float(p.get("rescale_factor", 1.0 / 255.0))
    _require(math.isclose(rescale, 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-12),
             "image processor must rescale by 1/255")
    size = p.get("size") or {}
    mean, std = p.get("image_mean"), p.get("image_std")
    _require(isinstance(mean, list) and len(mean) == 3 and isinstance(std, list) and len(std) == 3,
             "image_mean and image_std must have three channels")
    config = ProcessorConfig(
        patch_size=int(p.get("patch_size", 0)),
        temporal_patch_size=int(p.get("temporal_patch_size", 0)),
        merge_size=int(p.get("merge_size", 0)),
        min_pixels=int(size.get("shortest_edge", 0)),
        max_pixels=int(size.get("longest_edge", 0)),
        image_mean=tuple(float(value) for value in mean),
        image_std=tuple(float(value) for value in std),
        rescale_factor=rescale,
    )
    _require(config.patch_size == vision.patch_size, "processor and vision patch sizes differ")
    _require(config.temporal_patch_size == vision.temporal_patch_size,
             "processor and vision temporal patch sizes differ")
    _require(config.merge_size == vision.spatial_merge_size, "processor and vision merge sizes differ")
    _require(0 < config.min_pixels < config.max_pixels, "processor pixel budget is invalid")
    _require(all(value > 0.0 for value in config.image_std), "image_std must be positive")
    return config


def _tokenizer_facts(model_dir: Path) -> tuple[dict[str, int], tuple[int, ...]]:
    tokenizer = _read_json(model_dir / "tokenizer.json")
    added = tokenizer.get("added_tokens")
    _require(isinstance(added, list) and added, "tokenizer.json has no added tokens")
    ids = {str(token["content"]): int(token["id"]) for token in added}
    special = tuple(sorted(int(token["id"]) for token in added if token.get("special")))
    missing = [token for token in PROMPT_TOKENS if token not in ids]
    _require(not missing, f"tokenizer.json lacks prompt tokens {missing}")
    return {token: ids[token] for token in PROMPT_TOKENS}, special


def _eos_token_ids(model_dir: Path, raw: dict) -> tuple[int, ...]:
    generation = model_dir / "generation_config.json"
    value = None
    if generation.is_file():
        config = _read_json(generation)
        _require(not config.get("do_sample", False), "generation must be greedy by default")
        value = config.get("eos_token_id")
    if value is None:
        value = raw["text_config"].get("eos_token_id")
    values = [value] if isinstance(value, int) else value
    _require(isinstance(values, list) and values and all(isinstance(v, int) for v in values),
             "eos_token_id must be one id or a list of ids")
    return tuple(int(v) for v in values)


def _check_chat_template(model_dir: Path) -> None:
    """The native renderer reproduces this checkpoint's template; refuse a template without its markers."""
    path = model_dir / "chat_template.jinja"
    _require(path.is_file(), "checkpoint has no chat_template.jinja")
    template = path.read_text(encoding="utf-8")
    for marker in ("[gMASK]<sop>", "<|user|>", "<|assistant|>", "<|system|>",
                   "<|begin_of_image|><|image|><|end_of_image|>", "<think></think>"):
        _require(marker in template, f"chat template lacks {marker!r}")


def load_config(model_dir: Path) -> GlmOcrConfig:
    raw = _read_json(model_dir / "config.json")
    _require(raw.get("model_type") == _MODEL_TYPE, f"model_type must be {_MODEL_TYPE}")
    _require(raw.get("architectures") == [_ARCHITECTURE], f"architectures must be [{_ARCHITECTURE}]")
    vision = _vision_config(raw)
    text = _text_config(raw)
    _require(vision.out_hidden_size == text.hidden_size,
             "vision out_hidden_size must equal the decoder hidden size")
    processor = _processor_config(model_dir, vision)
    prompt_ids, special = _tokenizer_facts(model_dir)
    _check_chat_template(model_dir)
    config = GlmOcrConfig(
        text=text,
        vision=vision,
        processor=processor,
        image_token_id=int(raw.get("image_token_id", -1)),
        image_start_token_id=int(raw.get("image_start_token_id", -1)),
        image_end_token_id=int(raw.get("image_end_token_id", -1)),
        eos_token_ids=_eos_token_ids(model_dir, raw),
        prompt_token_ids=prompt_ids,
        special_token_ids=special,
    )
    _require(config.image_token_id == prompt_ids["<|image|>"], "image_token_id disagrees with the tokenizer")
    _require(config.image_start_token_id == prompt_ids["<|begin_of_image|>"],
             "image_start_token_id disagrees with the tokenizer")
    _require(config.image_end_token_id == prompt_ids["<|end_of_image|>"],
             "image_end_token_id disagrees with the tokenizer")
    _require(all(0 <= token < text.vocab_size for token in config.eos_token_ids),
             "eos ids must lie in the vocabulary")
    _require(config.max_image_patches % vision.merge_area == 0, "max image patches must merge evenly")
    return config


class Checkpoint:
    """Read-only access to the safetensors weights of one GLM-OCR checkpoint."""

    def __init__(self, model_dir: Path, config: GlmOcrConfig) -> None:
        self._readers = _open_safetensors(model_dir)
        self._owner = {}
        for reader in self._readers:
            for name in reader.keys():
                self._owner[name] = reader
        _check_inventory(self._owner, config)

    def get(self, name: str) -> np.ndarray:
        """One tensor as float32 (every checkpoint dtype is exactly representable)."""
        reader = self._owner.get(name)
        if reader is None:
            raise KeyError(f"glm_ocr checkpoint has no tensor {name}")
        return np.asarray(reader.get_tensor(name), dtype=np.float32)

    def raw(self, name: str) -> np.ndarray:
        """One tensor in its stored dtype."""
        return self._owner[name].get_tensor(name)


def _open_safetensors(model_dir: Path) -> list:
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [safe_open(str(single), framework="numpy")]
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError("glm_ocr needs model.safetensors or model.safetensors.index.json")
    shards = sorted(set(_read_json(index).get("weight_map", {}).values()))
    _require(bool(shards), "safetensors index lists no shards")
    return [safe_open(str(model_dir / shard), framework="numpy") for shard in shards]


def expected_shapes(config: GlmOcrConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the inference graphs consume, with its checkpoint shape."""
    v, t = config.vision, config.text
    shapes: dict[str, tuple[int, ...]] = {
        "model.visual.patch_embed.proj.weight": (
            v.hidden_size, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size),
        "model.visual.patch_embed.proj.bias": (v.hidden_size,),
        "model.visual.post_layernorm.weight": (v.hidden_size,),
        "model.visual.downsample.weight": (
            v.out_hidden_size, v.hidden_size, v.spatial_merge_size, v.spatial_merge_size),
        "model.visual.downsample.bias": (v.out_hidden_size,),
        "model.visual.merger.proj.weight": (v.out_hidden_size, v.out_hidden_size),
        "model.visual.merger.post_projection_norm.weight": (v.out_hidden_size,),
        "model.visual.merger.post_projection_norm.bias": (v.out_hidden_size,),
        "model.visual.merger.gate_proj.weight": (v.out_hidden_size * v.in_channels, v.out_hidden_size),
        "model.visual.merger.up_proj.weight": (v.out_hidden_size * v.in_channels, v.out_hidden_size),
        "model.visual.merger.down_proj.weight": (v.out_hidden_size, v.out_hidden_size * v.in_channels),
        "model.language_model.embed_tokens.weight": (t.vocab_size, t.hidden_size),
        "model.language_model.norm.weight": (t.hidden_size,),
        "lm_head.weight": (t.vocab_size, t.hidden_size),
    }
    for index in range(v.depth):
        block = f"model.visual.blocks.{index}"
        shapes.update({
            f"{block}.norm1.weight": (v.hidden_size,),
            f"{block}.norm2.weight": (v.hidden_size,),
            f"{block}.attn.qkv.weight": (3 * v.hidden_size, v.hidden_size),
            f"{block}.attn.qkv.bias": (3 * v.hidden_size,),
            f"{block}.attn.proj.weight": (v.hidden_size, v.hidden_size),
            f"{block}.attn.proj.bias": (v.hidden_size,),
            f"{block}.attn.q_norm.weight": (v.head_dim,),
            f"{block}.attn.k_norm.weight": (v.head_dim,),
            f"{block}.mlp.gate_proj.weight": (v.intermediate_size, v.hidden_size),
            f"{block}.mlp.gate_proj.bias": (v.intermediate_size,),
            f"{block}.mlp.up_proj.weight": (v.intermediate_size, v.hidden_size),
            f"{block}.mlp.up_proj.bias": (v.intermediate_size,),
            f"{block}.mlp.down_proj.weight": (v.hidden_size, v.intermediate_size),
            f"{block}.mlp.down_proj.bias": (v.hidden_size,),
        })
    q_width, kv_width = t.num_attention_heads * t.head_dim, t.num_key_value_heads * t.head_dim
    for index in range(t.num_hidden_layers):
        layer = f"model.language_model.layers.{index}"
        shapes.update({
            f"{layer}.input_layernorm.weight": (t.hidden_size,),
            f"{layer}.post_self_attn_layernorm.weight": (t.hidden_size,),
            f"{layer}.post_attention_layernorm.weight": (t.hidden_size,),
            f"{layer}.post_mlp_layernorm.weight": (t.hidden_size,),
            f"{layer}.self_attn.q_proj.weight": (q_width, t.hidden_size),
            f"{layer}.self_attn.k_proj.weight": (kv_width, t.hidden_size),
            f"{layer}.self_attn.v_proj.weight": (kv_width, t.hidden_size),
            f"{layer}.self_attn.o_proj.weight": (t.hidden_size, q_width),
            f"{layer}.mlp.gate_up_proj.weight": (2 * t.intermediate_size, t.hidden_size),
            f"{layer}.mlp.down_proj.weight": (t.hidden_size, t.intermediate_size),
        })
    return shapes


def _check_inventory(owner: dict, config: GlmOcrConfig) -> None:
    """Each consumed tensor exists with its shape; the only extra tensors are the MTP layers."""
    shapes = expected_shapes(config)
    for name, shape in shapes.items():
        reader = owner.get(name)
        _require(reader is not None, f"checkpoint lacks {name}")
        actual = tuple(reader.get_slice(name).get_shape())
        _require(actual == shape, f"{name} has shape {actual}, expected {shape}")
    t = config.text
    mtp = tuple(f"model.language_model.layers.{t.num_hidden_layers + index}."
                for index in range(t.num_nextn_predict_layers))
    extra = sorted(name for name in owner if name not in shapes and not name.startswith(mtp))
    _require(not extra, f"checkpoint has tensors this family does not consume: {extra[:5]}")
