# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict MiniCPM-V 4.6 checkpoint configuration (``config.json``).

Only the geometry this family implements is accepted; any other value is an
error instead of a silently different graph.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

_INT32_MAX = 2**31 - 1
DEFAULT_CAPACITY = 4096


def load_json(path: Path) -> object:
    """Parse a checkpoint JSON file; a repeated key or a NaN / Infinity literal is an error."""
    def unique(pairs: list[tuple[str, object]]) -> dict:
        # One pass: tokenizer.json holds objects with hundreds of thousands of keys.
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"MiniCPM-V 4.6 {path.name} repeats the key {key!r}")
            result[key] = value
        return result

    def constant(name: str) -> None:
        raise ValueError(f"MiniCPM-V 4.6 {path.name} contains {name}, which is not a JSON number")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=constant)


def stored_value(value: float, fmt: str) -> float:
    """``value`` as a ``struct`` float format stores it ('f' FP32, 'e' FP16); infinity when it overflows."""
    try:
        return struct.unpack(f"<{fmt}", struct.pack(f"<{fmt}", value))[0]
    except OverflowError:
        return math.inf


def positive_when_stored(value: float, fmt: str) -> bool:
    stored = stored_value(value, fmt)
    return math.isfinite(stored) and stored > 0


def same_value(value: object, expected: object) -> bool:
    """Equality that also requires the JSON type at every level: True is not 1, and 3.0 is not 3."""
    if isinstance(expected, dict):
        return (isinstance(value, dict) and value.keys() == expected.keys()
                and all(same_value(value[key], want) for key, want in expected.items()))
    if isinstance(expected, list):
        return (isinstance(value, list) and len(value) == len(expected)
                and all(same_value(item, want) for item, want in zip(value, expected)))
    return type(value) is type(expected) and value == expected


def _integer(raw: dict, key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or not 0 < value <= _INT32_MAX:
        raise ValueError(f"MiniCPM-V 4.6 config field {key!r} must be a positive 32-bit integer")
    return value


def _token_id(raw: dict, key: str, vocab_size: int) -> int:
    value = raw.get(key)
    if type(value) is not int or not 0 <= value < vocab_size:
        raise ValueError(f"MiniCPM-V 4.6 config field {key!r} must be a token id below {vocab_size}")
    return value


def _number(raw: dict, key: str) -> float:
    value = raw.get(key)
    if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            or not positive_when_stored(value, "f")):
        raise ValueError(f"MiniCPM-V 4.6 config field {key!r} must be a positive number that FP32 represents")
    return float(value)


def _require(raw: dict, key: str, expected, default=None) -> None:
    if not same_value(raw.get(key, default), expected):
        raise ValueError(f"MiniCPM-V 4.6 requires {key}={expected!r}, got {raw.get(key, default)!r}")


def parse_layer_types(raw_types: list[str]) -> list[str]:
    """Normalize decoder layer types to 'deltanet' (linear attention) or 'attention'."""
    mapping = {
        "linear_attention": "deltanet",
        "full_attention": "attention",
    }
    result = []
    for value in raw_types:
        if value not in mapping:
            raise ValueError(f"MiniCPM-V 4.6 does not support decoder layer type {value!r}")
        result.append(mapping[value])
    return result


@dataclass(frozen=True)
class TextConfig:
    """The ``qwen3_5_text`` decoder of the checkpoint."""

    hidden_size: int
    intermediate_size: int
    vocab_size: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    max_position_embeddings: int

    @classmethod
    def from_dict(cls, raw: dict) -> TextConfig:
        if not isinstance(raw, dict):
            raise ValueError("MiniCPM-V 4.6 config.json requires a text_config object")
        _require(raw, "model_type", "qwen3_5_text")
        _require(raw, "hidden_act", "silu")
        _require(raw, "attention_bias", False)
        _require(raw, "attn_output_gate", True)
        _require(raw, "mamba_ssm_dtype", "float32")
        _require(raw, "tie_word_embeddings", True)
        _require(raw, "mlp_only_layers", [])
        rope = raw.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_type") != "default" or "mrope_section" in rope:
            raise ValueError("MiniCPM-V 4.6 requires default one-dimensional rotary parameters")
        raw_types = raw.get("layer_types")
        if not isinstance(raw_types, list) or not all(isinstance(kind, str) for kind in raw_types):
            raise ValueError("MiniCPM-V 4.6 layer_types must be a JSON array of layer type names")
        layer_types = tuple(parse_layer_types(raw_types))
        num_layers = _integer(raw, "num_hidden_layers")
        if len(layer_types) != num_layers or "deltanet" not in layer_types or "attention" not in layer_types:
            raise ValueError("MiniCPM-V 4.6 layer_types must list every decoder layer of both kinds")
        head_dim = _integer(raw, "head_dim")
        partial = rope.get("partial_rotary_factor", raw.get("partial_rotary_factor"))
        if type(partial) not in (int, float) or not 0 < partial <= 1:
            raise ValueError("MiniCPM-V 4.6 partial_rotary_factor must be in (0, 1]")
        rotary_dim = int(head_dim * partial)
        if rotary_dim < 2 or rotary_dim % 2 or rotary_dim != head_dim * partial:
            raise ValueError("MiniCPM-V 4.6 rotary dimensions must be an even number of head dimensions")
        config = cls(
            hidden_size=_integer(raw, "hidden_size"),
            intermediate_size=_integer(raw, "intermediate_size"),
            vocab_size=_integer(raw, "vocab_size"),
            layer_types=layer_types,
            num_attention_heads=_integer(raw, "num_attention_heads"),
            num_key_value_heads=_integer(raw, "num_key_value_heads"),
            head_dim=head_dim,
            linear_num_key_heads=_integer(raw, "linear_num_key_heads"),
            linear_num_value_heads=_integer(raw, "linear_num_value_heads"),
            linear_key_head_dim=_integer(raw, "linear_key_head_dim"),
            linear_value_head_dim=_integer(raw, "linear_value_head_dim"),
            linear_conv_kernel_dim=_integer(raw, "linear_conv_kernel_dim"),
            rms_norm_eps=_number(raw, "rms_norm_eps"),
            rope_theta=_number(rope, "rope_theta"),
            rotary_dim=rotary_dim,
            max_position_embeddings=_integer(raw, "max_position_embeddings"),
        )
        if (config.num_attention_heads % config.num_key_value_heads
                or config.linear_num_value_heads % config.linear_num_key_heads
                or config.linear_key_head_dim != config.linear_value_head_dim):
            raise ValueError("MiniCPM-V 4.6 decoder head geometry is not supported")
        return config


@dataclass(frozen=True)
class VisionConfig:
    """The SigLIP-style vision tower with its windowed merger after ``insert_layer_id``."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    positions_per_side: int
    layer_norm_eps: float
    insert_layer_id: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_width(self) -> int:
        return 3 * self.patch_size * self.patch_size

    @classmethod
    def from_dict(cls, raw: dict, insert_layer_id: object) -> VisionConfig:
        if not isinstance(raw, dict):
            raise ValueError("MiniCPM-V 4.6 config.json requires a vision_config object")
        _require(raw, "model_type", "minicpmv4_6_vision")
        _require(raw, "hidden_act", "gelu_pytorch_tanh")
        _require(raw, "num_channels", 3)
        _require(raw, "window_kernel_size", [2, 2], default=[2, 2])
        image_size = _integer(raw, "image_size")
        patch_size = _integer(raw, "patch_size")
        if image_size % patch_size:
            raise ValueError("MiniCPM-V 4.6 vision image_size must be a multiple of patch_size")
        # The position embedding grid must have at least two rows (the runtime's bucketing needs them).
        if image_size // patch_size < 2:
            raise ValueError("MiniCPM-V 4.6 vision image_size must span at least two patches")
        config = cls(
            hidden_size=_integer(raw, "hidden_size"),
            intermediate_size=_integer(raw, "intermediate_size"),
            num_hidden_layers=_integer(raw, "num_hidden_layers"),
            num_attention_heads=_integer(raw, "num_attention_heads"),
            patch_size=patch_size,
            positions_per_side=image_size // patch_size,
            layer_norm_eps=_number(raw, "layer_norm_eps"),
            insert_layer_id=insert_layer_id if type(insert_layer_id) is int else -1,
        )
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("MiniCPM-V 4.6 vision hidden size must split evenly over heads")
        if not 0 <= config.insert_layer_id < config.num_hidden_layers - 1:
            raise ValueError("MiniCPM-V 4.6 requires the windowed merger inside the vision encoder")
        return config


@dataclass(frozen=True)
class CheckpointConfig:
    """Top-level ``MiniCPMV4_6ForConditionalGeneration`` configuration."""

    text: TextConfig
    vision: VisionConfig
    image_token_id: int
    video_token_id: int
    eos_token_id: int

    @classmethod
    def from_dir(cls, model_dir: Path) -> CheckpointConfig:
        raw = load_json(Path(model_dir) / "config.json")
        if not isinstance(raw, dict):
            raise ValueError("MiniCPM-V 4.6 config.json must contain one JSON object")
        _require(raw, "model_type", "minicpmv4_6")
        _require(raw, "architectures", ["MiniCPMV4_6ForConditionalGeneration"])
        _require(raw, "tie_word_embeddings", True)
        _require(raw, "drop_vision_last_layer", False, default=False)
        _require(raw, "merge_kernel_size", [2, 2], default=[2, 2])
        _require(raw, "merger_times", 1, default=1)
        _require(raw, "downsample_mode", "16x", default="16x")
        text = TextConfig.from_dict(raw.get("text_config"))
        vision = VisionConfig.from_dict(raw.get("vision_config"), raw.get("insert_layer_id"))
        return cls(
            text=text,
            vision=vision,
            image_token_id=_token_id(raw, "image_token_id", text.vocab_size),
            video_token_id=_token_id(raw, "video_token_id", text.vocab_size),
            eos_token_id=_token_id(raw, "eos_token_id", text.vocab_size),
        )


def generation_ids(model_dir: Path, config: CheckpointConfig) -> list[int]:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return [config.eos_token_id]
    generation = load_json(path)
    value = generation.get("eos_token_id", config.eos_token_id) if isinstance(generation, dict) else None
    ids = [value] if type(value) is int else value
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < config.text.vocab_size for i in ids)):
        raise ValueError("MiniCPM-V 4.6 generation_config.json eos_token_id must be distinct token ids")
    return ids


def capacity_rows(requested, max_positions: int) -> int:
    """KV cache rows: the request's max_sequence_length, or the default when it is None."""
    if requested is None:
        requested = min(DEFAULT_CAPACITY, max_positions)
    elif type(requested) is not int:
        raise ValueError("minicpmv4_6 max_sequence_length must be an integer")
    elif requested > max_positions:
        raise ValueError("minicpmv4_6 max_sequence_length exceeds the checkpoint's context capacity")
    # The default as well: the runtime needs a prompt row and a generated row.
    if requested < 2:
        raise ValueError("minicpmv4_6 max_sequence_length must hold a prompt and one generated token")
    return requested
