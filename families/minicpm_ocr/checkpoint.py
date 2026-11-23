# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-OCR checkpoint reading: config validation and safetensors tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Registers the NumPy bfloat16 dtype that safetensors returns for BF16 tensors.
import ml_dtypes
import numpy as np
from safetensors import safe_open

from .checkpoint_contract import require_unquantized

_BFLOAT16 = np.dtype(ml_dtypes.bfloat16)


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float
    linear_conv_kernel: int
    linear_key_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    vocab_size: int
    max_position_embeddings: int


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    patch_size: int
    image_size: int
    layer_norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def position_grid(self) -> int:
        return self.image_size // self.patch_size


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    vision: VisionConfig
    image_token_id: int
    merge_kernel: tuple[int, int]
    eos_token_ids: tuple[int, ...]
    processor: dict


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"minicpm_ocr: {message}")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain one JSON object")
    return value


def load_config(model_dir: Path) -> ModelConfig:
    """Validate the exact checkpoint contract this family implements."""
    raw = _read_json(model_dir / "config.json")
    _require(raw.get("model_type") == "minicpm_ocr", "config.json model_type must be minicpm_ocr")
    _require(raw.get("architectures") == ["MiniCPMOcrForConditionalGeneration"],
             "only MiniCPMOcrForConditionalGeneration is supported")
    require_unquantized(raw)
    # The "4x" mode skips the in-tower window merger; "16x" is a different graph.
    _require(raw.get("downsample_mode", "4x") == "4x", "only downsample_mode '4x' is supported")
    _require(int(raw.get("merger_times", 1)) == 1, "only one merger round is supported")
    merge = tuple(raw.get("merge_kernel_size", (2, 2)))
    _require(merge == (2, 2), "only a 2x2 merge kernel is supported")
    _require(raw.get("tie_word_embeddings", True) is True, "the LM head must be tied to the embedding")
    text = raw["text_config"]
    vision = raw["vision_config"]
    _require(text.get("model_type") == "qwen3_5_text", "text_config must be qwen3_5_text")
    _require(vision.get("model_type") == "minicpm_ocr_vision", "vision_config must be minicpm_ocr_vision")
    _require(int(text.get("mtp_num_hidden_layers", 0)) == 0, "MTP layers are not supported")
    _require(not text.get("mlp_only_layers"), "mlp_only_layers is not supported")
    _require(text.get("hidden_act") == "silu" and text.get("attn_output_gate") is True,
             "the text decoder must use SiLU and an attention output gate")
    _require(text.get("attention_bias") is False, "attention biases are not supported")
    rope = text.get("rope_parameters") or {}
    _require(rope.get("rope_type", "default") == "default" and "mrope_section" not in rope,
             "only default (1-D) RoPE is supported")
    layer_types = tuple(text["layer_types"])
    _require(set(layer_types) <= {"linear_attention", "full_attention"}
             and len(layer_types) == int(text["num_hidden_layers"]), "invalid layer_types")
    head_dim = int(text["head_dim"])
    rotary_dim = int(head_dim * float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))))
    _require(rotary_dim % 2 == 0 and 0 < rotary_dim <= head_dim, "invalid partial rotary dimension")
    _require(int(text["linear_num_value_heads"]) == int(text["linear_num_key_heads"]),
             "grouped Gated DeltaNet heads are not supported")
    _require(vision.get("hidden_act") == "gelu_pytorch_tanh", "the vision MLP must use gelu_pytorch_tanh")
    text_config = TextConfig(
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        num_layers=int(text["num_hidden_layers"]),
        layer_types=layer_types,
        num_heads=int(text["num_attention_heads"]),
        num_kv_heads=int(text["num_key_value_heads"]),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope_theta=float(rope["rope_theta"]),
        rms_norm_eps=float(text["rms_norm_eps"]),
        linear_conv_kernel=int(text["linear_conv_kernel_dim"]),
        linear_key_heads=int(text["linear_num_key_heads"]),
        linear_value_heads=int(text["linear_num_value_heads"]),
        linear_key_dim=int(text["linear_key_head_dim"]),
        linear_value_dim=int(text["linear_value_head_dim"]),
        vocab_size=int(text["vocab_size"]),
        max_position_embeddings=int(text["max_position_embeddings"]),
    )
    vision_config = VisionConfig(
        hidden_size=int(vision["hidden_size"]),
        intermediate_size=int(vision["intermediate_size"]),
        num_layers=int(vision["num_hidden_layers"]),
        num_heads=int(vision["num_attention_heads"]),
        patch_size=int(vision["patch_size"]),
        image_size=int(vision["image_size"]),
        layer_norm_eps=float(vision["layer_norm_eps"]),
    )
    _require(vision_config.hidden_size % vision_config.num_heads == 0, "invalid vision heads")
    generation = _read_json(model_dir / "generation_config.json")
    eos = generation.get("eos_token_id")
    eos_ids = tuple(eos) if isinstance(eos, list) else (eos,)
    _require(all(isinstance(i, int) and i >= 0 for i in eos_ids) and eos_ids, "invalid eos_token_id")
    processor = _read_json(model_dir / "preprocessor_config.json")
    return ModelConfig(text_config, vision_config, int(raw["image_token_id"]), merge, eos_ids, processor)


def _open_safetensors(model_dir: Path):
    """Open every safetensors shard of the checkpoint with NumPy (BF16 via ml_dtypes)."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        files = sorted(set(_read_json(index_path)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    readers = {}
    for name in files:
        handle = safe_open(str(model_dir / name), framework="numpy")
        for key in handle.keys():
            readers[key] = handle
    _require(bool(readers), "the checkpoint has no safetensors tensors")
    return readers


class Checkpoint:
    """Tensor access by name, returned as float32 copies."""

    def __init__(self, model_dir: Path) -> None:
        self._readers = _open_safetensors(model_dir)

    def has(self, name: str) -> bool:
        return name in self._readers

    def tensor(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        _require(name in self._readers, f"missing checkpoint tensor {name}")
        value = self._readers[name].get_tensor(name)
        _require(value.dtype in (np.dtype(np.float32), np.dtype(np.float16), _BFLOAT16),
                 f"{name} has unsupported dtype {value.dtype}")
        value = np.asarray(value, dtype=np.float32)
        if shape is not None:
            _require(tuple(value.shape) == tuple(shape), f"{name} has shape {value.shape}, expected {shape}")
        return value
