# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the YOLOS fields used by this family."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_type: str
    architecture: str
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_json(text: str) -> "ModelConfig":
        raw = json.loads(text)
        model_type = raw.get("model_type")
        architectures = raw.get("architectures")
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("YOLOS config requires model_type")
        if isinstance(architectures, list) and architectures:
            architecture = str(architectures[0])
        else:
            architecture = str(raw.get("architecture", model_type))
        return ModelConfig(model_type=model_type, architecture=architecture, raw=raw)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing model config: {config_path}")
        return cls.from_json(config_path.read_text(encoding="utf-8"))


def _image_size(raw: dict) -> tuple[int, int]:
    """YOLOS states a non-square native size as [height, width]."""
    value = raw.get("image_size")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return int(value), int(value)
    raise ValueError("YOLOS config requires image_size")


def resolve(raw: dict) -> dict:
    """The subset of the config this family builds against."""
    height, width = _image_size(raw)
    patch = int(raw.get("patch_size", 16))
    if height % patch or width % patch:
        raise ValueError("YOLOS image_size must be a whole number of patches")
    hidden = int(raw["hidden_size"])
    heads = int(raw["num_attention_heads"])
    if hidden % heads:
        raise ValueError("YOLOS hidden_size must divide by num_attention_heads")
    return {
        "image_height": height,
        "image_width": width,
        "patch_size": patch,
        "hidden_size": hidden,
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "num_attention_heads": heads,
        "head_dim": hidden // heads,
        "intermediate_size": int(raw["intermediate_size"]),
        "layer_norm_eps": float(raw.get("layer_norm_eps", 1e-12)),
        "num_detection_tokens": int(raw.get("num_detection_tokens", 100)),
        "use_mid_position_embeddings": bool(raw.get("use_mid_position_embeddings", True)),
        "num_patches": (height // patch) * (width // patch),
    }
