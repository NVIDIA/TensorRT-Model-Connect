# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Depth Anything fields used by this family."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# What transformers fills in when the checkpoint stays silent. The backbone
# config of Depth-Anything-V2-Small-hf states only hidden_size, image_size,
# num_attention_heads, out_indices and patch_size, so the rest is load-bearing.
# Values taken from Dinov2Config and DepthAnythingConfig, not guessed.
_DEFAULT_DEPTH_TYPE = "relative"
_DEFAULT_MAX_DEPTH = 1.0
_DINOV2_LAYER_NORM_EPS = 1e-6
_DINOV2_NUM_HIDDEN_LAYERS = 12


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
            raise ValueError("Depth Anything config requires model_type")
        if isinstance(architectures, list) and architectures:
            architecture = str(architectures[0])
        else:
            architecture = str(raw.get("architecture", model_type))
        return ModelConfig(model_type=model_type, architecture=architecture, raw=raw)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        path = Path(model_dir) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing model config: {path}")
        return cls.from_json(path.read_text(encoding="utf-8"))


def resolve(raw: dict) -> dict:
    """The subset of the config this family builds against."""
    backbone = raw.get("backbone_config")
    if not isinstance(backbone, dict):
        raise ValueError("Depth Anything config requires backbone_config")

    hidden = int(backbone["hidden_size"])
    heads = int(backbone["num_attention_heads"])
    if hidden % heads:
        raise ValueError("Depth Anything hidden_size must divide by num_attention_heads")

    # out_indices are 1-based stage numbers; the encoder layers they name are one
    # lower. Reading them is better than assuming an even spread.
    out_indices = backbone.get("out_indices")
    if not isinstance(out_indices, list) or len(out_indices) != 4:
        raise ValueError("Depth Anything requires four backbone out_indices")

    factors = raw.get("reassemble_factors")
    if not isinstance(factors, list) or len(factors) != 4:
        raise ValueError("Depth Anything requires four reassemble_factors")

    depth_type = raw.get("depth_estimation_type") or _DEFAULT_DEPTH_TYPE
    if depth_type != "relative":
        raise NotImplementedError(
            f"depth_anything supports depth_estimation_type='relative'; got {depth_type!r}"
        )

    patch = int(raw.get("patch_size", backbone.get("patch_size", 14)))
    image = int(backbone.get("image_size", 518))
    if image % patch:
        raise ValueError("Depth Anything image_size must be a whole number of patches")

    return {
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "head_dim": hidden // heads,
        # num_hidden_layers is only a fallback: build() overrides it with the
        # layer count actually present in the checkpoint, which is authoritative.
        "num_hidden_layers": int(
            backbone.get("num_hidden_layers") or _DINOV2_NUM_HIDDEN_LAYERS),
        "layer_norm_eps": float(
            backbone.get("layer_norm_eps") or _DINOV2_LAYER_NORM_EPS),
        "patch_size": patch,
        "image_size": image,
        "patch_grid": image // patch,
        "out_layer_indices": [int(index) - 1 for index in out_indices],
        "reassemble_factors": [float(value) for value in factors],
        "neck_hidden_sizes": [int(value) for value in raw["neck_hidden_sizes"]],
        "fusion_hidden_size": int(raw["fusion_hidden_size"]),
        "head_hidden_size": int(raw["head_hidden_size"]),
        "head_in_index": int(raw.get("head_in_index", -1)),
        "reassemble_hidden_size": int(raw.get("reassemble_hidden_size", hidden)),
        "max_depth": float(raw.get("max_depth") or _DEFAULT_MAX_DEPTH),
    }
