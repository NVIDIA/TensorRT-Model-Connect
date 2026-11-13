# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Grounding DINO config the builders need.

The Swin backbone emits strides 8, 16 and 32; the fourth pyramid level is a
strided convolution over the last of them. Every shape is fixed at build time
because shifted-window attention needs a static partition, so the bundle
declares one input size and the runtime resizes to it.
"""

from __future__ import annotations

import json
from pathlib import Path

_BACKBONE_STRIDES = (8, 16, 32)
# Swin-T defaults; the published backbone_config leaves them null.
_DEFAULT_PATCH_SIZE = 4
_DEFAULT_WINDOW_SIZE = 7


def feature_shapes(image_size: int, num_levels: int) -> list[tuple[int, int]]:
    """The pyramid the encoder flattens, as (height, width) per level."""
    shapes = [((image_size + stride - 1) // stride,) * 2 for stride in _BACKBONE_STRIDES]
    while len(shapes) < num_levels:
        height, width = shapes[-1]
        shapes.append(((height + 1) // 2, (width + 1) // 2))
    return shapes[:num_levels]


def _value(raw: dict, key: str, default):
    value = raw.get(key)
    return default if value is None else value


def resolve(model_dir: str | Path, *, image_size: int) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing grounding_dino config: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not raw.get("two_stage", True):
        raise NotImplementedError("grounding_dino supports only two-stage checkpoints")
    if raw.get("position_embedding_type", "sine") != "sine":
        raise NotImplementedError("grounding_dino supports only sine position embeddings")
    if not raw.get("embedding_init_target", True):
        raise NotImplementedError(
            "grounding_dino expects the learned query table, not gathered proposals")
    if str(raw.get("activation_function", "relu")) != "relu":
        raise NotImplementedError("grounding_dino supports only the relu activation")

    backbone = raw.get("backbone_config") or {}
    if str(backbone.get("model_type", "swin")) != "swin":
        raise NotImplementedError("grounding_dino supports only the Swin backbone")
    depths = [int(v) for v in _value(backbone, "depths", [2, 2, 6, 2])]
    heads = [int(v) for v in _value(backbone, "num_heads", [3, 6, 12, 24])]
    out_indices = [int(v) for v in _value(backbone, "out_indices", [2, 3, 4])]
    if len(depths) != len(heads):
        raise ValueError("grounding_dino Swin depths and num_heads disagree")

    text = raw.get("text_config") or {}
    levels = int(raw.get("num_feature_levels", 4))

    processor = Path(model_dir) / "preprocessor_config.json"
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if processor.is_file():
        payload = json.loads(processor.read_text(encoding="utf-8"))
        mean = [float(v) for v in payload.get("image_mean", mean)]
        std = [float(v) for v in payload.get("image_std", std)]
        if not payload.get("do_normalize", True):
            raise NotImplementedError("grounding_dino requires a normalising preprocessor")

    return {
        "image_size": int(image_size),
        "num_feature_levels": levels,
        "shapes": feature_shapes(int(image_size), levels),
        "swin": {
            "patch_size": int(_value(backbone, "patch_size", _DEFAULT_PATCH_SIZE)),
            "window_size": int(_value(backbone, "window_size", _DEFAULT_WINDOW_SIZE)),
            "depths": depths,
            "num_heads": heads,
            "out_indices": out_indices,
        },
        "text": {
            "num_hidden_layers": int(_value(text, "num_hidden_layers", 12)),
            "num_attention_heads": int(_value(text, "num_attention_heads", 12)),
        },
        "d_model": int(raw.get("d_model", 256)),
        "encoder_layers": int(raw.get("encoder_layers", 6)),
        "encoder_attention_heads": int(raw.get("encoder_attention_heads", 8)),
        "encoder_n_points": int(raw.get("encoder_n_points", 4)),
        "decoder_layers": int(raw.get("decoder_layers", 6)),
        "decoder_attention_heads": int(raw.get("decoder_attention_heads", 8)),
        "decoder_n_points": int(raw.get("decoder_n_points", 4)),
        # Both fusion blocks run four heads; the field is not in the config.
        "fusion_attention_heads": 4,
        "text_enhancer_heads": 4,
        "num_queries": int(raw.get("num_queries", 900)),
        "max_text_len": int(raw.get("max_text_len", 256)),
        # The box head is a fixed three-layer MLP, not a config field.
        "box_head_layers": 3,
        "image_mean": mean,
        "image_std": std,
    }
