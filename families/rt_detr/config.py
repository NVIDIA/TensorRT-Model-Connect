# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the RT-DETR config the builders need."""

from __future__ import annotations

import json
from pathlib import Path


def resolve(model_dir: str | Path, *, image_size: int) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing rt_detr config: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    backbone = raw.get("backbone_config") or {}
    if raw.get("model_type") != "rt_detr":
        raise ValueError("rt_detr requires the original RT-DETR architecture")
    if backbone.get("layer_type", "bottleneck") != "bottleneck" or backbone.get("downsample_in_bottleneck", False):
        raise NotImplementedError("rt_detr requires the ResNet-D bottleneck with 3x3 downsampling")

    if raw.get("decoder_method", "default") != "default":
        raise NotImplementedError(
            "rt_detr supports only decoder_method=default")
    if raw.get("learn_initial_query", False):
        raise NotImplementedError("rt_detr does not support learned initial queries")
    if raw.get("encoder_layers", 1) != 1:
        raise NotImplementedError("rt_detr supports exactly one encoder layer")

    strides = [int(v) for v in raw.get("feat_strides", [8, 16, 32])]
    if strides != [8, 16, 32]:
        raise NotImplementedError("rt_detr requires feature strides [8, 16, 32] in backbone order")
    shapes = [(image_size // stride, image_size // stride) for stride in strides]
    num_labels = raw.get("num_labels", len(raw.get("id2label", {})) or 80)
    if type(num_labels) is not int or num_labels <= 0:
        raise ValueError("rt_detr num_labels must be a positive integer")
    return {
        "image_size": int(image_size),
        "shapes": shapes,
        "depths": [int(v) for v in backbone.get("depths", [2, 2, 2, 2])],
        "encoder_hidden_dim": int(raw.get("encoder_hidden_dim", 256)),
        "encoder_attention_heads": int(raw.get("encoder_attention_heads", 8)),
        "encode_proj_layers": [int(v) for v in raw.get("encode_proj_layers", [2])],
        "positional_encoding_temperature": float(
            raw.get("positional_encoding_temperature", 10000)),
        # CSPRepLayer carries three RepVGG bottlenecks at this hidden expansion.
        "bottlenecks": 3,
        "d_model": int(raw.get("d_model", 256)),
        "decoder_layers": int(raw.get("decoder_layers", 6)),
        "decoder_attention_heads": int(raw.get("decoder_attention_heads", 8)),
        "decoder_n_points": int(raw.get("decoder_n_points", 4)),
        "num_queries": int(raw.get("num_queries", 300)),
        "num_labels": num_labels,
        # The module's own constant, not a config field.
        "offset_scale": 0.5,
    }
