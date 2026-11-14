# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the RT-DETR v2 config the builders need."""

from __future__ import annotations

import json
from pathlib import Path


def resolve(model_dir: str | Path, *, image_size: int) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing rt_detr_v2 config: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    backbone = raw.get("backbone_config") or {}

    if raw.get("decoder_method", "default") != "default":
        raise NotImplementedError(
            "rt_detr_v2 supports only decoder_method=default")

    strides = [int(v) for v in raw.get("feat_strides", [8, 16, 32])]
    shapes = [(image_size // stride, image_size // stride) for stride in strides]
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
        "num_labels": int(raw.get("num_labels", 80)),
        # The module's own constant, not a config field.
        "offset_scale": 0.5,
    }
