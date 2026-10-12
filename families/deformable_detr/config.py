# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Deformable DETR config the builders need."""

from __future__ import annotations

import json
from pathlib import Path

# Strides 8, 16 and 32 leave the backbone; the fourth level is a strided
# convolution on the last of them, so it halves with rounding up.
_BACKBONE_STRIDES = (8, 16, 32)


def feature_shapes(image_size: int, num_levels: int) -> list[tuple[int, int]]:
    """The measured pyramid, as (height, width) per level.

    Starting from the stem instead of stride 8 is the error that turned a
    100x100 leading map into a 200x200 one, so the strides are named here
    rather than derived from a loop over the backbone's stages.
    """
    shapes = [(image_size // stride, image_size // stride) for stride in _BACKBONE_STRIDES]
    while len(shapes) < num_levels:
        height, width = shapes[-1]
        shapes.append(((height + 1) // 2, (width + 1) // 2))
    return shapes[:num_levels]


def resolve(model_dir: str | Path, *, image_size: int) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing deformable_detr config: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    if raw.get("two_stage", False):
        raise NotImplementedError("deformable_detr does not support two_stage checkpoints")
    if raw.get("with_box_refine", False):
        raise NotImplementedError(
            "deformable_detr does not support iterative box refinement")
    if raw.get("position_embedding_type", "sine") != "sine":
        raise NotImplementedError("deformable_detr supports only sine position embeddings")
    if raw.get("dilation", False):
        raise NotImplementedError("deformable_detr does not support a dilated backbone")
    if str(raw.get("activation_function", "relu")) != "relu":
        raise NotImplementedError("deformable_detr supports only the relu activation")

    levels = int(raw.get("num_feature_levels", 4))
    labels = raw.get("num_labels")
    if labels is None:
        labels = len(raw.get("id2label", {})) or 91

    processor = Path(model_dir) / "preprocessor_config.json"
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = True
    if processor.is_file():
        payload = json.loads(processor.read_text(encoding="utf-8"))
        mean = [float(v) for v in payload.get("image_mean", mean)]
        std = [float(v) for v in payload.get("image_std", std)]
        normalize = bool(payload.get("do_normalize", True))

    return {
        "image_size": int(image_size),
        "num_feature_levels": levels,
        "shapes": feature_shapes(int(image_size), levels),
        "d_model": int(raw.get("d_model", 256)),
        "encoder_layers": int(raw.get("encoder_layers", 6)),
        "encoder_attention_heads": int(raw.get("encoder_attention_heads", 8)),
        "encoder_n_points": int(raw.get("encoder_n_points", 4)),
        "decoder_layers": int(raw.get("decoder_layers", 6)),
        "decoder_attention_heads": int(raw.get("decoder_attention_heads", 8)),
        "decoder_n_points": int(raw.get("decoder_n_points", 4)),
        "num_queries": int(raw.get("num_queries", 300)),
        "num_labels": int(labels),
        # The box head is a fixed three-layer MLP, not a config field.
        "box_head_layers": 3,
        "image_mean": mean,
        "image_std": std,
        "do_normalize": normalize,
    }
