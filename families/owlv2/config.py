# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the OWLv2 geometry, mostly out of the checkpoint's own tensor shapes.

This checkpoint's ``config.json`` is nearly empty. Its ``vision_config`` states
only ``image_size`` and ``patch_size``; its ``text_config`` states nothing at
all. Every width, depth and vocabulary size below is therefore derived from the
shapes of the tensors that were actually shipped, which cannot disagree with
the weights the way a stale config field can.

Two numbers do not appear in any tensor shape - the head count and the layer
norm epsilon - so those fall back to the architecture's published defaults, and
the fallback is used only when ``config.json`` is silent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Defaults of Owlv2VisionConfig / Owlv2TextConfig, used only where config.json
# is silent. A head count cannot be read off a tensor shape.
_DEFAULT_HEADS = {"vision": 12, "text": 8}
_DEFAULT_EPS = 1e-5

_FALLBACK_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_FALLBACK_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _require(shapes, key):
    if key not in shapes:
        raise KeyError(f"owlv2 checkpoint is missing '{key}'")
    return shapes[key]


def _count_layers(shapes, prefix):
    pattern = re.compile(re.escape(prefix) + r"\.encoder\.layers\.(\d+)\.")
    indices = {int(match.group(1)) for key in shapes if (match := pattern.match(key))}
    if not indices or sorted(indices) != list(range(len(indices))):
        raise ValueError(f"owlv2 {prefix} encoder layers are not a contiguous range")
    return len(indices)


def _tower(shapes, stated, prefix, which):
    hidden = _require(shapes, f"{prefix}.encoder.layers.0.layer_norm1.weight")[0]
    heads = int(stated.get("num_attention_heads", _DEFAULT_HEADS[which]))
    if hidden % heads:
        raise ValueError(f"owlv2 {which} width {hidden} is not divisible by {heads} heads")
    activation = stated.get("hidden_act", "quick_gelu")
    if activation != "quick_gelu":
        raise NotImplementedError(f"owlv2 {which} uses unsupported activation: {activation}")
    return {
        "hidden_size": int(hidden),
        "num_heads": heads,
        "head_dim": int(hidden) // heads,
        "intermediate_size": int(_require(shapes, f"{prefix}.encoder.layers.0.mlp.fc1.weight")[0]),
        "num_layers": _count_layers(shapes, prefix),
        "eps": float(stated.get("layer_norm_eps", _DEFAULT_EPS)),
    }


def resolve(model_dir: Path, shapes: dict) -> dict:
    """Return the geometry the builders and the runtime need.

    ``shapes`` maps every checkpoint tensor name to its shape.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    text = _tower(shapes, config.get("text_config", {}), "owlv2.text_model", "text")
    vision = _tower(shapes, config.get("vision_config", {}), "owlv2.vision_model", "vision")

    text["vocab_size"] = int(
        _require(shapes, "owlv2.text_model.embeddings.token_embedding.weight")[0])
    # The text position table is the prompt length: this checkpoint trains on 16
    # tokens per query, not CLIP's 77.
    text["max_position_embeddings"] = int(
        _require(shapes, "owlv2.text_model.embeddings.position_embedding.weight")[0])

    patch = _require(shapes, "owlv2.vision_model.embeddings.patch_embedding.weight")
    vision["patch_size"] = int(patch[-1])
    positions = int(_require(shapes, "owlv2.vision_model.embeddings.position_embedding.weight")[0])
    # One class token precedes a square patch grid.
    grid = round((positions - 1) ** 0.5)
    if grid * grid + 1 != positions:
        raise ValueError(f"owlv2 position table of {positions} rows is not a square grid plus one")
    vision["num_patches"] = grid * grid
    vision["grid"] = grid
    vision["num_positions"] = positions
    vision["image_size"] = grid * vision["patch_size"]
    stated_size = config.get("vision_config", {}).get("image_size")
    if stated_size is not None and int(stated_size) != vision["image_size"]:
        raise ValueError(
            f"owlv2 config.json states image_size {stated_size} but the position table "
            f"describes {vision['image_size']}")

    preprocessor = model_dir / "preprocessor_config.json"
    image_mean, image_std = _FALLBACK_IMAGE_MEAN, _FALLBACK_IMAGE_STD
    if preprocessor.is_file():
        loaded = json.loads(preprocessor.read_text())
        image_mean = [float(v) for v in loaded.get("image_mean", image_mean)]
        image_std = [float(v) for v in loaded.get("image_std", image_std)]
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("owlv2 image statistics must have three channels")

    return {
        "text": text,
        "vision": vision,
        "query_dim": int(_require(shapes, "class_head.dense0.weight")[0]),
        "image_mean": image_mean,
        "image_std": image_std,
    }
