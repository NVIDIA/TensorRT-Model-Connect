# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read independent BERT text and CLIP vision geometry from the checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

# The pixel statistics belong to the checkpoint, not to this file; these are
# only the fallback for a checkpoint that ships no preprocessor config.
_FALLBACK_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_FALLBACK_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _require(mapping, key, where):
    if key not in mapping:
        raise KeyError(f"chinese_clip {where} is missing '{key}'")
    return mapping[key]


def _tower(config, name):
    tower = _require(config, name, "config.json")
    hidden = int(_require(tower, "hidden_size", name))
    heads = int(_require(tower, "num_attention_heads", name))
    if hidden % heads:
        raise ValueError(f"chinese_clip {name} width {hidden} is not divisible by {heads} heads")
    activation = tower.get("hidden_act", "gelu" if name == "text_config" else "quick_gelu")
    if activation != ("gelu" if name == "text_config" else "quick_gelu"):
        raise NotImplementedError(f"chinese_clip {name} uses unsupported activation: {activation}")
    return {
        "hidden_size": hidden,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "intermediate_size": int(_require(tower, "intermediate_size", name)),
        "num_layers": int(_require(tower, "num_hidden_layers", name)),
        "eps": float(tower.get("layer_norm_eps", 1e-12 if name == "text_config" else 1e-5)),
    }


def resolve(model_dir: Path) -> dict:
    """Return the geometry both builders and the runtime need."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    text = _tower(config, "text_config")
    vision = _tower(config, "vision_config")
    text_config = config["text_config"]
    vision_config = config["vision_config"]

    text["vocab_size"] = int(_require(text_config, "vocab_size", "text_config"))
    text["max_position_embeddings"] = int(
        _require(text_config, "max_position_embeddings", "text_config")
    )
    vision["image_size"] = int(_require(vision_config, "image_size", "vision_config"))
    vision["patch_size"] = int(_require(vision_config, "patch_size", "vision_config"))
    if vision["image_size"] % vision["patch_size"]:
        raise ValueError("chinese_clip image size is not a whole number of patches")
    grid = vision["image_size"] // vision["patch_size"]
    # One class token precedes the patch grid, and the position table covers both.
    vision["num_positions"] = grid * grid + 1

    preprocessor = model_dir / "preprocessor_config.json"
    image_mean, image_std = _FALLBACK_IMAGE_MEAN, _FALLBACK_IMAGE_STD
    if preprocessor.is_file():
        loaded = json.loads(preprocessor.read_text())
        if (
            loaded.get("do_center_crop", True)
            or not loaded.get("do_resize", True)
            or not loaded.get("do_normalize", True)
            or not loaded.get("do_rescale", True)
            or loaded.get("rescale_factor", 1 / 255) != 1 / 255
            or loaded.get("resample", 3) != 3
            or loaded.get("size") != {"height": vision["image_size"], "width": vision["image_size"]}
        ):
            raise NotImplementedError(
                "Chinese-CLIP requires direct bicubic resize to its vision size"
            )
        image_mean = [float(v) for v in loaded.get("image_mean", image_mean)]
        image_std = [float(v) for v in loaded.get("image_std", image_std)]
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("chinese_clip image statistics must have three channels")

    return {
        "text": text,
        "vision": vision,
        "projection_dim": int(_require(config, "projection_dim", "config.json")),
        "image_mean": image_mean,
        "image_std": image_std,
    }
