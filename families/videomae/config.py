# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model and processor configuration of a VideoMAE pre-training checkpoint.

Dimensions come from ``config.json``, with the defaults of transformers'
``VideoMAEConfig`` for keys a file omits. Flags that change what the network
computes are accepted only at the values this family reproduces.

The runtime takes clips that already have the model's frame count and frame
size, so the processor must leave such a clip unchanged apart from rescaling
and normalization: its resize and center crop have to be the identity at that
size. ``preprocessor_config.json`` is read with the defaults of
``VideoMAEImageProcessor``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ARCHITECTURE = "VideoMAEForPreTraining"

# VideoMAEConfig defaults (transformers 5.2.0) for keys a config.json omits.
_MODEL_DEFAULTS = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "num_frames": 16,
    "tubelet_size": 2,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
    "use_mean_pooling": True,
    "decoder_num_attention_heads": 6,
    "decoder_hidden_size": 384,
    "decoder_num_hidden_layers": 4,
    "decoder_intermediate_size": 1536,
    "norm_pix_loss": True,
}

# VideoMAEImageProcessor defaults (transformers 5.2.0) for keys a
# preprocessor_config.json omits.
_PROCESSOR_DEFAULTS = {
    "do_resize": True,
    "size": {"shortest_edge": 224},
    "do_center_crop": True,
    "crop_size": {"height": 224, "width": 224},
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def _object_without_repeated_keys(pairs: list) -> dict:
    keys = [key for key, _ in pairs]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"videomae refuses repeated JSON keys: {', '.join(repeated)}")
    return dict(pairs)


def _read_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"videomae checkpoint file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=_object_without_repeated_keys)
    if not isinstance(value, dict):
        raise ValueError(f"videomae expects a JSON object in {path}")
    return value


def _number(value) -> bool:
    """A finite JSON number; booleans and null are not numbers."""
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"videomae config.json needs a positive integer {name}; got {value!r}")
    return value


def _square(value, name: str) -> int:
    """``image_size`` and ``patch_size`` may be an int or a pair; the family needs a square."""
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"videomae config.json needs one or two values for {name}; got {value!r}")
        height, width = (_positive_int(item, name) for item in value)
        if height != width:
            raise NotImplementedError(f"videomae needs a square {name}; got {value!r}")
        return height
    return _positive_int(value, name)


def _expect(config: dict, key: str, expected) -> None:
    value = config[key]
    if value != expected or type(value) is not type(expected):
        raise NotImplementedError(
            f"videomae reproduces {key}={expected!r}; this checkpoint has {value!r}")


def _model(path: Path) -> dict:
    raw = _read_object(path)
    architectures = raw.get("architectures")
    if (raw.get("model_type") != "videomae" or not isinstance(architectures, list)
            or ARCHITECTURE not in architectures):
        raise ValueError(f"videomae builds {ARCHITECTURE} checkpoints only: {path}")
    config = {**_MODEL_DEFAULTS, **raw}
    # Pixel targets normalized per tube (the reconstruction the runtime inverts),
    # RGB frames, the exact error-function GELU, q and v biases (k has none).
    _expect(config, "norm_pix_loss", True)
    _expect(config, "num_channels", 3)
    _expect(config, "hidden_act", "gelu")
    _expect(config, "qkv_bias", True)
    if not isinstance(config["use_mean_pooling"], bool):
        raise ValueError("videomae config.json needs a boolean use_mean_pooling")

    image_size = _square(config["image_size"], "image_size")
    patch_size = _square(config["patch_size"], "patch_size")
    num_frames = _positive_int(config["num_frames"], "num_frames")
    tubelet = _positive_int(config["tubelet_size"], "tubelet_size")
    if image_size % patch_size or num_frames % tubelet:
        raise ValueError("videomae needs frames and frame size divisible into whole tubes")
    grid = image_size // patch_size
    tubes = num_frames // tubelet * grid * grid
    if tubes < 2 or tubelet * patch_size * patch_size < 2:
        raise ValueError("videomae needs at least two tubes of at least two pixels")

    widths = {}
    for prefix in ("", "decoder_"):
        width = _positive_int(config[f"{prefix}hidden_size"], f"{prefix}hidden_size")
        heads = _positive_int(config[f"{prefix}num_attention_heads"],
                              f"{prefix}num_attention_heads")
        if width % heads:
            raise ValueError(f"videomae {prefix}hidden_size is not a multiple of its head count")
        widths[prefix] = {
            "width": width,
            "heads": heads,
            "layers": _positive_int(config[f"{prefix}num_hidden_layers"],
                                    f"{prefix}num_hidden_layers"),
            "intermediate": _positive_int(config[f"{prefix}intermediate_size"],
                                          f"{prefix}intermediate_size"),
        }
    eps = config["layer_norm_eps"]
    # The graph adds it in float32: it must stay positive and finite there.
    with np.errstate(over="ignore"):
        if not _number(eps) or not 0.0 < np.float32(eps) < np.inf:
            raise ValueError(f"videomae config.json needs a positive float32 layer_norm_eps; got {eps!r}")

    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "num_frames": num_frames,
        "tubelet_size": tubelet,
        "grid": grid,
        "tubes": tubes,
        "tube_values": tubelet * patch_size * patch_size * 3,
        "encoder": widths[""],
        "decoder": widths["decoder_"],
        "layer_norm_eps": float(eps),
        "final_layernorm": not config["use_mean_pooling"],
    }


def _size(value, *, default_to_square: bool) -> dict:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"height": value, "width": value} if default_to_square else {"shortest_edge": value}
    if isinstance(value, dict) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value.values()):
        return dict(value)
    raise ValueError(f"videomae cannot read the processor size {value!r}")


def _flag(processor: dict, key: str) -> bool:
    value = processor[key]
    if not isinstance(value, bool):
        raise ValueError(f"videomae needs a boolean {key} in preprocessor_config.json; got {value!r}")
    return value


def _processor(path: Path, image_size: int) -> dict:
    processor = {**_PROCESSOR_DEFAULTS, **_read_object(path)}
    square = {"height": image_size, "width": image_size}
    if _flag(processor, "do_resize"):
        size = _size(processor["size"], default_to_square=False)
        if size not in ({"shortest_edge": image_size}, square):
            raise NotImplementedError(
                f"videomae needs a processor that keeps {image_size}x{image_size} frames; "
                f"its resize is {size!r}")
    if _flag(processor, "do_center_crop"):
        crop = _size(processor["crop_size"], default_to_square=True)
        if crop != square:
            raise NotImplementedError(
                f"videomae needs a processor that keeps {image_size}x{image_size} frames; "
                f"its center crop is {crop!r}")
    factor = processor["rescale_factor"]
    if not _number(factor):
        raise ValueError(f"videomae needs a numeric rescale_factor; got {factor!r}")
    if not _flag(processor, "do_rescale") or not math.isclose(factor, 1 / 255, rel_tol=0.0,
                                                               abs_tol=1e-12):
        raise NotImplementedError("videomae reproduces the 1/255 rescale of 8-bit frames only")
    if not _flag(processor, "do_normalize"):
        raise NotImplementedError("videomae reproduces a normalizing processor only")
    mean, std = processor["image_mean"], processor["image_std"]
    if (not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3
            or not all(_number(value) for value in mean + std) or min(std) <= 0):
        raise ValueError("videomae needs three finite image_mean values and three positive image_std values")
    # The runtime normalizes in float32, (pixel - mean) / std, for pixels in [0, 1].
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean32, std32 = np.float32(mean), np.float32(std)
        normalized = (np.float32([[0.0], [1.0]]) - mean32) / std32
    if not (np.isfinite(mean32).all() and np.isfinite(std32).all() and (std32 > 0).all()
            and np.isfinite(normalized).all()):
        raise ValueError("videomae needs image statistics that keep normalized pixels finite in float32")
    return {"image_mean": [float(value) for value in mean],
            "image_std": [float(value) for value in std]}


def resolve(model_dir: Path) -> dict:
    """Return the network dimensions and the processor statistics of one checkpoint."""
    model_dir = Path(model_dir)
    config = _model(model_dir / "config.json")
    config.update(_processor(model_dir / "preprocessor_config.json", config["image_size"]))
    return config
