# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the SigLIP geometry out of the tensors, not out of ``config.json``.

This checkpoint's ``config.json`` states almost nothing. Measured on
``google/siglip-base-patch16-224``, only ``vision_config.patch_size`` and four
text keys are present; ``hidden_size``, ``num_hidden_layers``,
``num_attention_heads``, ``image_size``, ``hidden_act`` and ``layer_norm_eps``
are all supplied by the reference library's defaults. A builder that trusts
the file gets whatever that library's defaults happen to be on the day, and a
builder that hard-codes the base geometry silently mis-builds every other
size.

So every number here is derived from a tensor shape, and whatever the file
*does* state is checked against that shape and refused on disagreement. The
file is treated as a claim; the tensors are the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

# The reference applies these before the tower; they belong to the checkpoint.
_FALLBACK_IMAGE_MEAN = [0.5, 0.5, 0.5]
_FALLBACK_IMAGE_STD = [0.5, 0.5, 0.5]

# transformers resolves both towers to this when config.json omits it, and
# every published SigLIP checkpoint has been trained with it. It is not
# derivable from a tensor shape, so it is named here and refused if the file
# disagrees rather than silently accepted.
_EXPECTED_ACTIVATION = "gelu_pytorch_tanh"
_DEFAULT_EPS = 1e-6


def _shape(weights, key: str) -> tuple:
    if key not in weights:
        raise KeyError(f"siglip checkpoint is missing '{key}'")
    return tuple(int(v) for v in weights[key].shape)


def _agree(stated, derived, what: str) -> None:
    """Refuse a config.json claim that the tensors contradict."""
    if stated is not None and int(stated) != int(derived):
        raise ValueError(
            f"siglip config.json says {what}={stated} but the tensors say {derived}")


def _count_layers(weights, prefix: str) -> int:
    indices = set()
    needle = f"{prefix}.encoder.layers."
    for key in weights:
        if key.startswith(needle):
            indices.add(int(key[len(needle):].split(".", 1)[0]))
    if not indices:
        raise KeyError(f"siglip checkpoint has no {prefix} encoder layers")
    if indices != set(range(len(indices))):
        raise ValueError(f"siglip {prefix} encoder layer indices are not contiguous")
    return len(indices)


def _tower(weights, config, prefix: str, name: str) -> dict:
    stated = config.get(name, {})

    # fc1 is [intermediate, hidden]; one tensor fixes both widths at once.
    intermediate, hidden = _shape(weights, f"{prefix}.encoder.layers.0.mlp.fc1.weight")
    _agree(stated.get("hidden_size"), hidden, f"{name}.hidden_size")
    _agree(stated.get("intermediate_size"), intermediate, f"{name}.intermediate_size")

    layers = _count_layers(weights, prefix)
    _agree(stated.get("num_hidden_layers"), layers, f"{name}.num_hidden_layers")

    # Head count is the one number no tensor shape carries: every projection
    # is [hidden, hidden] whatever the head count. It is read from the file
    # when stated. Otherwise the reference's rule of a 64-wide head applies,
    # and a width that rule cannot divide is refused rather than guessed -
    # a wrong head count reshapes the same weights into a different model.
    heads = stated.get("num_attention_heads")
    if heads is None:
        if hidden % 64:
            raise ValueError(
                f"siglip {name} states no num_attention_heads and its width {hidden} is "
                "not a whole number of 64-wide heads")
        heads = hidden // 64
    heads = int(heads)
    if heads <= 0 or hidden % heads:
        raise ValueError(f"siglip {name} width {hidden} is not divisible by {heads} heads")

    activation = stated.get("hidden_act", _EXPECTED_ACTIVATION)
    if activation != _EXPECTED_ACTIVATION:
        # The tanh approximation and the exact GELU agree to about 1e-3, which
        # survives a cosine and moves the logits. They are not interchangeable.
        raise NotImplementedError(
            f"siglip {name} is qualified for {_EXPECTED_ACTIVATION}, not {activation}")

    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "eps": float(stated.get("layer_norm_eps", _DEFAULT_EPS)),
    }


def resolve(model_dir: Path, weights: dict) -> dict:
    """Return the geometry both builders and the runtime need."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    architectures = config.get("architectures")
    if architectures is not None and "SiglipModel" not in architectures:
        raise NotImplementedError(
            f"siglip builds SiglipModel, not {architectures}")

    text = _tower(weights, config, "text_model", "text_config")
    vision = _tower(weights, config, "vision_model", "vision_config")

    vocab_size, text_width = _shape(weights, "text_model.embeddings.token_embedding.weight")
    _agree(config.get("text_config", {}).get("vocab_size"), vocab_size, "text_config.vocab_size")
    if text_width != text["hidden_size"]:
        raise ValueError("siglip token embedding width disagrees with the encoder width")
    text["vocab_size"] = vocab_size

    # The position table is the whole prompt window. The text tower pools the
    # LAST row of the window, so this length is a correctness rule and not a
    # capacity: measured on the base checkpoint, shortening the window from 64
    # to 32 moves the pooled embedding by 1.98 (cosine 0.863).
    max_text_len, _ = _shape(weights, "text_model.embeddings.position_embedding.weight")
    _agree(config.get("text_config", {}).get("max_position_embeddings"), max_text_len,
           "text_config.max_position_embeddings")
    text["max_text_len"] = max_text_len
    text["projection_size"] = _shape(weights, "text_model.head.weight")[0]

    out_channels, channels, kernel_h, kernel_w = _shape(
        weights, "vision_model.embeddings.patch_embedding.weight")
    if channels != 3:
        raise NotImplementedError(f"siglip takes 3-channel images, not {channels}")
    if kernel_h != kernel_w:
        raise NotImplementedError("siglip needs a square patch")
    if out_channels != vision["hidden_size"]:
        raise ValueError("siglip patch embedding width disagrees with the encoder width")
    _agree(config.get("vision_config", {}).get("patch_size"), kernel_h, "vision_config.patch_size")
    vision["patch_size"] = kernel_h

    # No class token: the position table has exactly one row per patch, so the
    # grid - and with it the only input size this engine is valid for - is read
    # off the table rather than off the stated image_size.
    num_positions, position_width = _shape(
        weights, "vision_model.embeddings.position_embedding.weight")
    if position_width != vision["hidden_size"]:
        raise ValueError("siglip position table width disagrees with the encoder width")
    grid = int(round(num_positions ** 0.5))
    if grid * grid != num_positions:
        raise NotImplementedError(
            f"siglip needs a square patch grid, not {num_positions} positions")
    vision["num_positions"] = num_positions
    vision["grid"] = grid
    vision["image_size"] = grid * kernel_h
    _agree(config.get("vision_config", {}).get("image_size"), vision["image_size"],
           "vision_config.image_size")

    # The pooling head is a torch MultiheadAttention, so its projections arrive
    # packed as one [3 * width, width] tensor rather than as q/k/v.
    packed, packed_width = _shape(weights, "vision_model.head.attention.in_proj_weight")
    if packed_width != vision["hidden_size"] or packed != 3 * vision["hidden_size"]:
        raise ValueError("siglip pooling head projection is not a packed QKV tensor")

    preprocessor = model_dir / "preprocessor_config.json"
    image_mean, image_std = _FALLBACK_IMAGE_MEAN, _FALLBACK_IMAGE_STD
    resample = 3
    if preprocessor.is_file():
        loaded = json.loads(preprocessor.read_text())
        image_mean = [float(v) for v in loaded.get("image_mean", image_mean)]
        image_std = [float(v) for v in loaded.get("image_std", image_std)]
        resample = int(loaded.get("resample", resample))
        if loaded.get("do_center_crop"):
            raise NotImplementedError("siglip does not centre-crop; this checkpoint asks for it")
        if not loaded.get("do_resize", True):
            raise NotImplementedError("siglip requires a resize to the trained grid")
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("siglip image statistics must have three channels")
    if any(v == 0 for v in image_std):
        raise ValueError("siglip image deviation must be nonzero")
    # SigLIP squashes to a square; it does not preserve aspect ratio. 3 is
    # bicubic and 2 is bilinear in the reference's enumeration, and the two
    # checkpoint generations do not agree on which, so it is carried rather
    # than assumed.
    if resample != 3:
        # 3 is Pillow's bicubic, which the runtime reproduces coefficient for
        # coefficient. SigLIP 2 asks for 2 (bilinear); resampling it with the
        # wrong filter would move every pixel a little and nothing loudly.
        raise NotImplementedError(
            f"siglip is qualified for bicubic resize (3), not {resample}")

    return {
        "text": text,
        "vision": vision,
        "image_mean": image_mean,
        "image_std": image_std,
        "resample": resample,
    }
