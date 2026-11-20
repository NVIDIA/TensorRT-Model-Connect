# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the ViT geometry out of the tensors, checking the file against them.

This checkpoint states its geometry fully, which makes it tempting to read
and trust. Two of the stated values differ from what the other ViTs in this
repo use, and they are worth very different amounts:

* ``layer_norm_eps`` is **1e-12** here, where CLIP uses 1e-5 and SigLIP and
  DINOv2 use 1e-6. The value is four to seven orders of magnitude apart, but
  measured on this checkpoint the *output* barely moves: substituting 1e-6
  shifts the logits by 2.2e-05 and 1e-5 by 2.7e-04, against a correct build's
  own 1.0e-05 of numerical noise. It is read from the file because that is
  what the file is for, not because a wrong value would be caught.
* the resize is **bilinear**, where every other image family here reproduces
  Pillow's bicubic. That one does matter: bicubic moves the logits by 0.376
  and changes the top-5.

So the epsilon is read and the filter is read, and neither is assumed - but
only one of them is a rule the gate can defend.
"""
from __future__ import annotations

import json
from pathlib import Path

_FALLBACK_IMAGE_MEAN = [0.5, 0.5, 0.5]
_FALLBACK_IMAGE_STD = [0.5, 0.5, 0.5]
_EXPECTED_ACTIVATION = "gelu"
# Pillow's resampling filter numbers, as the preprocessor config spells them.
_BILINEAR, _BICUBIC = 2, 3


def _shape(weights, key: str) -> tuple:
    if key not in weights:
        raise KeyError(f"vit checkpoint is missing '{key}'")
    return tuple(int(v) for v in weights[key].shape)


def _agree(stated, derived, what: str) -> None:
    if stated is not None and int(stated) != int(derived):
        raise ValueError(f"vit config.json says {what}={stated} but the tensors say {derived}")


def _count_layers(weights, prefix: str) -> int:
    needle = f"{prefix}encoder.layer."
    indices = set()
    for key in weights:
        if key.startswith(needle):
            indices.add(int(key[len(needle):].split(".", 1)[0]))
    if not indices:
        raise KeyError("vit checkpoint has no encoder layers")
    if indices != set(range(len(indices))):
        raise ValueError("vit encoder layer indices are not contiguous")
    return len(indices)


def resolve(model_dir: Path, weights: dict) -> dict:
    """Return the geometry both the builder and the runtime need."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    architectures = config.get("architectures")
    if architectures is not None and "ViTForImageClassification" not in architectures:
        # ViTModel has no classifier and ViTForMaskedImageModeling has a
        # decoder instead; neither produces class scores.
        raise NotImplementedError(
            f"vit builds ViTForImageClassification, not {architectures}")

    activation = config.get("hidden_act", _EXPECTED_ACTIVATION)
    if activation != _EXPECTED_ACTIVATION:
        raise NotImplementedError(
            f"vit is qualified for the exact {_EXPECTED_ACTIVATION}, not {activation}")
    if config.get("qkv_bias") is False:
        raise NotImplementedError("vit is qualified for a biased QKV projection")

    prefix = "vit." if any(key.startswith("vit.") for key in weights) else ""
    intermediate, hidden = _shape(weights, f"{prefix}encoder.layer.0.intermediate.dense.weight")
    _agree(config.get("hidden_size"), hidden, "hidden_size")
    _agree(config.get("intermediate_size"), intermediate, "intermediate_size")

    layers = _count_layers(weights, prefix)
    _agree(config.get("num_hidden_layers"), layers, "num_hidden_layers")

    heads = config.get("num_attention_heads")
    if heads is None:
        raise KeyError("vit config.json must state num_attention_heads")
    heads = int(heads)
    if heads <= 0 or hidden % heads:
        raise ValueError(f"vit width {hidden} is not divisible by {heads} heads")

    out_channels, channels, kernel_h, kernel_w = _shape(
        weights, f"{prefix}embeddings.patch_embeddings.projection.weight")
    if channels != 3:
        raise NotImplementedError(f"vit takes 3-channel images, not {channels}")
    if kernel_h != kernel_w:
        raise NotImplementedError("vit needs a square patch")
    if out_channels != hidden:
        raise ValueError("vit patch embedding width disagrees with the encoder width")
    _agree(config.get("patch_size"), kernel_h, "patch_size")

    # One class row then one row per patch. Unlike DINOv2's, this table
    # already matches the grid the engine runs at, so nothing is resized -
    # and a size that would need resizing is refused rather than guessed.
    positions, position_width = _shape(weights, f"{prefix}embeddings.position_embeddings")[1:]
    if position_width != hidden:
        raise ValueError("vit position table width disagrees with the encoder width")
    grid = int(round((positions - 1) ** 0.5))
    if grid * grid + 1 != positions:
        raise NotImplementedError(
            f"vit needs one class row plus a square grid, not {positions} rows")
    image_size = grid * kernel_h
    _agree(config.get("image_size"), image_size, "image_size")

    num_classes, classifier_width = _shape(weights, "classifier.weight")
    if classifier_width != hidden:
        raise ValueError("vit classifier width disagrees with the encoder width")
    labels = config.get("id2label") or {}
    if labels:
        if len(labels) != num_classes:
            raise ValueError(
                f"vit states {len(labels)} labels but the classifier has {num_classes} rows")
        missing = [index for index in range(num_classes) if str(index) not in labels]
        if missing:
            raise ValueError(f"vit id2label has no entry for class {missing[0]}")
        class_labels = [str(labels[str(index)]) for index in range(num_classes)]
    else:
        # An unnamed vocabulary is reported as ordinals rather than invented.
        class_labels = []

    # Named only if the checkpoint names it. A thousand classes is not proof
    # of ImageNet-1k, and a wrong vocabulary name is worse than none: it
    # claims the scores can be compared with another model's.
    vocabulary_id = config.get("vocabulary_id", "")
    if not isinstance(vocabulary_id, str):
        raise ValueError("vit vocabulary_id must be a string")

    preprocessor = model_dir / "preprocessor_config.json"
    image_mean, image_std = _FALLBACK_IMAGE_MEAN, _FALLBACK_IMAGE_STD
    resample = _BILINEAR
    crop = image_size
    if preprocessor.is_file():
        loaded = json.loads(preprocessor.read_text())
        image_mean = [float(v) for v in loaded.get("image_mean", image_mean)]
        image_std = [float(v) for v in loaded.get("image_std", image_std)]
        resample = int(loaded.get("resample", resample))
        # "size" is written either as a bare number or as a {height, width}
        # pair depending on when the checkpoint was published; the reference
        # processor normalises them to the same thing and so does this.
        size = loaded.get("size", crop)
        if isinstance(size, dict):
            height = int(size.get("height", size.get("shortest_edge", crop)))
            width = int(size.get("width", size.get("shortest_edge", crop)))
        elif isinstance(size, (int, float)):
            height = width = int(size)
        else:
            raise NotImplementedError(f"vit cannot read a size of {size!r}")
        if height != width:
            raise NotImplementedError("vit resizes to a square")
        crop = height
        if loaded.get("do_center_crop"):
            raise NotImplementedError("vit squashes to the square; this checkpoint asks for a crop")
        if not loaded.get("do_resize", True):
            raise NotImplementedError("vit requires a resize to the trained grid")
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("vit image statistics must have three channels")
    if any(v == 0 for v in image_std):
        raise ValueError("vit image deviation must be nonzero")
    if resample not in (_BILINEAR, _BICUBIC):
        raise NotImplementedError(
            f"vit is qualified for bilinear and bicubic resize, not {resample}")
    if crop != image_size:
        # The position table has one row per patch of the trained grid, so a
        # different processed size would need the table interpolated.
        raise NotImplementedError(
            f"vit preprocessor resizes to {crop} but the position table is sized for {image_size}")

    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": hidden // heads,
        # Measured on google/vit-base-patch16-224: 1e-12. Unusual, but worth
        # only 2.2e-05 of logit movement if it were 1e-6 - see the module
        # docstring. Read, not assumed; not load-bearing.
        "eps": float(config.get("layer_norm_eps", 1e-12)),
        "prefix": prefix,
        "patch_size": kernel_h,
        "grid": grid,
        "image_size": image_size,
        "num_classes": num_classes,
        "labels": class_labels,
        "vocabulary_id": vocabulary_id,
        "image_mean": image_mean,
        "image_std": image_std,
        "resample": resample,
    }
