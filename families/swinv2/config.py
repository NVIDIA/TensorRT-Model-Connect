# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the SwinV2 geometry from config.json, checking every value against the tensors.

The engine is built for one fixed input size, so everything the reference
derives from that size is derived here once: the feature-map side of each
stage, and for each block its window and its shift. The reference does that
per block in ``Swinv2Layer._compute_window_shift``:

* the window is the configured window, cut down to the stage's side when the
  stage is smaller (``microsoft/swinv2-tiny-patch4-window16-256`` runs its last
  stage, 8 x 8, with an 8 x 8 window);
* every odd block shifts by half the **configured** window, unless the stage is
  no larger than its window, in which case no block of the stage shifts (the
  same checkpoint's third stage, 16 x 16 with a 16 x 16 window, never shifts).

Sizes the reference would pad - a side that a window or a 2 x 2 merge does not
divide - are refused rather than padded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ARCHITECTURE = "Swinv2ForImageClassification"
_EXPECTED_ACTIVATION = "gelu"
# Pillow's resampling filter number for bicubic, as the preprocessor config spells it.
_BICUBIC = 3
# The slow ViT image processor, which resizes with Pillow. Its fast variant
# resizes with torchvision instead and is not what the runtime reproduces.
_PROCESSORS = ("ViTImageProcessor", "ViTFeatureExtractor")
# What that processor uses when preprocessor_config.json leaves a value out
# (transformers v5.2.0 ViTImageProcessor.__init__): bilinear, 224 x 224, and
# IMAGENET_STANDARD_MEAN / _STD. Missing values take these and then meet the
# same checks as stated ones.
_PROCESSOR_DEFAULTS = {
    "resample": 2,
    "size": {"height": 224, "width": 224},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
# The reference builds the patch-embedding norm and the patch-merging norms with
# nn.LayerNorm's default epsilon, not with config.layer_norm_eps.
_DEFAULT_NORM_EPSILON = 1e-5
_BLOCK = re.compile(r"^swinv2\.encoder\.layers\.(\d+)\.blocks\.(\d+)\.")


def _shape(weights, key: str) -> tuple:
    if key not in weights:
        raise KeyError(f"swinv2 checkpoint is missing '{key}'")
    return tuple(int(v) for v in weights[key].shape)


def _agree(stated, derived, what: str) -> None:
    if stated is not None and stated != derived:
        raise ValueError(f"swinv2 config.json says {what}={stated} but the tensors say {derived}")


def _expect(weights, key: str, shape: tuple) -> None:
    found = _shape(weights, key)
    if found != shape:
        raise ValueError(f"swinv2 tensor {key} has shape {found}, expected {shape}")


def _square(value, what: str) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise NotImplementedError(f"swinv2 needs a square {what}, not {value}")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"swinv2 {what} must be a positive integer, not {value!r}")
    return int(value)


def _depths(weights) -> list[int]:
    blocks: dict[int, set[int]] = {}
    for key in weights:
        match = _BLOCK.match(key)
        if match:
            blocks.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    if not blocks:
        raise KeyError("swinv2 checkpoint has no swinv2.encoder.layers.<i>.blocks.<j> tensors")
    if sorted(blocks) != list(range(len(blocks))):
        raise ValueError("swinv2 stage indices are not contiguous")
    depths = []
    for stage in range(len(blocks)):
        if sorted(blocks[stage]) != list(range(len(blocks[stage]))):
            raise ValueError(f"swinv2 stage {stage} block indices are not contiguous")
        depths.append(len(blocks[stage]))
    return depths


def _labels(config: dict, num_classes: int) -> list[str]:
    labels = config.get("id2label") or {}
    if not labels:
        # An unnamed vocabulary is reported as ordinals rather than invented.
        return []
    if len(labels) != num_classes:
        raise ValueError(f"swinv2 states {len(labels)} labels but the classifier has {num_classes} rows")
    missing = [index for index in range(num_classes) if str(index) not in labels]
    if missing:
        raise ValueError(f"swinv2 id2label has no entry for class {missing[0]}")
    return [str(labels[str(index)]) for index in range(num_classes)]


def _preprocessing(model_dir: Path, image_size: int) -> dict:
    path = model_dir / "preprocessor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"swinv2 needs the checkpoint's preprocessor_config.json: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    kind = loaded.get("image_processor_type") or loaded.get("feature_extractor_type")
    if kind not in _PROCESSORS:
        raise NotImplementedError(f"swinv2 reproduces the ViT image processor, not {kind!r}")
    if loaded.get("do_center_crop") or not loaded.get("do_resize", True):
        raise NotImplementedError("swinv2 resizes the whole image to the square without a crop")
    if not loaded.get("do_rescale", True) or not loaded.get("do_normalize", True):
        raise NotImplementedError("swinv2 is qualified for rescaled and normalized pixels")
    rescale = float(loaded.get("rescale_factor", 1.0 / 255.0))
    if abs(rescale - 1.0 / 255.0) > 1e-12:
        raise NotImplementedError(f"swinv2 is qualified for a 1/255 rescale, not {rescale}")
    resample = int(loaded.get("resample", _PROCESSOR_DEFAULTS["resample"]))
    if resample != _BICUBIC:
        raise NotImplementedError(f"swinv2 is qualified for Pillow's bicubic resize, not {resample}")
    size = loaded.get("size", _PROCESSOR_DEFAULTS["size"])
    if isinstance(size, dict):
        if set(size) != {"height", "width"}:
            raise NotImplementedError(f"swinv2 resizes to a height and width, not {size}")
        height, width = int(size["height"]), int(size["width"])
    elif isinstance(size, int) and not isinstance(size, bool):
        height = width = size
    else:
        raise NotImplementedError(f"swinv2 cannot read a resize target of {size!r}")
    if height != width or height != image_size:
        # The engine runs at the checkpoint's image size; a different processed
        # size would change every stage's window geometry.
        raise NotImplementedError(
            f"swinv2 preprocessor resizes to {height}x{width}, the model runs at {image_size}")
    mean = [float(v) for v in loaded.get("image_mean", _PROCESSOR_DEFAULTS["image_mean"])]
    std = [float(v) for v in loaded.get("image_std", _PROCESSOR_DEFAULTS["image_std"])]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("swinv2 image statistics must have three channels")
    if any(value == 0.0 for value in std):
        raise ValueError("swinv2 image deviation must be nonzero")
    return {"image_mean": mean, "image_std": std, "resample": resample}


def _blocks(stages: list[dict], window_size: int) -> None:
    """Fill each stage's per-block windows and shifts, as the reference does."""
    for stage in stages:
        side = stage["side"]
        window = min(side, window_size)
        if side % window:
            raise NotImplementedError(
                f"swinv2 stage side {side} is not a multiple of its window {window}; "
                "the reference would pad it")
        stage["window"] = window
        stage["shifts"] = [
            0 if index % 2 == 0 or side <= window else window_size // 2
            for index in range(stage["depth"])
        ]


def resolve(model_dir: Path, weights: dict) -> dict:
    """Return the geometry and metadata both the builder and the runtime need."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "swinv2":
        raise ValueError(f"swinv2 reads model_type swinv2, not {config.get('model_type')!r}")
    architectures = config.get("architectures") or []
    if _ARCHITECTURE not in architectures:
        # Swinv2Model has no classifier, Swinv2ForMaskedImageModeling has a
        # pixel decoder and Swinv2Backbone returns feature maps; none of them
        # produces class scores.
        raise NotImplementedError(f"swinv2 builds {_ARCHITECTURE}, not {architectures}")
    activation = config.get("hidden_act", _EXPECTED_ACTIVATION)
    if activation != _EXPECTED_ACTIVATION:
        raise NotImplementedError(f"swinv2 is qualified for the exact GELU, not {activation}")
    if config.get("use_absolute_embeddings", False):
        raise NotImplementedError("swinv2 is qualified without absolute position embeddings")
    if not config.get("qkv_bias", True):
        raise NotImplementedError("swinv2 is qualified for biased query and value projections")

    embed_dim, channels, patch, patch_w = _shape(
        weights, "swinv2.embeddings.patch_embeddings.projection.weight")
    if channels != 3 or patch != patch_w:
        raise NotImplementedError("swinv2 takes 3-channel images and square patches")
    _agree(config.get("num_channels"), channels, "num_channels")
    _agree(config.get("embed_dim"), embed_dim, "embed_dim")
    _agree(_square(config.get("patch_size", patch), "patch_size"), patch, "patch_size")
    image_size = _square(config.get("image_size"), "image_size")
    if image_size % patch:
        raise NotImplementedError("swinv2 image size must be a multiple of the patch size")

    depths = _depths(weights)
    _agree(list(config.get("depths", depths)), depths, "depths")
    window_size = _square(config.get("window_size"), "window_size")
    pretrained = [int(v) for v in config.get("pretrained_window_sizes") or [0] * len(depths)]
    if len(pretrained) != len(depths):
        raise ValueError("swinv2 pretrained_window_sizes must name one window per stage")

    stages = []
    side, dim = image_size // patch, embed_dim
    stated_heads = list(config.get("num_heads", []))
    for index, depth in enumerate(depths):
        prefix = f"swinv2.encoder.layers.{index}"
        heads = _shape(weights, f"{prefix}.blocks.0.attention.self.logit_scale")[0]
        if stated_heads:
            _agree(stated_heads[index], heads, f"num_heads[{index}]")
        if dim % heads:
            raise ValueError(f"swinv2 stage {index} width {dim} is not divisible by {heads} heads")
        _expect(weights, f"{prefix}.blocks.0.attention.self.query.weight", (dim, dim))
        hidden = _shape(weights, f"{prefix}.blocks.0.intermediate.dense.weight")[0]
        stage = {"depth": depth, "side": side, "dim": dim, "heads": heads, "mlp": hidden,
                 "pretrained_window": pretrained[index]}
        last = index == len(depths) - 1
        merges = f"{prefix}.downsample.reduction.weight" in weights
        if merges == last:
            raise ValueError(f"swinv2 stage {index} {'has' if merges else 'lacks'} a patch merge")
        if merges:
            if side % 2:
                raise NotImplementedError("swinv2 patch merging of an odd side would pad")
            _expect(weights, f"{prefix}.downsample.reduction.weight", (2 * dim, 4 * dim))
        stages.append(stage)
        if merges:
            side, dim = side // 2, 2 * dim
    _blocks(stages, window_size)
    _agree(config.get("num_layers"), len(depths), "num_layers")
    _agree(config.get("hidden_size"), stages[-1]["dim"], "hidden_size")

    ratio = config.get("mlp_ratio")
    if ratio is not None and any(stage["mlp"] != int(float(ratio) * stage["dim"]) for stage in stages):
        raise ValueError("swinv2 config.json mlp_ratio disagrees with the MLP widths")

    num_classes, width = _shape(weights, "classifier.weight")
    if width != stages[-1]["dim"]:
        raise ValueError("swinv2 classifier width disagrees with the last stage")
    vocabulary_id = config.get("vocabulary_id", "")
    if not isinstance(vocabulary_id, str):
        raise ValueError("swinv2 vocabulary_id must be a string")

    return {
        "image_size": image_size,
        "patch_size": patch,
        "embed_dim": embed_dim,
        "stages": stages,
        "eps": float(config.get("layer_norm_eps", 1e-5)),
        "norm_eps": _DEFAULT_NORM_EPSILON,
        "num_classes": num_classes,
        "labels": _labels(config, num_classes),
        # Named only if the checkpoint names it: a thousand classes is not proof
        # of ImageNet-1k, and a wrong vocabulary name claims comparability.
        "vocabulary_id": vocabulary_id,
        **_preprocessing(model_dir, image_size),
    }
