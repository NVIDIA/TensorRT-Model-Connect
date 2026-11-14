# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the SigLIP 2 NaFlex geometry out of the tensors and the processor file.

The published ``config.json`` states the architecture, the sub-model types and the
text vocabulary size, nothing else: every width, depth, head count, activation and
epsilon is supplied by the reference library's defaults. Every number a tensor shape
fixes is therefore read from that shape, whatever the file does state is checked
against it and refused on disagreement, and the few numbers no shape carries (head
count, activation, epsilon) are the reference's documented defaults unless the file
states them.

NaFlex has no fixed input size. The processor resizes each image, keeping its aspect
ratio, to the largest multiple of the patch size whose patch count fits
``max_num_patches``, and the vision tower interpolates its square position table to
that grid. Both numbers live in ``preprocessor_config.json``; the table's grid comes
from the tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

# Reference defaults (Siglip2TextConfig / Siglip2VisionConfig) for numbers that no
# tensor shape carries. A head count is not derivable from any projection, which is
# [width, width] whatever the count, so a file that omits it gets the reference's 12.
_REFERENCE_HEADS = 12
_EXPECTED_ACTIVATION = "gelu_pytorch_tanh"
_DEFAULT_EPS = 1e-6
_CHANNELS = 3

# Siglip2ImageProcessor defaults, used only when preprocessor_config.json omits a key.
_DEFAULT_PATCH = 16
_DEFAULT_MAX_PATCHES = 256
_DEFAULT_MEAN = [0.5, 0.5, 0.5]
_DEFAULT_STD = [0.5, 0.5, 0.5]
# PIL's enumeration: 2 is bilinear. The runtime reproduces Pillow's 8-bit bilinear
# resize exactly, and nothing else.
_BILINEAR = 2
_RESCALE = 1.0 / 255.0
_PROCESSORS = {"Siglip2ImageProcessor", "Siglip2ImageProcessorFast"}


def _shape(weights, key: str) -> tuple:
    if key not in weights:
        raise KeyError(f"siglip2 checkpoint is missing '{key}'")
    return tuple(int(v) for v in weights[key].shape)


def _agree(stated, derived, what: str) -> None:
    """Refuse a config.json claim that the tensors contradict."""
    if stated is not None and int(stated) != int(derived):
        raise ValueError(
            f"siglip2 config.json says {what}={stated} but the tensors say {derived}")


def _count_layers(weights, prefix: str) -> int:
    indices = set()
    needle = f"{prefix}.encoder.layers."
    for key in weights:
        if key.startswith(needle):
            indices.add(int(key[len(needle):].split(".", 1)[0]))
    if not indices:
        raise KeyError(f"siglip2 checkpoint has no {prefix} encoder layers")
    if indices != set(range(len(indices))):
        raise ValueError(f"siglip2 {prefix} encoder layer indices are not contiguous")
    return len(indices)


def _tower(weights, config, prefix: str, name: str) -> dict:
    stated = config.get(name) or {}

    # fc1 is [intermediate, hidden]; one tensor fixes both widths at once.
    intermediate, hidden = _shape(weights, f"{prefix}.encoder.layers.0.mlp.fc1.weight")
    _agree(stated.get("hidden_size"), hidden, f"{name}.hidden_size")
    _agree(stated.get("intermediate_size"), intermediate, f"{name}.intermediate_size")

    layers = _count_layers(weights, prefix)
    _agree(stated.get("num_hidden_layers"), layers, f"{name}.num_hidden_layers")

    heads = int(stated.get("num_attention_heads", _REFERENCE_HEADS))
    if heads <= 0 or hidden % heads:
        raise ValueError(f"siglip2 {name} width {hidden} is not divisible by {heads} heads")

    activation = stated.get("hidden_act", _EXPECTED_ACTIVATION)
    if activation != _EXPECTED_ACTIVATION:
        # The tanh approximation and the exact GELU agree to about 1e-3, which moves
        # the logits; they are not interchangeable.
        raise NotImplementedError(
            f"siglip2 {name} is qualified for {_EXPECTED_ACTIVATION}, not {activation}")

    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "eps": float(stated.get("layer_norm_eps", _DEFAULT_EPS)),
    }


def _text(weights, config) -> dict:
    text = _tower(weights, config, "text_model", "text_config")
    stated = config.get("text_config") or {}

    vocab_size, width = _shape(weights, "text_model.embeddings.token_embedding.weight")
    _agree(stated.get("vocab_size"), vocab_size, "text_config.vocab_size")
    if width != text["hidden_size"]:
        raise ValueError("siglip2 token embedding width disagrees with the encoder width")
    text["vocab_size"] = vocab_size

    # The position table is the whole prompt window, and the text tower pools the LAST
    # row of that window: its length is a correctness rule, not a capacity.
    window, width = _shape(weights, "text_model.embeddings.position_embedding.weight")
    _agree(stated.get("max_position_embeddings"), window, "text_config.max_position_embeddings")
    if width != text["hidden_size"]:
        raise ValueError("siglip2 text position table width disagrees with the encoder width")
    text["max_text_len"] = window

    projection, width = _shape(weights, "text_model.head.weight")
    _agree(stated.get("projection_size"), projection, "text_config.projection_size")
    if width != text["hidden_size"]:
        raise ValueError("siglip2 text head input width disagrees with the encoder width")
    text["projection_size"] = projection
    return text


def _preprocessor(model_dir: Path) -> dict:
    path = model_dir / "preprocessor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing siglip2 image processor config: {path}")
    loaded = json.loads(path.read_text())

    kind = loaded.get("image_processor_type")
    if kind is not None and kind not in _PROCESSORS:
        raise NotImplementedError(f"siglip2 implements the Siglip2 image processor, not {kind}")
    if not loaded.get("do_resize", True):
        # NaFlex needs every side to be a whole number of patches.
        raise NotImplementedError("siglip2 requires the processor's aspect-preserving resize")
    resample = int(loaded.get("resample", _BILINEAR))
    if resample != _BILINEAR:
        raise NotImplementedError(
            f"siglip2 is qualified for Pillow bilinear resize ({_BILINEAR}), not {resample}")
    if not loaded.get("do_rescale", True):
        raise NotImplementedError("siglip2 requires the processor's 1/255 rescale")
    rescale = float(loaded.get("rescale_factor", _RESCALE))
    if abs(rescale - _RESCALE) > 1e-12:
        raise NotImplementedError(f"siglip2 is qualified for a 1/255 rescale, not {rescale}")
    if not loaded.get("do_normalize", True):
        raise NotImplementedError("siglip2 requires the processor's mean/std normalization")

    mean = [float(v) for v in loaded.get("image_mean", _DEFAULT_MEAN)]
    std = [float(v) for v in loaded.get("image_std", _DEFAULT_STD)]
    if len(mean) != _CHANNELS or len(std) != _CHANNELS:
        raise ValueError("siglip2 image statistics must have three channels")
    if any(v == 0 for v in std):
        raise ValueError("siglip2 image deviation must be nonzero")

    patch = int(loaded.get("patch_size", _DEFAULT_PATCH))
    max_patches = int(loaded.get("max_num_patches", _DEFAULT_MAX_PATCHES))
    if patch <= 0 or max_patches <= 0:
        raise ValueError("siglip2 patch size and patch budget must be positive")
    return {
        "image_mean": mean,
        "image_std": std,
        "resample": resample,
        "rescale_factor": rescale,
        "patch_size": patch,
        "max_num_patches": max_patches,
    }


def _vision(weights, config, processor: dict) -> dict:
    vision = _tower(weights, config, "vision_model", "vision_config")
    stated = config.get("vision_config") or {}
    width = vision["hidden_size"]

    channels = int(stated.get("num_channels", _CHANNELS))
    if channels != _CHANNELS:
        raise NotImplementedError(f"siglip2 takes 3-channel images, not {channels}")

    # NaFlex embeds already-flattened patches with a Linear layer. Its input width is
    # channels x patch x patch in (row, column, channel) order, so it fixes the patch
    # size the processor must cut; the two files are checked against it.
    out_features, in_features = _shape(weights, "vision_model.embeddings.patch_embedding.weight")
    if out_features != width:
        raise ValueError("siglip2 patch embedding width disagrees with the encoder width")
    patch = processor["patch_size"]
    if in_features != _CHANNELS * patch * patch:
        raise ValueError(
            f"siglip2 patch embedding takes {in_features} values per patch, not the "
            f"{_CHANNELS} x {patch} x {patch} the processor cuts")
    _agree(stated.get("patch_size"), patch, "vision_config.patch_size")
    vision["patch_size"] = patch
    vision["patch_features"] = in_features

    # The position table is square (the reference reshapes it to grid x grid before
    # interpolating), and its row count is not the patch budget: the budget is how many
    # patches an image may have, the table is resized to whatever grid it gets.
    positions, position_width = _shape(weights, "vision_model.embeddings.position_embedding.weight")
    if position_width != width:
        raise ValueError("siglip2 position table width disagrees with the encoder width")
    grid = int(round(positions ** 0.5))
    if grid * grid != positions:
        raise NotImplementedError(f"siglip2 needs a square position table, not {positions} rows")
    _agree(stated.get("num_patches"), positions, "vision_config.num_patches")
    vision["num_positions"] = positions
    vision["grid"] = grid
    vision["max_num_patches"] = processor["max_num_patches"]

    # The pooling head is a torch MultiheadAttention: its projections arrive packed as
    # one [3 * width, width] tensor in q, k, v order.
    packed, packed_width = _shape(weights, "vision_model.head.attention.in_proj_weight")
    if packed_width != width or packed != 3 * width:
        raise ValueError("siglip2 pooling head projection is not a packed QKV tensor")
    probe = _shape(weights, "vision_model.head.probe")
    if probe != (1, 1, width):
        raise ValueError(f"siglip2 pooling probe has shape {probe}, expected (1, 1, {width})")
    if stated.get("vision_use_head") is False:
        raise NotImplementedError("siglip2 embeds images through the attention-pooling head")
    return vision


def resolve(model_dir: Path, weights: dict) -> dict:
    """Return the geometry both builders and the runtime need."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    architectures = config.get("architectures")
    if architectures is not None and "Siglip2Model" not in architectures:
        # Siglip2ForImageClassification shares the model type but replaces the
        # pooling head and the text tower with a classifier.
        raise NotImplementedError(f"siglip2 builds Siglip2Model, not {architectures}")

    processor = _preprocessor(model_dir)
    text = _text(weights, config)
    vision = _vision(weights, config, processor)
    if text["projection_size"] != vision["hidden_size"]:
        raise ValueError("siglip2 text and image embeddings have different widths")
    return {
        "text": text,
        "vision": vision,
        "image_mean": processor["image_mean"],
        "image_std": processor["image_std"],
        "resample": processor["resample"],
        "rescale_factor": processor["rescale_factor"],
    }
