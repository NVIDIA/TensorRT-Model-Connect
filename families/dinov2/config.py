# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the DINOv2 geometry out of the tensors, checking the file against them.

``config.json`` states more here than SigLIP's does, but not everything that
matters: ``intermediate_size`` is absent and implied by ``mlp_ratio``, and the
grid the position table was trained on is stated only indirectly through
``image_size``. Every number below is therefore derived from a tensor shape,
and whatever the file does state is checked against that shape and refused on
disagreement.
"""

from __future__ import annotations

import json
from pathlib import Path

# ImageNet statistics; only the fallback for a checkpoint shipping no
# preprocessor config.
_FALLBACK_IMAGE_MEAN = [0.485, 0.456, 0.406]
_FALLBACK_IMAGE_STD = [0.229, 0.224, 0.225]
_DEFAULT_EPS = 1e-6
_EXPECTED_ACTIVATION = "gelu"


def _shape(weights, key: str) -> tuple:
    if key not in weights:
        raise KeyError(f"dinov2 checkpoint is missing '{key}'")
    return tuple(int(v) for v in weights[key].shape)


def _agree(stated, derived, what: str) -> None:
    if stated is not None and int(stated) != int(derived):
        raise ValueError(
            f"dinov2 config.json says {what}={stated} but the tensors say {derived}")


def _count_layers(weights) -> int:
    needle = "encoder.layer."
    indices = set()
    for key in weights:
        if key.startswith(needle):
            indices.add(int(key[len(needle):].split(".", 1)[0]))
    if not indices:
        raise KeyError("dinov2 checkpoint has no encoder layers")
    if indices != set(range(len(indices))):
        raise ValueError("dinov2 encoder layer indices are not contiguous")
    return len(indices)


def resolve(model_dir: Path, weights: dict) -> dict:
    """Return the geometry both the builder and the runtime need."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    architectures = config.get("architectures")
    if architectures is not None and "Dinov2Model" not in architectures:
        raise NotImplementedError(f"dinov2 builds Dinov2Model, not {architectures}")

    if config.get("use_swiglu_ffn"):
        # The giant variant swaps the MLP for a gated one with a differently
        # shaped fc1. It would load and compute something else entirely.
        raise NotImplementedError("dinov2 does not support the SwiGLU feed-forward variant")
    if config.get("num_register_tokens"):
        # Registers are a different model_type with an extra tensor and extra
        # rows between the class token and the patches.
        raise NotImplementedError(
            "dinov2 does not support register tokens; that is dinov2_with_registers")

    activation = config.get("hidden_act", _EXPECTED_ACTIVATION)
    if activation != _EXPECTED_ACTIVATION:
        # "gelu" here is the exact error-function form, not the tanh
        # approximation that other vision towers use. They differ by about
        # 1e-3, which a cosine will not notice.
        raise NotImplementedError(
            f"dinov2 is qualified for the exact {_EXPECTED_ACTIVATION}, not {activation}")

    intermediate, hidden = _shape(weights, "encoder.layer.0.mlp.fc1.weight")
    _agree(config.get("hidden_size"), hidden, "hidden_size")
    _agree(config.get("intermediate_size"), intermediate, "intermediate_size")
    mlp_ratio = config.get("mlp_ratio")
    if mlp_ratio is not None and int(mlp_ratio) * hidden != intermediate:
        raise ValueError(
            f"dinov2 mlp_ratio {mlp_ratio} disagrees with the fc1 width {intermediate}")

    layers = _count_layers(weights)
    _agree(config.get("num_hidden_layers"), layers, "num_hidden_layers")

    heads = config.get("num_attention_heads")
    if heads is None:
        raise KeyError("dinov2 config.json must state num_attention_heads")
    heads = int(heads)
    if heads <= 0 or hidden % heads:
        raise ValueError(f"dinov2 width {hidden} is not divisible by {heads} heads")

    out_channels, channels, kernel_h, kernel_w = _shape(
        weights, "embeddings.patch_embeddings.projection.weight")
    if channels != 3:
        raise NotImplementedError(f"dinov2 takes 3-channel images, not {channels}")
    if kernel_h != kernel_w:
        raise NotImplementedError("dinov2 needs a square patch")
    if out_channels != hidden:
        raise ValueError("dinov2 patch embedding width disagrees with the encoder width")
    _agree(config.get("patch_size"), kernel_h, "patch_size")

    # One class row then one row per patch of the TRAINED grid, which is not
    # the grid the engine runs at: the table is interpolated down at build
    # time. The trained grid is read off the table rather than from the stated
    # image_size, and the stated size is then checked against it.
    positions, position_width = _shape(weights, "embeddings.position_embeddings")[1:]
    if position_width != hidden:
        raise ValueError("dinov2 position table width disagrees with the encoder width")
    trained_grid = int(round((positions - 1) ** 0.5))
    if trained_grid * trained_grid + 1 != positions:
        raise NotImplementedError(
            f"dinov2 needs one class row plus a square grid, not {positions} rows")
    _agree(config.get("image_size"), trained_grid * kernel_h, "image_size")

    preprocessor = model_dir / "preprocessor_config.json"
    image_mean, image_std = _FALLBACK_IMAGE_MEAN, _FALLBACK_IMAGE_STD
    shortest_edge, crop = 256, 224
    resample = 3
    if preprocessor.is_file():
        loaded = json.loads(preprocessor.read_text())
        image_mean = [float(v) for v in loaded.get("image_mean", image_mean)]
        image_std = [float(v) for v in loaded.get("image_std", image_std)]
        resample = int(loaded.get("resample", resample))
        size = loaded.get("size") or {}
        shortest_edge = int(size.get("shortest_edge", shortest_edge))
        crop_size = loaded.get("crop_size") or {}
        height = int(crop_size.get("height", crop))
        width = int(crop_size.get("width", crop))
        if height != width:
            raise NotImplementedError("dinov2 crops to a square")
        crop = height
        if not loaded.get("do_center_crop", True):
            raise NotImplementedError("dinov2 requires the centre crop its table was sized for")
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("dinov2 image statistics must have three channels")
    if any(v == 0 for v in image_std):
        raise ValueError("dinov2 image deviation must be nonzero")
    if resample != 3:
        raise NotImplementedError(f"dinov2 is qualified for bicubic resize (3), not {resample}")
    if crop % kernel_h:
        raise NotImplementedError(
            f"dinov2 crop {crop} is not a whole number of {kernel_h}-pixel patches")
    if shortest_edge < crop:
        raise ValueError("dinov2 resize must be at least the crop size")

    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "eps": float(config.get("layer_norm_eps", _DEFAULT_EPS)),
        "patch_size": kernel_h,
        "trained_grid": trained_grid,
        "grid": crop // kernel_h,
        "image_size": crop,
        "shortest_edge": shortest_edge,
        "image_mean": image_mean,
        "image_std": image_std,
        "resample": resample,
    }
