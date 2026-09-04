# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT builder for Hugging Face DINOv3 ConvNeXt encoders.

This module mirrors ``DINOv3ConvNextModel`` with TensorRT Network API calls.
It deliberately owns its checkpoint mapping and graph construction and does
not use ONNX or import helpers from another model family.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import tensorrt as trt

from .checkpoint_mapper import (
    WeightDict,
    as_weight,
    load_first,
    open_checkpoint,
    target_dtype,
    transpose_linear,
)


_DEFAULT_HIDDEN_SIZES = (96, 192, 384, 768)
_DEFAULT_DEPTHS = (3, 3, 9, 3)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _raw_config(config_or_raw: object) -> dict[str, Any]:
    if isinstance(config_or_raw, dict):
        return config_or_raw
    raw = getattr(config_or_raw, "raw", None)
    if isinstance(raw, dict):
        return raw
    raise TypeError("DINOv3 ConvNeXt config must be a dict or expose a dict .raw")


def _positive_ints(value: object, *, name: str, default: Sequence[int]) -> tuple[int, ...]:
    selected = default if value is None else value
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)) or not selected:
        raise ValueError(f"DINOv3 ConvNeXt {name} must be a non-empty integer sequence")
    result: list[int] = []
    for item in selected:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"DINOv3 ConvNeXt {name} must contain positive integers")
        result.append(int(item))
    return tuple(result)


def _image_shape(value: object) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        height = width = int(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError("DINOv3 ConvNeXt image_size must be an int or a pair")
    if height <= 0 or width <= 0:
        raise ValueError("DINOv3 ConvNeXt image dimensions must be positive")
    return height, width


def _downsampled_size(size: int, num_stages: int) -> int:
    # Stage 0 is a 4x4/stride-4 stem; every later stage is 2x2/stride-2.
    result = size // 4
    for _ in range(1, num_stages):
        result //= 2
    return result


def resolve_convnext_config(config_or_raw: object) -> dict[str, Any]:
    """Resolve and validate the architecture fields used by official configs."""
    raw = _raw_config(config_or_raw)
    cached = raw.get("_dinov3_convnext_config")
    if isinstance(cached, dict):
        return cached

    model_type = str(raw.get("model_type", "dinov3_convnext") or "dinov3_convnext")
    if model_type != "dinov3_convnext":
        raise ValueError(
            f"DINOv3 ConvNeXt builder requires model_type='dinov3_convnext', got {model_type!r}"
        )
    hidden_sizes = _positive_ints(
        raw.get("hidden_sizes"), name="hidden_sizes", default=_DEFAULT_HIDDEN_SIZES
    )
    depths = _positive_ints(raw.get("depths"), name="depths", default=_DEFAULT_DEPTHS)
    if len(hidden_sizes) != len(depths):
        raise ValueError(
            "DINOv3 ConvNeXt hidden_sizes and depths must have the same length "
            f"({len(hidden_sizes)} vs {len(depths)})"
        )
    image_h, image_w = _image_shape(raw.get("image_size", 224))
    grid_h = _downsampled_size(image_h, len(hidden_sizes))
    grid_w = _downsampled_size(image_w, len(hidden_sizes))
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "DINOv3 ConvNeXt image_size is too small for its downsampling stages: "
            f"{image_h}x{image_w}, stages={len(hidden_sizes)}"
        )
    num_channels = int(raw.get("num_channels", 3))
    if num_channels <= 0:
        raise ValueError("DINOv3 ConvNeXt num_channels must be positive")
    eps = float(raw.get("layer_norm_eps", 1.0e-6))
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("DINOv3 ConvNeXt layer_norm_eps must be finite and positive")

    resolved = {
        "model_type": model_type,
        "num_channels": num_channels,
        "hidden_sizes": hidden_sizes,
        "depths": depths,
        "hidden_act": str(raw.get("hidden_act", "gelu")),
        "layer_norm_eps": eps,
        "image_h": image_h,
        "image_w": image_w,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_layers": sum(depths),
        "output_dim": hidden_sizes[-1],
        "num_tokens": 1 + grid_h * grid_w,
    }
    raw["_dinov3_convnext_config"] = resolved
    return resolved


def _load_hf_tensor(readers, suffix: str) -> np.ndarray:
    """Load prefixed or unprefixed Transformers weights."""
    return load_first(readers, f"model.{suffix}", suffix)


def _expect_shape(value: np.ndarray, expected: tuple[int, ...], name: str) -> np.ndarray:
    if tuple(value.shape) != expected:
        raise ValueError(
            f"Unexpected DINOv3 ConvNeXt tensor shape for {name}: "
            f"expected {expected}, got {tuple(value.shape)}"
        )
    return value


def load_convnext_weights(
    model_dir: str | Path,
    config_or_raw: object,
    *,
    precision: str = "fp32",
) -> WeightDict:
    """Load official DINOv3 ConvNeXt safetensors into canonical graph layout."""
    cfg = resolve_convnext_config(config_or_raw)
    # Validate precision before opening a potentially large sharded checkpoint.
    target_dtype(precision)
    readers = open_checkpoint(model_dir)
    weights = WeightDict()
    hidden_sizes = cfg["hidden_sizes"]
    depths = cfg["depths"]

    def store(logical: str, checkpoint: str, shape: tuple[int, ...], *, transpose=False):
        value = _expect_shape(_load_hf_tensor(readers, checkpoint), shape, checkpoint)
        weights[logical] = (
            transpose_linear(value, checkpoint, precision)
            if transpose
            else as_weight(value, precision)
        )

    for stage_idx, (channels, depth) in enumerate(zip(hidden_sizes, depths, strict=True)):
        in_channels = cfg["num_channels"] if stage_idx == 0 else hidden_sizes[stage_idx - 1]
        conv_index = 0 if stage_idx == 0 else 1
        norm_index = 1 if stage_idx == 0 else 0
        kernel = 4 if stage_idx == 0 else 2
        base = f"stages.{stage_idx}.downsample_layers"
        key = f"stage.{stage_idx}"
        norm_channels = channels if stage_idx == 0 else in_channels
        store(
            f"{key}.downsample.weight",
            f"{base}.{conv_index}.weight",
            (channels, in_channels, kernel, kernel),
        )
        store(
            f"{key}.downsample.bias", f"{base}.{conv_index}.bias", (channels,)
        )
        store(
            f"{key}.downsample_norm.weight",
            f"{base}.{norm_index}.weight",
            (norm_channels,),
        )
        store(
            f"{key}.downsample_norm.bias",
            f"{base}.{norm_index}.bias",
            (norm_channels,),
        )

        for block_idx in range(depth):
            hf_base = f"stages.{stage_idx}.layers.{block_idx}"
            key_base = f"{key}.block.{block_idx}"
            for suffix, expected, logical_suffix, transpose in (
                ("depthwise_conv.weight", (channels, 1, 7, 7), "depthwise.weight", False),
                ("depthwise_conv.bias", (channels,), "depthwise.bias", False),
                ("layer_norm.weight", (channels,), "norm.weight", False),
                ("layer_norm.bias", (channels,), "norm.bias", False),
                ("pointwise_conv1.weight", (4 * channels, channels), "pointwise1.weight", True),
                ("pointwise_conv1.bias", (4 * channels,), "pointwise1.bias", False),
                ("pointwise_conv2.weight", (channels, 4 * channels), "pointwise2.weight", True),
                ("pointwise_conv2.bias", (channels,), "pointwise2.bias", False),
                ("gamma", (channels,), "gamma", False),
            ):
                store(
                    f"{key_base}.{logical_suffix}",
                    f"{hf_base}.{suffix}",
                    expected,
                    transpose=transpose,
                )

    final_channels = hidden_sizes[-1]
    for suffix in ("weight", "bias"):
        store(f"final_norm.{suffix}", f"layer_norm.{suffix}", (final_channels,))
    return weights


def convnext_bundle_metadata(config_or_raw: object) -> dict[str, Any]:
    """Return runtime fields shared by ConvNeXt and the image-feature pipeline."""
    cfg = resolve_convnext_config(config_or_raw)
    return {
        "model_type": cfg["model_type"],
        "hidden_size": cfg["output_dim"],
        "num_hidden_layers": cfg["num_layers"],
        "input_image_h": cfg["image_h"],
        "input_image_w": cfg["image_w"],
        "image_mean": list(_IMAGENET_MEAN),
        "image_std": list(_IMAGENET_STD),
        "interpolation": "bilinear",
        "feature_grid_h": cfg["grid_h"],
        "feature_grid_w": cfg["grid_w"],
        "num_feature_tokens": cfg["num_tokens"],
        "vision_output_dim": cfg["output_dim"],
    }


def _add_conv2d(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    output_channels: int,
    kernel: int,
    stride: int,
    padding: int = 0,
    groups: int = 1,
):
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=output_channels,
        kernel_shape=(kernel, kernel),
        kernel=trt.Weights(np.ascontiguousarray(weight)),
        bias=trt.Weights(np.ascontiguousarray(bias)),
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create a DINOv3 ConvNeXt convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def _nchw_layer_norm(network, tensor, channels: int, weight, bias, eps, dtype, graph_ops):
    tensor = graph_ops.shuffle(
        network, tensor, first_transpose=trt.Permutation([0, 2, 3, 1])
    )
    normalized = graph_ops.layer_norm(
        network, tensor, channels, weight, bias, eps, dtype
    )
    return graph_ops.shuffle(
        network, normalized, first_transpose=trt.Permutation([0, 3, 1, 2])
    )


def _add_block(network, tensor, weights, prefix, channels, cfg, dtype, graph_ops):
    residual = tensor
    hidden = _add_conv2d(
        network,
        tensor,
        weights[f"{prefix}.depthwise.weight"],
        weights[f"{prefix}.depthwise.bias"],
        output_channels=channels,
        kernel=7,
        stride=1,
        padding=3,
        groups=channels,
    )
    hidden = graph_ops.shuffle(
        network, hidden, first_transpose=trt.Permutation([0, 2, 3, 1])
    )
    hidden = graph_ops.layer_norm(
        network,
        hidden,
        channels,
        weights[f"{prefix}.norm.weight"],
        weights[f"{prefix}.norm.bias"],
        cfg["layer_norm_eps"],
        dtype,
    )
    hidden = graph_ops.linear_with_bias(network, hidden, weights, f"{prefix}.pointwise1", dtype)
    hidden = graph_ops.activation(network, hidden, cfg["hidden_act"], dtype)
    hidden = graph_ops.linear_with_bias(network, hidden, weights, f"{prefix}.pointwise2", dtype)
    hidden = graph_ops.multiply_last_dim(network, hidden, weights[f"{prefix}.gamma"], dtype)
    hidden = graph_ops.shuffle(
        network, hidden, first_transpose=trt.Permutation([0, 3, 1, 2])
    )
    # DropPath is an identity during eval, matching Hugging Face model.eval().
    return network.add_elementwise(
        residual, hidden, trt.ElementWiseOperation.SUM
    ).get_output(0)


def build_convnext_engine(
    config_or_raw: object,
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a batch-1 DINOv3 ConvNeXt feature extractor with TensorRT APIs."""
    from . import graph_ops

    cfg = resolve_convnext_config(config_or_raw)
    work_np_dtype = target_dtype(precision)
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32
    builder, network, builder_config = graph_ops.new_network(verbose)

    pixel_values = network.add_input(
        "pixel_values",
        trt.float32,
        (1, cfg["num_channels"], cfg["image_h"], cfg["image_w"]),
    )
    hidden = graph_ops.cast(network, pixel_values, work_trt_dtype)

    for stage_idx, (channels, depth) in enumerate(
        zip(cfg["hidden_sizes"], cfg["depths"], strict=True)
    ):
        prefix = f"stage.{stage_idx}"
        in_channels = cfg["num_channels"] if stage_idx == 0 else cfg["hidden_sizes"][stage_idx - 1]
        if stage_idx == 0:
            hidden = _add_conv2d(
                network,
                hidden,
                weights[f"{prefix}.downsample.weight"],
                weights[f"{prefix}.downsample.bias"],
                output_channels=channels,
                kernel=4,
                stride=4,
            )
            hidden = _nchw_layer_norm(
                network,
                hidden,
                channels,
                weights[f"{prefix}.downsample_norm.weight"],
                weights[f"{prefix}.downsample_norm.bias"],
                cfg["layer_norm_eps"],
                work_np_dtype,
                graph_ops,
            )
        else:
            hidden = _nchw_layer_norm(
                network,
                hidden,
                in_channels,
                weights[f"{prefix}.downsample_norm.weight"],
                weights[f"{prefix}.downsample_norm.bias"],
                cfg["layer_norm_eps"],
                work_np_dtype,
                graph_ops,
            )
            hidden = _add_conv2d(
                network,
                hidden,
                weights[f"{prefix}.downsample.weight"],
                weights[f"{prefix}.downsample.bias"],
                output_channels=channels,
                kernel=2,
                stride=2,
            )

        for block_idx in range(depth):
            hidden = _add_block(
                network,
                hidden,
                weights,
                f"{prefix}.block.{block_idx}",
                channels,
                cfg,
                work_np_dtype,
                graph_ops,
            )

    final_channels = cfg["output_dim"]
    num_patches = cfg["grid_h"] * cfg["grid_w"]

    pooled = network.add_reduce(
        hidden,
        trt.ReduceOperation.AVG,
        (1 << 2) | (1 << 3),
        False,
    ).get_output(0)
    pooled_row = graph_ops.shuffle(network, pooled, reshape_dims=(1, 1, final_channels))

    patch_rows = graph_ops.shuffle(
        network,
        hidden,
        first_transpose=trt.Permutation([0, 2, 3, 1]),
        reshape_dims=(1, num_patches, final_channels),
    )
    tokens = network.add_concatenation([pooled_row, patch_rows])
    tokens.axis = 1
    last_hidden_state = graph_ops.layer_norm(
        network,
        tokens.get_output(0),
        final_channels,
        weights["final_norm.weight"],
        weights["final_norm.bias"],
        cfg["layer_norm_eps"],
        work_np_dtype,
    )
    pooler_output = graph_ops.slice_tensor(
        network, last_hidden_state, (0, 0, 0), (1, 1, final_channels)
    )
    pooler_output = graph_ops.shuffle(
        network, pooler_output, reshape_dims=(1, final_channels)
    )

    last_hidden_state = graph_ops.cast(network, last_hidden_state, trt.float32)
    pooler_output = graph_ops.cast(network, pooler_output, trt.float32)
    last_hidden_state.name = "last_hidden_state"
    pooler_output.name = "pooler_output"
    network.mark_output(last_hidden_state)
    network.mark_output(pooler_output)

    if verbose:
        print(
            "[trtmc build] DINOv3 ConvNeXt: "
            f"image={cfg['image_h']}x{cfg['image_w']}, "
            f"stages={len(cfg['depths'])}, depths={list(cfg['depths'])}, "
            f"channels={list(cfg['hidden_sizes'])}, tokens={cfg['num_tokens']}, "
            f"precision={precision}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT DINOv3 ConvNeXt engine build failed")
    return bytes(plan)


__all__ = [
    "build_convnext_engine",
    "convnext_bundle_metadata",
    "load_convnext_weights",
    "resolve_convnext_config",
]
