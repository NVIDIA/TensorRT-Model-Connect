# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm ConvNeXt classifiers.

ConvNeXt is a convolutional network shaped like a transformer: a large-kernel
depthwise convolution stands in for attention, followed by a LayerNorm and a
two-layer MLP with one GELU.

timm applies that MLP by transposing to channels-last, running two Linear
layers, and transposing back. A Linear over the channel axis is exactly a 1x1
convolution on NCHW, so it is emitted that way here and the network never
leaves NCHW.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorrt as trt

from . import graph
from .checkpoint import Checkpoint


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


# ConvNeXt uses a looser LayerNorm epsilon than the transformer families.
_LAYER_NORM_EPSILON = 1e-6
_BLOCK = re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"ConvNeXt model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("ConvNeXt config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("convnext"):
        raise ValueError(f"unsupported timm ConvNeXt model identity: {identity!r}")
    return value


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("pretrained_cfg")
    source = nested if isinstance(nested, dict) else raw
    input_size = source.get("input_size", [3, 224, 224])
    if isinstance(input_size, int):
        height = width = input_size
    elif (
        isinstance(input_size, list)
        and len(input_size) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) for value in input_size)
        and input_size[0] == 3
    ):
        height, width = input_size[-2:]
    else:
        raise ValueError("ConvNeXt pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("ConvNeXt image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("ConvNeXt image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.95)),
        "interpolation": str(source.get("interpolation", "bicubic")),
    }
    if (
        result["image_height"] <= 0
        or result["image_width"] <= 0
        or result["num_classes"] <= 0
        or not 0.0 < result["crop_pct"] <= 1.0
        or any(value == 0.0 for value in result["std"])
        or result["interpolation"] not in {"bilinear", "bicubic"}
    ):
        raise ValueError("ConvNeXt preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[int]:
    depths: dict[int, set[int]] = {}
    for name in checkpoint.names:
        match = _BLOCK.match(name)
        if match:
            depths.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    if not depths:
        raise ValueError("ConvNeXt checkpoint has no stages.<stage>.blocks.<index> tensors")
    stages = sorted(depths)
    if stages != list(range(len(stages))):
        raise ValueError("ConvNeXt stage indices are not contiguous")
    result: list[int] = []
    for stage in stages:
        indices = sorted(depths[stage])
        if indices != list(range(len(indices))):
            raise ValueError(f"ConvNeXt stage {stage} block indices are not contiguous")
        result.append(len(indices))
    return result


def _weights(
    checkpoint: Checkpoint,
    depths: list[int],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    def take(name: str) -> np.ndarray:
        return checkpoint.tensor(name).astype(dtype)

    result: dict[str, np.ndarray] = {
        "stem.weight": take("stem.0.weight"),
        "stem.bias": take("stem.0.bias"),
        "stem.norm.weight": take("stem.1.weight"),
        "stem.norm.bias": take("stem.1.bias"),
        "head.norm.weight": take("head.norm.weight"),
        "head.norm.bias": take("head.norm.bias"),
        "head.fc.weight": take("head.fc.weight"),
        "head.fc.bias": take("head.fc.bias"),
    }
    for stage, depth in enumerate(depths):
        downsample = f"stages.{stage}.downsample"
        if f"{downsample}.1.weight" in checkpoint.names:
            # The norm comes *before* the strided convolution here, unlike
            # every other downsample in this repository.
            result[f"{downsample}.norm.weight"] = take(f"{downsample}.0.weight")
            result[f"{downsample}.norm.bias"] = take(f"{downsample}.0.bias")
            result[f"{downsample}.weight"] = take(f"{downsample}.1.weight")
            result[f"{downsample}.bias"] = take(f"{downsample}.1.bias")
        for index in range(depth):
            prefix = f"stages.{stage}.blocks.{index}"
            for leaf in ("conv_dw", "mlp.fc1", "mlp.fc2"):
                result[f"{prefix}.{leaf}.weight"] = take(f"{prefix}.{leaf}.weight")
                result[f"{prefix}.{leaf}.bias"] = take(f"{prefix}.{leaf}.bias")
            result[f"{prefix}.norm.weight"] = take(f"{prefix}.norm.weight")
            result[f"{prefix}.norm.bias"] = take(f"{prefix}.norm.bias")
            if f"{prefix}.gamma" in checkpoint.names:
                result[f"{prefix}.gamma"] = take(f"{prefix}.gamma")
    if result["head.fc.weight"].ndim != 2 or result["head.fc.bias"].shape != (
        result["head.fc.weight"].shape[0],
    ):
        raise ValueError("ConvNeXt classifier weights have incompatible shapes")
    return result


def _linear_as_convolution(network, tensor, weight: np.ndarray, bias: np.ndarray, dtype):
    """A Linear over the channel axis, emitted as a 1x1 convolution."""
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    return graph.convolution(
        network, tensor, weight.reshape(out_features, in_features, 1, 1), bias, dtype=dtype
    )


def _block(network, tensor, weights, prefix: str, dtype):
    shortcut = tensor
    depthwise = weights[f"{prefix}.conv_dw.weight"]
    tensor = graph.convolution(
        network, tensor, depthwise, weights[f"{prefix}.conv_dw.bias"],
        padding=int(depthwise.shape[2]) // 2, groups=int(depthwise.shape[0]), dtype=dtype,
    )
    tensor = graph.layer_norm_channels(
        network, tensor, weights[f"{prefix}.norm.weight"], weights[f"{prefix}.norm.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=dtype,
    )
    tensor = _linear_as_convolution(
        network, tensor, weights[f"{prefix}.mlp.fc1.weight"],
        weights[f"{prefix}.mlp.fc1.bias"], dtype,
    )
    tensor = graph.gelu(network, tensor, dtype=dtype)
    tensor = _linear_as_convolution(
        network, tensor, weights[f"{prefix}.mlp.fc2.weight"],
        weights[f"{prefix}.mlp.fc2.bias"], dtype,
    )
    if f"{prefix}.gamma" in weights:
        # Layer scale: a learned per-channel gain on the residual branch.
        tensor = graph.channel_scale(network, tensor, weights[f"{prefix}.gamma"], dtype=dtype)
    return graph.add(network, tensor, shortcut)


def _build_engine(
    raw: dict[str, Any],
    checkpoint: Checkpoint,
    precision: str,
    verbose: bool,
) -> tuple[bytes, dict[str, Any]]:
    if precision == "fp16":
        numpy_dtype, tensor_dtype = np.float16, trt.float16
    elif precision == "fp32":
        numpy_dtype, tensor_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"unsupported timm ConvNeXt precision: {precision}")
    config = _preprocess_config(raw)
    depths = _layout(checkpoint)
    weights = _weights(checkpoint, depths, numpy_dtype)
    if weights["head.fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("ConvNeXt classifier dimensions do not match config.json")

    patch = int(weights["stem.weight"].shape[2])
    total_stride = patch * 2 ** (len(depths) - 1)
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"ConvNeXt input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_convnext: "
            f"image={height}x{width}, depths={depths}, patch={patch}, "
            f"classes={config['num_classes']}, precision={precision}",
            file=sys.stderr,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 1
    builder_config.avg_timing_iterations = 8
    builder_config.max_aux_streams = 0
    builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, height, width))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the ConvNeXt input cast")
        hidden = cast.get_output(0)

    hidden = graph.convolution(
        network, hidden, weights["stem.weight"], weights["stem.bias"],
        stride=patch, dtype=numpy_dtype,
    )
    hidden = graph.layer_norm_channels(
        network, hidden, weights["stem.norm.weight"], weights["stem.norm.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=numpy_dtype,
    )

    for stage, depth in enumerate(depths):
        downsample = f"stages.{stage}.downsample"
        if f"{downsample}.weight" in weights:
            hidden = graph.layer_norm_channels(
                network, hidden, weights[f"{downsample}.norm.weight"],
                weights[f"{downsample}.norm.bias"],
                epsilon=_LAYER_NORM_EPSILON, dtype=numpy_dtype,
            )
            reducer = weights[f"{downsample}.weight"]
            hidden = graph.convolution(
                network, hidden, reducer, weights[f"{downsample}.bias"],
                stride=int(reducer.shape[2]), dtype=numpy_dtype,
            )
        for index in range(depth):
            hidden = _block(
                network, hidden, weights, f"stages.{stage}.blocks.{index}", numpy_dtype
            )

    # The head pools first and normalises after, so the norm sees one value per
    # channel rather than the whole feature map.
    hidden = graph.mean_spatial(network, hidden)
    hidden = graph.layer_norm_channels(
        network, hidden, weights["head.norm.weight"], weights["head.norm.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=numpy_dtype,
    )
    logits = graph.classifier(
        network, hidden, weights["head.fc.weight"], weights["head.fc.bias"], dtype=numpy_dtype
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the ConvNeXt output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm ConvNeXt engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm ConvNeXt image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_convnext does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_convnext does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_convnext does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_convnext does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_convnext does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_convnext does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_convnext does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_convnext supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_convnext does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_convnext does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_convnext", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": runtime["image_height"],
            "input_image_w": runtime["image_width"],
            "crop_pct": runtime["crop_pct"],
            "interpolation": runtime["interpolation"],
            "image_mean": runtime["mean"],
            "image_std": runtime["std"],
        },
    )
