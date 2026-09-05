# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm RepVGG classifiers."""

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


_BATCH_NORM_EPSILON = 1e-5
_BLOCK = re.compile(r"^stages\.(\d+)\.(\d+)\.(.+)$")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"RepVGG model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RepVGG config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("repvgg"):
        raise ValueError(f"unsupported timm RepVGG model identity: {identity!r}")
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
        raise ValueError("RepVGG pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("RepVGG image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("RepVGG image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "num_features": int(raw.get("num_features", 1408)),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.875)),
        "interpolation": str(source.get("interpolation", "bilinear")),
    }
    if (
        result["image_height"] <= 0
        or result["image_width"] <= 0
        or result["num_classes"] <= 0
        or result["num_features"] <= 0
        or not 0.0 < result["crop_pct"] <= 1.0
        or any(value == 0.0 for value in result["std"])
        or result["interpolation"] not in {"bilinear", "bicubic"}
    ):
        raise ValueError("RepVGG preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    leaves: dict[tuple[int, int], set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.fullmatch(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".", 1)[0])
    if not leaves:
        raise ValueError("RepVGG checkpoint has no stages.<stage>.<block> tensors")
    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(len(stages))):
        raise ValueError("RepVGG stage indices are not contiguous")
    blocks: list[dict[str, object]] = []
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"RepVGG stage {stage} block indices are not contiguous")
        for index in indices:
            branches = leaves[(stage, index)]
            if not {"conv_kxk", "conv_1x1"}.issubset(branches):
                raise ValueError(f"RepVGG stages.{stage}.{index} is missing a convolution branch")
            has_identity = "identity" in branches
            blocks.append(
                {
                    "prefix": f"stages.{stage}.{index}",
                    "has_identity": has_identity,
                    "stride": 1 if has_identity else 2,
                }
            )
    return blocks


def _batch_norm(checkpoint: Checkpoint, prefix: str, channels: int) -> tuple[np.ndarray, ...]:
    values = tuple(
        checkpoint.tensor(f"{prefix}.{suffix}")
        for suffix in ("weight", "bias", "running_mean", "running_var")
    )
    if any(value.shape != (channels,) for value in values):
        raise ValueError(f"RepVGG batch norm shape mismatch: {prefix}")
    return values


def _fold_branch(
    weight: np.ndarray,
    parameters: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    gamma, beta, mean, variance = parameters
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (
        np.ascontiguousarray(weight * scale.reshape(-1, 1, 1, 1), dtype=np.float32),
        np.ascontiguousarray(beta - mean * scale, dtype=np.float32),
    )


def _fused_block(
    checkpoint: Checkpoint,
    prefix: str,
    has_identity: bool,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    spatial = checkpoint.tensor(f"{prefix}.conv_kxk.conv.weight")
    if spatial.ndim != 4 or spatial.shape[-2:] != (3, 3):
        raise ValueError(f"RepVGG {prefix} must have one 3x3 branch")
    output_channels, input_channels = spatial.shape[:2]
    fused_weight, fused_bias = _fold_branch(
        spatial,
        _batch_norm(checkpoint, f"{prefix}.conv_kxk.bn", output_channels),
    )

    pointwise = checkpoint.tensor(f"{prefix}.conv_1x1.conv.weight")
    if pointwise.shape != (output_channels, input_channels, 1, 1):
        raise ValueError(f"RepVGG {prefix} convolution branches have incompatible shapes")
    point_weight, point_bias = _fold_branch(
        pointwise,
        _batch_norm(checkpoint, f"{prefix}.conv_1x1.bn", output_channels),
    )
    fused_weight[:, :, 1:2, 1:2] += point_weight
    fused_bias += point_bias

    if has_identity:
        if input_channels != output_channels:
            raise ValueError(f"RepVGG {prefix} grouped identity blocks are unsupported")
        gamma, beta, mean, variance = _batch_norm(checkpoint, f"{prefix}.identity", output_channels)
        scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
        indices = np.arange(output_channels)
        fused_weight[indices, indices, 1, 1] += scale
        fused_bias += beta - mean * scale
    return (
        np.ascontiguousarray(fused_weight, dtype=dtype),
        np.ascontiguousarray(fused_bias, dtype=dtype),
    )


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    result["stem.weight"], result["stem.bias"] = _fused_block(checkpoint, "stem", False, dtype)
    for block in blocks:
        prefix = str(block["prefix"])
        result[f"{prefix}.weight"], result[f"{prefix}.bias"] = _fused_block(
            checkpoint,
            prefix,
            bool(block["has_identity"]),
            dtype,
        )
    result["head.weight"] = np.ascontiguousarray(checkpoint.tensor("head.fc.weight"), dtype=dtype)
    result["head.bias"] = np.ascontiguousarray(checkpoint.tensor("head.fc.bias"), dtype=dtype)
    if result["head.weight"].ndim != 2 or result["head.bias"].shape != (
        result["head.weight"].shape[0],
    ):
        raise ValueError("RepVGG classifier weights have incompatible shapes")
    return result


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
        raise ValueError(f"unsupported timm RepVGG precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    head = weights["head.weight"]
    if head.shape != (config["num_classes"], config["num_features"]):
        raise ValueError("RepVGG classifier dimensions do not match config.json")

    total_stride = 2
    for block in blocks:
        total_stride *= int(block["stride"])
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"RepVGG input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_repvgg: "
            f"image={height}x{width}, blocks={len(blocks)}, "
            f"classes={config['num_classes']}, precision={precision}",
            file=sys.stderr,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.avg_timing_iterations = 8
    builder_config.max_aux_streams = 0
    builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, height, width))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the RepVGG input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the RepVGG input cast")
        hidden = cast.get_output(0)
    hidden = graph.convolution(
        network,
        hidden,
        weights["stem.weight"],
        weights["stem.bias"],
        stride=2,
        dtype=numpy_dtype,
    )
    hidden = graph.relu(network, hidden)
    for block in blocks:
        prefix = str(block["prefix"])
        hidden = graph.convolution(
            network,
            hidden,
            weights[f"{prefix}.weight"],
            weights[f"{prefix}.bias"],
            stride=int(block["stride"]),
            dtype=numpy_dtype,
        )
        hidden = graph.relu(network, hidden)
    hidden = graph.global_average_pool(
        network,
        hidden,
        height // total_stride,
        width // total_stride,
    )
    logits = graph.classifier(
        network,
        hidden,
        weights["head.weight"],
        weights["head.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the RepVGG output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm RepVGG engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one fused timm RepVGG image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_repvgg does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("timm_repvgg does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_repvgg does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_repvgg does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_repvgg does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_repvgg does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_repvgg does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_repvgg supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_repvgg does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_repvgg does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_repvgg", task=request.task, backend=request.backend)
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
