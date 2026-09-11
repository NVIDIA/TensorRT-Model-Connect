# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm GhostNet classifiers.

Everything about the layout is read from the checkpoint: which blocks are Ghost
bottlenecks and which are plain convolutions, whether a bottleneck strides,
whether it gates, and whether it projects its shortcut.

A Ghost module produces half its output channels with a pointwise convolution
and the other half by applying a cheap depthwise convolution to that result,
then concatenates the two. The order of that concatenation is fixed by how the
weights were trained; swapping the halves still builds.
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


_BATCH_NORM_EPSILON = 1e-5
_BLOCK = re.compile(r"^blocks\.(\d+)\.(\d+)\.(.+)$")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"GhostNet model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GhostNet config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("ghostnet"):
        raise ValueError(f"unsupported timm GhostNet model identity: {identity!r}")
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
        raise ValueError("GhostNet pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("GhostNet image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("GhostNet image std must contain three values")
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
        raise ValueError("GhostNet preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    leaves: dict[tuple[int, int], set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.fullmatch(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".", 1)[0])
    if not leaves:
        raise ValueError("GhostNet checkpoint has no blocks.<stage>.<index> tensors")
    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(len(stages))):
        raise ValueError("GhostNet block stage indices are not contiguous")
    blocks: list[dict[str, object]] = []
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"GhostNet stage {stage} block indices are not contiguous")
        for index in indices:
            present = leaves[(stage, index)]
            if "conv" in present:
                kind = "convolution"
            elif "ghost1" in present:
                kind = "bottleneck"
            else:
                raise ValueError(
                    f"GhostNet blocks.{stage}.{index} is neither a Ghost bottleneck "
                    "nor a convolution block"
                )
            blocks.append(
                {
                    "prefix": f"blocks.{stage}.{index}",
                    "kind": kind,
                    # A bottleneck reduces exactly when it carries a depthwise
                    # convolution between its two Ghost modules.
                    "stride": 2 if "conv_dw" in present else 1,
                    "has_se": "se" in present,
                    "has_shortcut": "shortcut" in present,
                }
            )
    return blocks


def _fold(
    checkpoint: Checkpoint, conv: str, norm: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a batch norm into the convolution ahead of it."""
    weight = checkpoint.tensor(conv)
    gamma = checkpoint.tensor(f"{norm}.weight")
    beta = checkpoint.tensor(f"{norm}.bias")
    mean = checkpoint.tensor(f"{norm}.running_mean")
    variance = checkpoint.tensor(f"{norm}.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"GhostNet norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"GhostNet norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"GhostNet norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    result["stem.weight"], result["stem.bias"] = _fold(
        checkpoint, "conv_stem.weight", "bn1", dtype
    )
    for block in blocks:
        prefix = str(block["prefix"])
        if block["kind"] == "convolution":
            weight, bias = _fold(checkpoint, f"{prefix}.conv.weight", f"{prefix}.bn1", dtype)
            result[f"{prefix}.conv.weight"], result[f"{prefix}.conv.bias"] = weight, bias
            continue
        for ghost in ("ghost1", "ghost2"):
            for part, leaf in (("primary", "primary_conv"), ("cheap", "cheap_operation")):
                weight, bias = _fold(
                    checkpoint,
                    f"{prefix}.{ghost}.{leaf}.0.weight",
                    f"{prefix}.{ghost}.{leaf}.1",
                    dtype,
                )
                result[f"{prefix}.{ghost}.{part}.weight"] = weight
                result[f"{prefix}.{ghost}.{part}.bias"] = bias
        if int(block["stride"]) != 1:
            weight, bias = _fold(checkpoint, f"{prefix}.conv_dw.weight", f"{prefix}.bn_dw", dtype)
            result[f"{prefix}.conv_dw.weight"], result[f"{prefix}.conv_dw.bias"] = weight, bias
        if bool(block["has_se"]):
            for leaf in ("conv_reduce", "conv_expand"):
                result[f"{prefix}.se.{leaf}.weight"] = checkpoint.tensor(
                    f"{prefix}.se.{leaf}.weight"
                ).astype(dtype)
                result[f"{prefix}.se.{leaf}.bias"] = checkpoint.tensor(
                    f"{prefix}.se.{leaf}.bias"
                ).astype(dtype)
        if bool(block["has_shortcut"]):
            for position, (conv, norm) in enumerate(((0, 1), (2, 3))):
                weight, bias = _fold(
                    checkpoint,
                    f"{prefix}.shortcut.{conv}.weight",
                    f"{prefix}.shortcut.{norm}",
                    dtype,
                )
                result[f"{prefix}.shortcut.{position}.weight"] = weight
                result[f"{prefix}.shortcut.{position}.bias"] = bias
    result["head.weight"] = checkpoint.tensor("conv_head.weight").astype(dtype)
    result["head.bias"] = checkpoint.tensor("conv_head.bias").astype(dtype)
    result["classifier.weight"] = checkpoint.tensor("classifier.weight").astype(dtype)
    result["classifier.bias"] = checkpoint.tensor("classifier.bias").astype(dtype)
    if result["classifier.weight"].ndim != 2 or result["classifier.bias"].shape != (
        result["classifier.weight"].shape[0],
    ):
        raise ValueError("GhostNet classifier weights have incompatible shapes")
    return result


def _ghost_module(network, tensor, weights, prefix: str, dtype, *, activate: bool):
    """Pointwise branch, then a cheap depthwise branch over it, concatenated.

    The primary half comes first in the concatenation; that order is what the
    weights were trained against.
    """
    primary = graph.convolution(
        network, tensor, weights[f"{prefix}.primary.weight"],
        weights[f"{prefix}.primary.bias"], dtype=dtype,
    )
    if activate:
        primary = graph.relu(network, primary)
    cheap_weight = weights[f"{prefix}.cheap.weight"]
    cheap = graph.convolution(
        network, primary, cheap_weight, weights[f"{prefix}.cheap.bias"],
        padding=int(cheap_weight.shape[2]) // 2, groups=int(cheap_weight.shape[0]), dtype=dtype,
    )
    if activate:
        cheap = graph.relu(network, cheap)
    return graph.concatenate(network, [primary, cheap])


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
        raise ValueError(f"unsupported timm GhostNet precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    if weights["classifier.weight"].shape[0] != config["num_classes"]:
        raise ValueError("GhostNet classifier dimensions do not match config.json")

    total_stride = 2
    for block in blocks:
        total_stride *= int(block["stride"])
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"GhostNet input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_ghostnet: "
            f"image={height}x{width}, blocks={len(blocks)}, "
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
        raise RuntimeError("TensorRT rejected the GhostNet input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the GhostNet input cast")
        hidden = cast.get_output(0)

    hidden = graph.convolution(
        network, hidden, weights["stem.weight"], weights["stem.bias"],
        stride=2, padding=1, dtype=numpy_dtype,
    )
    hidden = graph.relu(network, hidden)

    for block in blocks:
        prefix = str(block["prefix"])
        if block["kind"] == "convolution":
            hidden = graph.convolution(
                network, hidden, weights[f"{prefix}.conv.weight"],
                weights[f"{prefix}.conv.bias"], dtype=numpy_dtype,
            )
            hidden = graph.relu(network, hidden)
            continue

        stride = int(block["stride"])
        identity = hidden
        # The first Ghost module activates; the second does not.
        tensor = _ghost_module(
            network, hidden, weights, f"{prefix}.ghost1", numpy_dtype, activate=True
        )
        if stride != 1:
            depthwise = weights[f"{prefix}.conv_dw.weight"]
            tensor = graph.convolution(
                network, tensor, depthwise, weights[f"{prefix}.conv_dw.bias"],
                stride=stride, padding=int(depthwise.shape[2]) // 2,
                groups=int(depthwise.shape[0]), dtype=numpy_dtype,
            )
        if bool(block["has_se"]):
            tensor = graph.squeeze_excite(
                network, tensor,
                weights[f"{prefix}.se.conv_reduce.weight"],
                weights[f"{prefix}.se.conv_reduce.bias"],
                weights[f"{prefix}.se.conv_expand.weight"],
                weights[f"{prefix}.se.conv_expand.bias"],
                dtype=numpy_dtype,
            )
        tensor = _ghost_module(
            network, tensor, weights, f"{prefix}.ghost2", numpy_dtype, activate=False
        )
        if bool(block["has_shortcut"]):
            depthwise = weights[f"{prefix}.shortcut.0.weight"]
            identity = graph.convolution(
                network, identity, depthwise, weights[f"{prefix}.shortcut.0.bias"],
                stride=stride, padding=int(depthwise.shape[2]) // 2,
                groups=int(depthwise.shape[0]), dtype=numpy_dtype,
            )
            identity = graph.convolution(
                network, identity, weights[f"{prefix}.shortcut.1.weight"],
                weights[f"{prefix}.shortcut.1.bias"], dtype=numpy_dtype,
            )
        hidden = graph.add(network, tensor, identity)

    # GhostNet pools before its head convolution rather than after.
    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    hidden = graph.convolution(
        network, hidden, weights["head.weight"], weights["head.bias"], dtype=numpy_dtype
    )
    hidden = graph.relu(network, hidden)
    logits = graph.classifier(
        network, hidden, weights["classifier.weight"], weights["classifier.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the GhostNet output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm GhostNet engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm GhostNet image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_ghostnet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_ghostnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_ghostnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_ghostnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_ghostnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_ghostnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_ghostnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_ghostnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_ghostnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_ghostnet does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_ghostnet", task=request.task, backend=request.backend)
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
