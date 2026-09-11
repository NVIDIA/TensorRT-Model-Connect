# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm RegNet classifiers.

The whole layout comes from the checkpoint. Stages and blocks are read from the
`s<stage>.b<block>` keys, squeeze-excitation and the projection shortcut from
whether those leaves exist, and the group count of the 3x3 from its own weight
shape. Nothing about the width schedule is tabulated here.
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
_BLOCK = re.compile(r"^s(\d+)\.b(\d+)\.(.+)$")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"RegNet model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RegNet config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("regnet"):
        raise ValueError(f"unsupported timm RegNet model identity: {identity!r}")
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
        raise ValueError("RegNet pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("RegNet image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("RegNet image std must contain three values")
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
        raise ValueError("RegNet preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    leaves: dict[tuple[int, int], set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.fullmatch(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".", 1)[0])
    if not leaves:
        raise ValueError("RegNet checkpoint has no s<stage>.b<block> tensors")
    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(1, len(stages) + 1)):
        raise ValueError("RegNet stage indices are not contiguous from 1")
    blocks: list[dict[str, object]] = []
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"RegNet stage {stage} block indices are not contiguous from 1")
        for index in indices:
            present = leaves[(stage, index)]
            if not {"conv1", "conv2", "conv3"}.issubset(present):
                raise ValueError(f"RegNet s{stage}.b{index} is missing a bottleneck convolution")
            blocks.append(
                {
                    "prefix": f"s{stage}.b{index}",
                    # RegNet halves the resolution at the head of every stage.
                    "stride": 2 if index == 1 else 1,
                    "has_se": "se" in present,
                    "has_downsample": "downsample" in present,
                }
            )
    return blocks


def _fold(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Fold a `conv`/`bn` pair into one biased convolution.

    The statistics stay float32 through the division; only the result is cast,
    so a small running variance does not lose precision in fp16.
    """
    weight = checkpoint.tensor(f"{prefix}.conv.weight")
    gamma = checkpoint.tensor(f"{prefix}.bn.weight")
    beta = checkpoint.tensor(f"{prefix}.bn.bias")
    mean = checkpoint.tensor(f"{prefix}.bn.running_mean")
    variance = checkpoint.tensor(f"{prefix}.bn.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"RegNet norm {prefix}.bn has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"RegNet norm {prefix}.bn does not match its convolution")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"RegNet norm {prefix}.bn has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    result["stem.weight"], result["stem.bias"] = _fold(checkpoint, "stem", dtype)
    for block in blocks:
        prefix = str(block["prefix"])
        for leaf in ("conv1", "conv2", "conv3"):
            weight, bias = _fold(checkpoint, f"{prefix}.{leaf}", dtype)
            result[f"{prefix}.{leaf}.weight"] = weight
            result[f"{prefix}.{leaf}.bias"] = bias
        if bool(block["has_downsample"]):
            weight, bias = _fold(checkpoint, f"{prefix}.downsample", dtype)
            result[f"{prefix}.downsample.weight"] = weight
            result[f"{prefix}.downsample.bias"] = bias
        if bool(block["has_se"]):
            for leaf in ("fc1", "fc2"):
                result[f"{prefix}.se.{leaf}.weight"] = checkpoint.tensor(
                    f"{prefix}.se.{leaf}.weight"
                ).astype(dtype)
                result[f"{prefix}.se.{leaf}.bias"] = checkpoint.tensor(
                    f"{prefix}.se.{leaf}.bias"
                ).astype(dtype)
    result["head.fc.weight"] = checkpoint.tensor("head.fc.weight").astype(dtype)
    result["head.fc.bias"] = checkpoint.tensor("head.fc.bias").astype(dtype)
    if result["head.fc.weight"].ndim != 2 or result["head.fc.bias"].shape != (
        result["head.fc.weight"].shape[0],
    ):
        raise ValueError("RegNet classifier weights have incompatible shapes")
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
        raise ValueError(f"unsupported timm RegNet precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    if weights["head.fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("RegNet classifier dimensions do not match config.json")

    total_stride = 2
    for block in blocks:
        total_stride *= int(block["stride"])
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"RegNet input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_regnet: "
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
        raise RuntimeError("TensorRT rejected the RegNet input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the RegNet input cast")
        hidden = cast.get_output(0)

    hidden = graph.convolution(
        network,
        hidden,
        weights["stem.weight"],
        weights["stem.bias"],
        stride=2,
        padding=1,
        dtype=numpy_dtype,
    )
    hidden = graph.relu(network, hidden)

    for block in blocks:
        prefix = str(block["prefix"])
        stride = int(block["stride"])
        identity = hidden
        tensor = graph.convolution(
            network,
            hidden,
            weights[f"{prefix}.conv1.weight"],
            weights[f"{prefix}.conv1.bias"],
            dtype=numpy_dtype,
        )
        tensor = graph.relu(network, tensor)
        # The 3x3 is grouped; the group count follows from its own weight shape.
        grouped = weights[f"{prefix}.conv2.weight"]
        groups = max(1, int(grouped.shape[0]) // int(grouped.shape[1]))
        tensor = graph.convolution(
            network,
            tensor,
            grouped,
            weights[f"{prefix}.conv2.bias"],
            stride=stride,
            padding=1,
            groups=groups,
            dtype=numpy_dtype,
        )
        tensor = graph.relu(network, tensor)
        if bool(block["has_se"]):
            tensor = graph.squeeze_excite(
                network,
                tensor,
                weights[f"{prefix}.se.fc1.weight"],
                weights[f"{prefix}.se.fc1.bias"],
                weights[f"{prefix}.se.fc2.weight"],
                weights[f"{prefix}.se.fc2.bias"],
                dtype=numpy_dtype,
            )
        tensor = graph.convolution(
            network,
            tensor,
            weights[f"{prefix}.conv3.weight"],
            weights[f"{prefix}.conv3.bias"],
            dtype=numpy_dtype,
        )
        if bool(block["has_downsample"]):
            identity = graph.convolution(
                network,
                identity,
                weights[f"{prefix}.downsample.weight"],
                weights[f"{prefix}.downsample.bias"],
                stride=stride,
                dtype=numpy_dtype,
            )
        hidden = graph.relu(network, graph.add(network, tensor, identity))

    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    logits = graph.classifier(
        network,
        hidden,
        weights["head.fc.weight"],
        weights["head.fc.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the RegNet output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm RegNet engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm RegNet image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_regnet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_regnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_regnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_regnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_regnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_regnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_regnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_regnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_regnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_regnet does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_regnet", task=request.task, backend=request.backend)
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
