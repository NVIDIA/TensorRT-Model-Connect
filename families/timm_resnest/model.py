# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm ResNeSt classifiers.

ResNeSt replaces the 3x3 in a ResNet bottleneck with split attention: the
convolution emits `radix` copies of the output channels, a gate is computed from
their sum, and the copies are recombined with a **softmax across the radix
axis** rather than a per-channel sigmoid. That softmax is what distinguishes it
from the SE families, where each channel is gated independently.

Depth, radix and cardinality all come from the checkpoint. Two things do not,
both specific to the `d` variants targeted here, and both are named constants: a
stride never lands on the split-attention convolution but on an average pool
placed after it, and the shortcut is average-pooled before its 1x1 projection.
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
_STAGES = ("layer1", "layer2", "layer3", "layer4")

# Neither is recoverable from weights: a `d` variant moves the stride off the
# convolution onto a pool, and pools the shortcut before projecting it.
_AVERAGE_DOWN = {"kernel": 3, "stride": 2, "padding": 1}
_SHORTCUT_DOWN = {"kernel": 2, "stride": 2, "padding": 0}


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"ResNeSt model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("ResNeSt config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("resnest"):
        raise ValueError(f"unsupported timm ResNeSt model identity: {identity!r}")
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
        raise ValueError("ResNeSt pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("ResNeSt image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("ResNeSt image std must contain three values")
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
        raise ValueError("ResNeSt preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[int]:
    names = checkpoint.names
    depths: list[int] = []
    for stage in _STAGES:
        pattern = re.compile(rf"^{stage}\.(\d+)\.")
        indices = {int(match.group(1)) for match in map(pattern.match, names) if match}
        if not indices:
            raise ValueError(f"ResNeSt checkpoint has no blocks for stage {stage}")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"ResNeSt stage {stage} block indices are not contiguous")
        depths.append(len(indices))
    if not any(re.match(r"^conv1\.\d+\.weight$", name) for name in names):
        raise ValueError("ResNeSt checkpoint has no deep stem")
    return depths


def _fold_norm(
    checkpoint: Checkpoint, conv: str, norm: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a batch norm into the convolution ahead of it."""
    weight = checkpoint.tensor(conv)
    gamma = checkpoint.tensor(f"{norm}.weight")
    beta = checkpoint.tensor(f"{norm}.bias")
    mean = checkpoint.tensor(f"{norm}.running_mean")
    variance = checkpoint.tensor(f"{norm}.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"ResNeSt norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"ResNeSt norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"ResNeSt norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    bias = beta - mean * scale
    own = f"{conv[: -len('.weight')]}.bias"
    if own in checkpoint.names:
        # The split-attention `fc1` carries a bias on top of the norm it feeds.
        bias = bias + checkpoint.tensor(own) * scale
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), bias.astype(dtype)


def _weights(checkpoint: Checkpoint, depths: list[int], dtype: np.dtype):
    result: dict[str, np.ndarray] = {}
    # A norm inside the stem Sequential also has a `.weight`, so the
    # convolutions are picked out by tensor rank rather than by position.
    stem = sorted(
        int(match.group(1))
        for match in (re.match(r"^conv1\.(\d+)\.weight$", name) for name in checkpoint.names)
        if match and checkpoint.tensor(match.group(0)).ndim == 4
    )
    if not stem:
        raise ValueError("ResNeSt checkpoint has no stem convolutions")
    for position, index in enumerate(stem):
        norm = "bn1" if position == len(stem) - 1 else f"conv1.{index + 1}"
        weight, bias = _fold_norm(checkpoint, f"conv1.{index}.weight", norm, dtype)
        result[f"stem.{position}.weight"], result[f"stem.{position}.bias"] = weight, bias
    result["stem_depth"] = np.asarray(len(stem))

    for stage, depth in zip(_STAGES, depths):
        for index in range(depth):
            prefix = f"{stage}.{index}"
            for position, leaf in ((1, "conv1"), (3, "conv3")):
                weight, bias = _fold_norm(
                    checkpoint, f"{prefix}.{leaf}.weight", f"{prefix}.bn{position}", dtype
                )
                result[f"{prefix}.{leaf}.weight"] = weight
                result[f"{prefix}.{leaf}.bias"] = bias
            attention = f"{prefix}.conv2"
            weight, bias = _fold_norm(
                checkpoint, f"{attention}.conv.weight", f"{attention}.bn0", dtype
            )
            result[f"{attention}.conv.weight"], result[f"{attention}.conv.bias"] = weight, bias
            gate_weight, gate_bias = _fold_norm(
                checkpoint, f"{attention}.fc1.weight", f"{attention}.bn1", dtype
            )
            result[f"{attention}.fc1.weight"] = gate_weight
            result[f"{attention}.fc1.bias"] = gate_bias
            result[f"{attention}.fc2.weight"] = checkpoint.tensor(
                f"{attention}.fc2.weight"
            ).astype(dtype)
            result[f"{attention}.fc2.bias"] = checkpoint.tensor(
                f"{attention}.fc2.bias"
            ).astype(dtype)
            if f"{prefix}.downsample.1.weight" in checkpoint.names:
                weight, bias = _fold_norm(
                    checkpoint, f"{prefix}.downsample.1.weight", f"{prefix}.downsample.2", dtype
                )
                result[f"{prefix}.downsample.weight"] = weight
                result[f"{prefix}.downsample.bias"] = bias
    result["fc.weight"] = checkpoint.tensor("fc.weight").astype(dtype)
    result["fc.bias"] = checkpoint.tensor("fc.bias").astype(dtype)
    if result["fc.weight"].ndim != 2 or result["fc.bias"].shape != (result["fc.weight"].shape[0],):
        raise ValueError("ResNeSt classifier weights have incompatible shapes")
    return result


def _split_attention(network, tensor, weights, prefix: str, dtype, *, in_channels: int):
    """The split-attention convolution that replaces the bottleneck 3x3.

    The convolution emits `radix` copies of the output channels. The gate is
    computed from their sum and normalised with a softmax across the radix axis,
    so the copies compete; a per-channel sigmoid, as in the SE families, would
    let them all pass and still produce a plausible answer.
    """
    convolution = weights[f"{prefix}.conv.weight"]
    mid_channels = int(convolution.shape[0])
    out_channels = int(weights[f"{prefix}.fc1.weight"].shape[1])
    radix = mid_channels // out_channels
    if radix * out_channels != mid_channels:
        raise ValueError(f"ResNeSt {prefix}: {mid_channels} is not a multiple of {out_channels}")
    groups = in_channels // int(convolution.shape[1])
    cardinality = groups // radix
    if cardinality < 1 or cardinality * radix != groups:
        raise ValueError(f"ResNeSt {prefix}: {groups} groups is not a multiple of radix {radix}")

    hidden = graph.convolution(
        network, tensor, convolution, weights[f"{prefix}.conv.bias"],
        padding=int(convolution.shape[2]) // 2, groups=groups, dtype=dtype,
    )
    hidden = graph.relu(network, hidden)
    shape = [int(value) for value in hidden.shape]
    split = graph.reshape(network, hidden, (1, radix, out_channels, shape[2], shape[3]))
    pooled = graph.mean_spatial(network, graph.sum_over(network, split, 1), (2, 3))

    gate = graph.convolution(
        network, pooled, weights[f"{prefix}.fc1.weight"], weights[f"{prefix}.fc1.bias"],
        groups=cardinality, dtype=dtype,
    )
    gate = graph.relu(network, gate)
    gate = graph.convolution(
        network, gate, weights[f"{prefix}.fc2.weight"], weights[f"{prefix}.fc2.bias"],
        groups=cardinality, dtype=dtype,
    )
    per_group = mid_channels // (cardinality * radix)
    gate = graph.reshape(network, gate, (1, cardinality, radix, per_group))
    gate = graph.permute(network, gate, (0, 2, 1, 3))
    gate = graph.softmax(network, gate, 1)
    gate = graph.reshape(network, gate, (1, radix, out_channels, 1, 1))
    return graph.sum_over(network, graph.multiply(network, split, gate), 1)


def _block(network, tensor, weights, prefix: str, dtype, *, stride: int):
    shortcut = tensor
    tensor = graph.convolution(
        network, tensor, weights[f"{prefix}.conv1.weight"],
        weights[f"{prefix}.conv1.bias"], dtype=dtype,
    )
    tensor = graph.relu(network, tensor)
    tensor = _split_attention(
        network, tensor, weights, f"{prefix}.conv2", dtype,
        in_channels=int(weights[f"{prefix}.conv1.weight"].shape[0]),
    )
    if stride != 1:
        tensor = graph.average_pool(network, tensor, **_AVERAGE_DOWN)
    tensor = graph.convolution(
        network, tensor, weights[f"{prefix}.conv3.weight"],
        weights[f"{prefix}.conv3.bias"], dtype=dtype,
    )
    if f"{prefix}.downsample.weight" in weights:
        if stride != 1:
            shortcut = graph.average_pool(network, shortcut, **_SHORTCUT_DOWN)
        shortcut = graph.convolution(
            network, shortcut, weights[f"{prefix}.downsample.weight"],
            weights[f"{prefix}.downsample.bias"], dtype=dtype,
        )
    return graph.relu(network, graph.add(network, tensor, shortcut))


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
        raise ValueError(f"unsupported timm ResNeSt precision: {precision}")
    config = _preprocess_config(raw)
    depths = _layout(checkpoint)
    weights = _weights(checkpoint, depths, numpy_dtype)
    if weights["fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("ResNeSt classifier dimensions do not match config.json")

    total_stride = 4 * 2 ** (len(_STAGES) - 1)
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"ResNeSt input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_resnest: "
            f"image={height}x{width}, depths={depths}, "
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
        raise RuntimeError("TensorRT rejected the ResNeSt input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the ResNeSt input cast")
        hidden = cast.get_output(0)

    for position in range(int(weights["stem_depth"])):
        hidden = graph.convolution(
            network, hidden, weights[f"stem.{position}.weight"],
            weights[f"stem.{position}.bias"],
            stride=2 if position == 0 else 1, padding=1, dtype=numpy_dtype,
        )
        hidden = graph.relu(network, hidden)
    hidden = graph.max_pool(network, hidden, kernel=3, stride=2, padding=1)

    for position, (stage, depth) in enumerate(zip(_STAGES, depths)):
        for index in range(depth):
            hidden = _block(
                network, hidden, weights, f"{stage}.{index}", numpy_dtype,
                stride=2 if position > 0 and index == 0 else 1,
            )

    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    logits = graph.classifier(
        network, hidden, weights["fc.weight"], weights["fc.bias"], dtype=numpy_dtype
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the ResNeSt output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm ResNeSt engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm ResNeSt image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_resnest does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_resnest does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_resnest does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_resnest does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_resnest does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_resnest does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_resnest does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_resnest supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_resnest does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_resnest does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_resnest", task=request.task, backend=request.backend)
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
