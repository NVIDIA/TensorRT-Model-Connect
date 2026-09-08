# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm SENet classifiers.

SENet-154 is the original squeeze-and-excitation network rather than a ResNet
with a gate attached, so three things differ from `timm_seresnet` and all three
are read from the checkpoint: the stem is three 3x3 convolutions instead of one
7x7, the bottleneck 3x3 is grouped, and the projection shortcut is itself a 3x3
convolution in every stage that reduces.

A checkpoint without a squeeze-excitation gate is rejected rather than built
silently without it.
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


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"SENet model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SENet config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("senet"):
        raise ValueError(f"unsupported timm SENet model identity: {identity!r}")
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
        raise ValueError("SENet pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("SENet image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("SENet image std must contain three values")
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
        raise ValueError("SENet preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> dict[str, Any]:
    names = checkpoint.names
    depths: list[int] = []
    for stage in _STAGES:
        pattern = re.compile(rf"^{stage}\.(\d+)\.")
        indices = {int(match.group(1)) for match in map(pattern.match, names) if match}
        if not indices:
            raise ValueError(f"SENet checkpoint has no blocks for stage {stage}")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"SENet stage {stage} block indices are not contiguous")
        depths.append(len(indices))
    if not any(name.startswith("layer1.0.se.") for name in names):
        raise ValueError(
            "SENet checkpoint has no squeeze-excitation gate; a plain ResNet "
            "belongs to a different family"
        )
    # A third convolution is what separates a bottleneck from a basic block.
    return {"depths": depths, "bottleneck": "layer1.0.conv3.weight" in names}


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
        raise ValueError(f"SENet norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"SENet norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"SENet norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    layout: dict[str, Any],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    # The stem is an nn.Sequential of convolutions with norms interleaved, so
    # the convolutions are picked out by tensor rank rather than by position.
    # The last convolution's norm is the top-level bn1, not a member of the
    # sequence.
    stem = sorted(
        int(match.group(1))
        for match in (re.match(r"^conv1\.(\d+)\.weight$", name) for name in checkpoint.names)
        if match and checkpoint.tensor(match.group(0)).ndim == 4
    )
    if not stem:
        raise ValueError("SENet checkpoint has no deep stem convolutions")
    for position, index in enumerate(stem):
        norm = "bn1" if position == len(stem) - 1 else f"conv1.{index + 1}"
        weight, bias = _fold_norm(checkpoint, f"conv1.{index}.weight", norm, dtype)
        result[f"stem.{position}.weight"] = weight
        result[f"stem.{position}.bias"] = bias
    result["stem_depth"] = np.asarray(len(stem))
    convolutions = ("conv1", "conv2", "conv3") if layout["bottleneck"] else ("conv1", "conv2")
    for stage, depth in zip(_STAGES, layout["depths"]):
        for index in range(depth):
            prefix = f"{stage}.{index}"
            for position, leaf in enumerate(convolutions, start=1):
                weight, bias = _fold_norm(
                    checkpoint, f"{prefix}.{leaf}.weight", f"{prefix}.bn{position}", dtype
                )
                result[f"{prefix}.{leaf}.weight"] = weight
                result[f"{prefix}.{leaf}.bias"] = bias
            if f"{prefix}.downsample.0.weight" in checkpoint.names:
                weight, bias = _fold_norm(
                    checkpoint, f"{prefix}.downsample.0.weight", f"{prefix}.downsample.1", dtype
                )
                result[f"{prefix}.downsample.weight"] = weight
                result[f"{prefix}.downsample.bias"] = bias
            for leaf in ("fc1", "fc2"):
                result[f"{prefix}.se.{leaf}.weight"] = checkpoint.tensor(
                    f"{prefix}.se.{leaf}.weight"
                ).astype(dtype)
                result[f"{prefix}.se.{leaf}.bias"] = checkpoint.tensor(
                    f"{prefix}.se.{leaf}.bias"
                ).astype(dtype)
    result["fc.weight"] = checkpoint.tensor("fc.weight").astype(dtype)
    result["fc.bias"] = checkpoint.tensor("fc.bias").astype(dtype)
    if result["fc.weight"].ndim != 2 or result["fc.bias"].shape != (result["fc.weight"].shape[0],):
        raise ValueError("SENet classifier weights have incompatible shapes")
    return result


def _block(network, tensor, weights, prefix: str, dtype, *, stride: int, bottleneck: bool):
    """One residual block whose output is gated before the residual add.

    timm puts the spatial stride on the 3x3 convolution and applies the gate
    after the last norm but before the add, so the shortcut is never gated.
    """
    identity = tensor
    if bottleneck:
        tensor = graph.convolution(
            network, tensor, weights[f"{prefix}.conv1.weight"],
            weights[f"{prefix}.conv1.bias"], dtype=dtype,
        )
        tensor = graph.relu(network, tensor)
        grouped = weights[f"{prefix}.conv2.weight"]
        # Grouped (SE-ResNeXt) convolutions store (out, in / groups, kh, kw).
        groups = max(1, int(weights[f"{prefix}.conv1.weight"].shape[0]) // int(grouped.shape[1]))
        tensor = graph.convolution(
            network, tensor, grouped, weights[f"{prefix}.conv2.bias"],
            stride=stride, padding=1, groups=groups, dtype=dtype,
        )
        tensor = graph.relu(network, tensor)
        tensor = graph.convolution(
            network, tensor, weights[f"{prefix}.conv3.weight"],
            weights[f"{prefix}.conv3.bias"], dtype=dtype,
        )
    else:
        tensor = graph.convolution(
            network, tensor, weights[f"{prefix}.conv1.weight"],
            weights[f"{prefix}.conv1.bias"], stride=stride, padding=1, dtype=dtype,
        )
        tensor = graph.relu(network, tensor)
        tensor = graph.convolution(
            network, tensor, weights[f"{prefix}.conv2.weight"],
            weights[f"{prefix}.conv2.bias"], padding=1, dtype=dtype,
        )
    tensor = graph.squeeze_excite(
        network, tensor,
        weights[f"{prefix}.se.fc1.weight"], weights[f"{prefix}.se.fc1.bias"],
        weights[f"{prefix}.se.fc2.weight"], weights[f"{prefix}.se.fc2.bias"],
        dtype=dtype,
    )
    if f"{prefix}.downsample.weight" in weights:
        shortcut = weights[f"{prefix}.downsample.weight"]
        # SENet-154 projects with a 3x3 wherever the stage reduces and a 1x1 in
        # the first stage, so the padding follows the kernel it actually has.
        identity = graph.convolution(
            network, identity, shortcut, weights[f"{prefix}.downsample.bias"],
            stride=stride, padding=int(shortcut.shape[2]) // 2, dtype=dtype,
        )
    return graph.relu(network, graph.add(network, tensor, identity))


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
        raise ValueError(f"unsupported timm SENet precision: {precision}")
    config = _preprocess_config(raw)
    layout = _layout(checkpoint)
    weights = _weights(checkpoint, layout, numpy_dtype)
    if weights["fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("SENet classifier dimensions do not match config.json")

    # The stem strides twice, then every stage after the first halves again.
    total_stride = 4 * 2 ** (len(_STAGES) - 1)
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"SENet input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_senet: "
            f"image={height}x{width}, depths={layout['depths']}, "
            f"bottleneck={layout['bottleneck']}, classes={config['num_classes']}, "
            f"precision={precision}",
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
        raise RuntimeError("TensorRT rejected the SENet input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the SENet input cast")
        hidden = cast.get_output(0)

    for position in range(int(weights["stem_depth"])):
        hidden = graph.convolution(
            network, hidden, weights[f"stem.{position}.weight"],
            weights[f"stem.{position}.bias"],
            # Only the first stem convolution strides.
            stride=2 if position == 0 else 1, padding=1, dtype=numpy_dtype,
        )
        hidden = graph.relu(network, hidden)
    hidden = graph.max_pool(network, hidden, kernel=3, stride=2, padding=1)

    for position, (stage, depth) in enumerate(zip(_STAGES, layout["depths"])):
        for index in range(depth):
            hidden = _block(
                network, hidden, weights, f"{stage}.{index}", numpy_dtype,
                # Only the first block of stages after layer1 reduces.
                stride=2 if position > 0 and index == 0 else 1,
                bottleneck=bool(layout["bottleneck"]),
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
            raise RuntimeError("TensorRT rejected the SENet output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm SENet engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm SENet image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_senet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_senet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_senet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_senet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_senet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_senet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_senet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_senet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_senet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_senet does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_senet", task=request.task, backend=request.backend)
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
