# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm HRNet classifiers.

HRNet keeps several resolutions alive at once instead of narrowing to one. Each
stage runs one branch per resolution and then fuses every branch into every
other, so the graph is a grid rather than a chain.

The grid is read from the checkpoint: module count, branch count, and per-branch
block count for each stage, plus the `layer1` depth. Transitions are the one
thing not counted from their own keys, because an identity transition carries no
weights and leaves no trace; the branch count of the stage it feeds is
authoritative instead.
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
# The stages, in order, with the transition that feeds each one.
_STAGES = (("transition1", "stage2"), ("transition2", "stage3"), ("transition3", "stage4"))


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"HRNet model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("HRNet config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("hrnet"):
        raise ValueError(f"unsupported timm HRNet model identity: {identity!r}")
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
        raise ValueError("HRNet pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("HRNet image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("HRNet image std must contain three values")
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
        raise ValueError("HRNet preprocessing or classifier config is invalid")
    return result


def _count(names, pattern: str) -> int:
    regex = re.compile(pattern)
    indices = {int(match.group(1)) for match in map(regex.match, names) if match}
    if not indices:
        return 0
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"HRNet indices for {pattern} are not contiguous")
    return len(indices)


def _layout(checkpoint: Checkpoint) -> dict[str, Any]:
    names = checkpoint.names
    layer1 = _count(names, r"^layer1\.(\d+)\.")
    if layer1 == 0:
        raise ValueError("HRNet checkpoint has no layer1 blocks")
    stages: list[dict[str, Any]] = []
    for _, stage in _STAGES:
        modules = _count(names, rf"^{stage}\.(\d+)\.")
        if modules == 0:
            raise ValueError(f"HRNet checkpoint has no {stage} modules")
        branches = _count(names, rf"^{stage}\.0\.branches\.(\d+)\.")
        if branches == 0:
            raise ValueError(f"HRNet checkpoint has no {stage} branches")
        blocks = [
            _count(names, rf"^{stage}\.0\.branches\.{branch}\.(\d+)\.")
            for branch in range(branches)
        ]
        stages.append(
            {"name": stage, "modules": modules, "branches": branches, "blocks": blocks}
        )
    head_inputs = _count(names, r"^incre_modules\.(\d+)\.")
    if head_inputs != stages[-1]["branches"]:
        raise ValueError(
            f"HRNet incre_modules count {head_inputs} does not match the final "
            f"branch count {stages[-1]['branches']}"
        )
    return {"layer1": layer1, "stages": stages}


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
        raise ValueError(f"HRNet norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"HRNet norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"HRNet norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    bias = beta - mean * scale
    if f"{conv[: -len('.weight')]}.bias" in checkpoint.names:
        # The head convolutions carry a bias of their own; fold it in too.
        bias = bias + checkpoint.tensor(f"{conv[: -len('.weight')]}.bias") * scale
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), bias.astype(dtype)


def _residual(network, tensor, weights, prefix: str, dtype, *, bottleneck: bool):
    """A basic block or a bottleneck, with an optional projection shortcut."""
    shortcut = tensor
    leaves = ("conv1", "conv2", "conv3") if bottleneck else ("conv1", "conv2")
    for position, leaf in enumerate(leaves, start=1):
        weight = weights[f"{prefix}.{leaf}.weight"]
        tensor = graph.convolution(
            network, tensor, weight, weights[f"{prefix}.{leaf}.bias"],
            padding=int(weight.shape[2]) // 2, dtype=dtype,
        )
        if position != len(leaves):
            tensor = graph.relu(network, tensor)
    if f"{prefix}.downsample.weight" in weights:
        shortcut = graph.convolution(
            network, shortcut, weights[f"{prefix}.downsample.weight"],
            weights[f"{prefix}.downsample.bias"], dtype=dtype,
        )
    return graph.relu(network, graph.add(network, tensor, shortcut))


def _transition(network, weights, prefix: str, inputs, branches: int, dtype):
    """Widen the branch list to `branches` entries.

    An existing branch either passes through untouched or goes through one 3x3
    convolution; every new branch is built by halving the last existing branch.
    """
    outputs = []
    for index in range(branches):
        if index < len(inputs):
            if f"{prefix}.{index}.weight" in weights:
                tensor = graph.convolution(
                    network, inputs[index], weights[f"{prefix}.{index}.weight"],
                    weights[f"{prefix}.{index}.bias"], padding=1, dtype=dtype,
                )
                outputs.append(graph.relu(network, tensor))
            else:
                outputs.append(inputs[index])
            continue
        tensor = inputs[-1]
        step = 0
        while f"{prefix}.{index}.{step}.weight" in weights:
            tensor = graph.convolution(
                network, tensor, weights[f"{prefix}.{index}.{step}.weight"],
                weights[f"{prefix}.{index}.{step}.bias"], stride=2, padding=1, dtype=dtype,
            )
            tensor = graph.relu(network, tensor)
            step += 1
        if step == 0:
            raise ValueError(f"HRNet {prefix}.{index} has no downsampling convolutions")
        outputs.append(tensor)
    return outputs


def _fuse(network, weights, prefix: str, inputs, dtype):
    """Sum every branch into every branch, rescaling to match resolutions.

    A higher-index branch is at lower resolution, so it is projected with a 1x1
    convolution and upsampled. A lower-index branch is downsampled by a chain of
    strided 3x3 convolutions, and only the last of those omits its activation,
    because its output feeds the sum rather than another convolution.
    """
    outputs = []
    for row in range(len(inputs)):
        total = None
        for column, source in enumerate(inputs):
            if column == row:
                term = source
            elif column > row:
                term = graph.convolution(
                    network, source, weights[f"{prefix}.{row}.{column}.weight"],
                    weights[f"{prefix}.{row}.{column}.bias"], dtype=dtype,
                )
                term = graph.nearest_upsample(network, term, 2 ** (column - row))
            else:
                term = source
                for step in range(row - column):
                    term = graph.convolution(
                        network, term, weights[f"{prefix}.{row}.{column}.{step}.weight"],
                        weights[f"{prefix}.{row}.{column}.{step}.bias"],
                        stride=2, padding=1, dtype=dtype,
                    )
                    if step != row - column - 1:
                        term = graph.relu(network, term)
            total = term if total is None else graph.add(network, total, term)
        outputs.append(graph.relu(network, total))
    return outputs


def _weights(checkpoint: Checkpoint, layout: dict[str, Any], dtype: np.dtype):
    result: dict[str, np.ndarray] = {}

    def fold(conv: str, norm: str, key: str) -> None:
        weight, bias = _fold(checkpoint, conv, norm, dtype)
        result[f"{key}.weight"], result[f"{key}.bias"] = weight, bias

    fold("conv1.weight", "bn1", "stem.0")
    fold("conv2.weight", "bn2", "stem.1")
    for index in range(layout["layer1"]):
        prefix = f"layer1.{index}"
        for position, leaf in enumerate(("conv1", "conv2", "conv3"), start=1):
            fold(f"{prefix}.{leaf}.weight", f"{prefix}.bn{position}", f"{prefix}.{leaf}")
        if f"{prefix}.downsample.0.weight" in checkpoint.names:
            fold(f"{prefix}.downsample.0.weight", f"{prefix}.downsample.1", f"{prefix}.downsample")

    for (transition, _), stage in zip(_STAGES, layout["stages"]):
        for index in range(stage["branches"]):
            if f"{transition}.{index}.0.weight" in checkpoint.names:
                fold(f"{transition}.{index}.0.weight", f"{transition}.{index}.1",
                     f"{transition}.{index}")
            step = 0
            while f"{transition}.{index}.{step}.0.weight" in checkpoint.names:
                fold(f"{transition}.{index}.{step}.0.weight",
                     f"{transition}.{index}.{step}.1", f"{transition}.{index}.{step}")
                step += 1
        name = stage["name"]
        for module in range(stage["modules"]):
            for branch, depth in enumerate(stage["blocks"]):
                for index in range(depth):
                    prefix = f"{name}.{module}.branches.{branch}.{index}"
                    for position, leaf in enumerate(("conv1", "conv2"), start=1):
                        fold(f"{prefix}.{leaf}.weight", f"{prefix}.bn{position}",
                             f"{prefix}.{leaf}")
                    if f"{prefix}.downsample.0.weight" in checkpoint.names:
                        fold(f"{prefix}.downsample.0.weight", f"{prefix}.downsample.1",
                             f"{prefix}.downsample")
            fuse = f"{name}.{module}.fuse_layers"
            for row in range(stage["branches"]):
                for column in range(stage["branches"]):
                    if column > row:
                        fold(f"{fuse}.{row}.{column}.0.weight", f"{fuse}.{row}.{column}.1",
                             f"{fuse}.{row}.{column}")
                    elif column < row:
                        for step in range(row - column):
                            fold(f"{fuse}.{row}.{column}.{step}.0.weight",
                                 f"{fuse}.{row}.{column}.{step}.1",
                                 f"{fuse}.{row}.{column}.{step}")

    branches = layout["stages"][-1]["branches"]
    for index in range(branches):
        prefix = f"incre_modules.{index}.0"
        for position, leaf in enumerate(("conv1", "conv2", "conv3"), start=1):
            fold(f"{prefix}.{leaf}.weight", f"{prefix}.bn{position}", f"incre.{index}.{leaf}")
        fold(f"{prefix}.downsample.0.weight", f"{prefix}.downsample.1", f"incre.{index}.downsample")
    for index in range(branches - 1):
        fold(f"downsamp_modules.{index}.0.weight", f"downsamp_modules.{index}.1",
             f"downsamp.{index}")
    fold("final_layer.0.weight", "final_layer.1", "final")
    result["classifier.weight"] = checkpoint.tensor("classifier.weight").astype(dtype)
    result["classifier.bias"] = checkpoint.tensor("classifier.bias").astype(dtype)
    if result["classifier.weight"].ndim != 2 or result["classifier.bias"].shape != (
        result["classifier.weight"].shape[0],
    ):
        raise ValueError("HRNet classifier weights have incompatible shapes")
    return result


def _head(network, weights, branches, dtype):
    """Collapse the branches into one tensor at the lowest resolution.

    Each branch gets its own bottleneck, then the running sum is downsampled to
    meet the next branch.
    """
    total = None
    for index, source in enumerate(branches):
        widened = _residual(network, source, weights, f"incre.{index}", dtype, bottleneck=True)
        if total is None:
            total = widened
            continue
        reduced = graph.convolution(
            network, total, weights[f"downsamp.{index - 1}.weight"],
            weights[f"downsamp.{index - 1}.bias"], stride=2, padding=1, dtype=dtype,
        )
        total = graph.add(network, widened, graph.relu(network, reduced))
    tensor = graph.convolution(
        network, total, weights["final.weight"], weights["final.bias"], dtype=dtype
    )
    return graph.relu(network, tensor)


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
        raise ValueError(f"unsupported timm HRNet precision: {precision}")
    config = _preprocess_config(raw)
    layout = _layout(checkpoint)
    weights = _weights(checkpoint, layout, numpy_dtype)
    if weights["classifier.weight"].shape[0] != config["num_classes"]:
        raise ValueError("HRNet classifier dimensions do not match config.json")

    height = config["image_height"]
    width = config["image_width"]
    if verbose:
        print(
            "[trtmc build] timm_hrnet: "
            f"image={height}x{width}, "
            f"branches={[stage['branches'] for stage in layout['stages']]}, "
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
        raise RuntimeError("TensorRT rejected the HRNet input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the HRNet input cast")
        hidden = cast.get_output(0)

    # Two strided 3x3 convolutions take the input to one quarter size.
    for stem in ("stem.0", "stem.1"):
        hidden = graph.convolution(
            network, hidden, weights[f"{stem}.weight"], weights[f"{stem}.bias"],
            stride=2, padding=1, dtype=numpy_dtype,
        )
        hidden = graph.relu(network, hidden)

    for index in range(layout["layer1"]):
        hidden = _residual(network, hidden, weights, f"layer1.{index}", numpy_dtype,
                           bottleneck=True)

    branches = [hidden]
    for (transition, _), stage in zip(_STAGES, layout["stages"]):
        branches = _transition(
            network, weights, transition, branches, stage["branches"], numpy_dtype
        )
        for module in range(stage["modules"]):
            prefix = f"{stage['name']}.{module}"
            encoded = []
            for branch, depth in enumerate(stage["blocks"]):
                tensor = branches[branch]
                for index in range(depth):
                    tensor = _residual(
                        network, tensor, weights, f"{prefix}.branches.{branch}.{index}",
                        numpy_dtype, bottleneck=False,
                    )
                encoded.append(tensor)
            branches = _fuse(network, weights, f"{prefix}.fuse_layers", encoded, numpy_dtype)

    hidden = _head(network, weights, branches, numpy_dtype)
    shape = hidden.shape
    hidden = graph.global_average_pool(network, hidden, int(shape[2]), int(shape[3]))
    logits = graph.classifier(
        network, hidden, weights["classifier.weight"], weights["classifier.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the HRNet output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm HRNet engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm HRNet image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_hrnet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_hrnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_hrnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_hrnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_hrnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_hrnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_hrnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_hrnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_hrnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_hrnet does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_hrnet", task=request.task, backend=request.backend)
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
