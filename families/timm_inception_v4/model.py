# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm Inception-v4 classifiers.

Every block is named from the keys the checkpoint carries for it rather than
from a depth table. The pooling branches carry no weights, so several
topologies present the same top-level branch names and have to be told apart one
level deeper.

Two blocks cannot be separated by their weights at all: Mixed4a and Reduction-B
have identical branch shapes and differ only in stride, which a checkpoint does
not record. Their order in the network is the only honest discriminator, so the
first is Mixed4a and any later one is Reduction-B.
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


# Inception-v4 is a TensorFlow port and uses the TensorFlow batch-norm epsilon.
_BATCH_NORM_EPSILON = 1e-3

# A "same"-style branch pads according to its kernel; the factorised 1xN and Nx1
# convolutions pad on one axis only. Strided reduction convolutions pad zero and
# pass their padding explicitly.
_SAME_PADDING = {
    (1, 1): 0, (3, 3): 1, (1, 7): (0, 3), (7, 1): (3, 0), (1, 3): (0, 1), (3, 1): (1, 0),
}

# The three stem convolutions, in order, as (stride, padding).
_STEM = ((2, 0), (1, 0), (1, 1))
_BLOCK = re.compile(r"^features\.(\d+)\.(.+)$")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"InceptionV4 model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("InceptionV4 config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("inception_v4"):
        raise ValueError(f"unsupported timm Inception-v4 model identity: {identity!r}")
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
        raise ValueError("InceptionV4 pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("InceptionV4 image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("InceptionV4 image std must contain three values")
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
        raise ValueError("InceptionV4 preprocessing or classifier config is invalid")
    return result


def _classify(index: int, branches: set[str], keys: set[str]) -> str:
    """Name a block from the keys the checkpoint carries for it."""
    if branches == {"conv"}:
        # A single convolution beside a weightless pooling branch.
        return "mixed3a" if index == 3 else "mixed5a"
    if "branch1_0" in branches:
        return "inception_c"
    if branches == {"branch0", "branch1", "branch2", "branch3"}:
        return "inception_ab"
    if branches == {"branch0", "branch1"}:
        # Reduction-A's first branch is a single convolution, so it is
        # distinguishable one level deeper. Mixed4a and Reduction-B are not.
        if any(key.startswith("branch0.conv") for key in keys):
            return "reduction_a"
        return "chained_pair"
    raise ValueError(f"features.{index}: unrecognised Inception-v4 block topology")


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    tops: dict[int, set[str]] = {}
    detail: dict[int, set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.fullmatch(name)
        if match:
            index = int(match.group(1))
            tops.setdefault(index, set()).add(match.group(2).split(".", 1)[0])
            detail.setdefault(index, set()).add(match.group(2))
    if not tops:
        raise ValueError("Inception-v4 checkpoint has no features.<index> tensors")
    indices = sorted(tops)
    if indices != list(range(len(indices))):
        raise ValueError("Inception-v4 block indices are not contiguous")
    blocks: list[dict[str, object]] = []
    seen_chained_pair = False
    for index in indices:
        if index < len(_STEM):
            blocks.append({"index": index, "kind": "stem"})
            continue
        kind = _classify(index, tops[index], detail[index])
        if kind == "chained_pair":
            # The first such block is Mixed4a, which keeps its resolution; any
            # later one is Reduction-B, which halves it.
            kind = "reduction_b" if seen_chained_pair else "mixed4a"
            seen_chained_pair = True
        blocks.append({"index": index, "kind": kind})
    return blocks


def _fold(
    checkpoint: Checkpoint, prefix: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a ConvNormAct's batch norm into its convolution."""
    weight = checkpoint.tensor(f"{prefix}.conv.weight")
    gamma = checkpoint.tensor(f"{prefix}.bn.weight")
    beta = checkpoint.tensor(f"{prefix}.bn.bias")
    mean = checkpoint.tensor(f"{prefix}.bn.running_mean")
    variance = checkpoint.tensor(f"{prefix}.bn.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"Inception-v4 norm {prefix}.bn has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"Inception-v4 norm {prefix}.bn does not match its convolution")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"Inception-v4 norm {prefix}.bn has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(checkpoint: Checkpoint, dtype: np.dtype) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    # Every convolution here is a ConvNormAct, so fold them all uniformly.
    for name in sorted(checkpoint.names):
        if name.endswith(".conv.weight"):
            prefix = name[: -len(".conv.weight")]
            weight, bias = _fold(checkpoint, prefix, dtype)
            result[f"{prefix}.weight"], result[f"{prefix}.bias"] = weight, bias
    result["classifier.weight"] = checkpoint.tensor("last_linear.weight").astype(dtype)
    result["classifier.bias"] = checkpoint.tensor("last_linear.bias").astype(dtype)
    if result["classifier.weight"].ndim != 2 or result["classifier.bias"].shape != (
        result["classifier.weight"].shape[0],
    ):
        raise ValueError("Inception-v4 classifier weights have incompatible shapes")
    return result


def _conv(network, tensor, weights, prefix: str, dtype, *, stride: int = 1, padding=None):
    weight = weights[f"{prefix}.weight"]
    kernel = (int(weight.shape[2]), int(weight.shape[3]))
    pad = _SAME_PADDING[kernel] if padding is None else padding
    vertical, horizontal = (pad, pad) if isinstance(pad, int) else pad
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=int(weight.shape[0]),
        kernel_shape=kernel,
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(weights[f"{prefix}.bias"], dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected an Inception-v4 convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (vertical, horizontal)
    return graph.relu(network, layer.get_output(0))


def _chain(network, tensor, weights, prefix: str, dtype):
    """Apply prefix.0, prefix.1, ... for as many steps as the weights define."""
    step = 0
    while f"{prefix}.{step}.weight" in weights:
        tensor = _conv(network, tensor, weights, f"{prefix}.{step}", dtype)
        step += 1
    if step == 0:
        raise ValueError(f"Inception-v4 {prefix} has no convolutions")
    return tensor


def _chain_then_valid(network, tensor, weights, prefix: str, dtype):
    """A chain whose final convolution pads zero instead of "same"."""
    steps = 0
    while f"{prefix}.{steps}.weight" in weights:
        steps += 1
    if steps == 0:
        raise ValueError(f"Inception-v4 {prefix} has no convolutions")
    for step in range(steps):
        tensor = _conv(
            network, tensor, weights, f"{prefix}.{step}", dtype,
            padding=0 if step == steps - 1 else None,
        )
    return tensor


def _block(network, tensor, weights, index: int, kind: str, dtype):
    name = f"features.{index}"
    if kind == "mixed3a":
        pool = graph.max_pool(network, tensor, kernel=3, stride=2, padding=0)
        conv = _conv(network, tensor, weights, f"{name}.conv", dtype, stride=2, padding=0)
        return graph.concatenate(network, [pool, conv])
    if kind == "mixed5a":
        conv = _conv(network, tensor, weights, f"{name}.conv", dtype, stride=2, padding=0)
        pool = graph.max_pool(network, tensor, kernel=3, stride=2, padding=0)
        return graph.concatenate(network, [conv, pool])
    if kind == "mixed4a":
        # Both branches end on a convolution that pads zero, so they shrink to
        # the same size; padding that last step "same" leaves the two branches
        # two pixels apart and the concatenation fails.
        left = _conv(network, tensor, weights, f"{name}.branch0.0", dtype)
        left = _conv(network, left, weights, f"{name}.branch0.1", dtype, padding=0)
        right = _chain_then_valid(network, tensor, weights, f"{name}.branch1", dtype)
        return graph.concatenate(network, [left, right])
    if kind == "inception_ab":
        # InceptionA and InceptionB share a shape: a 1x1 branch, two chains of
        # differing depth, and a pooled branch. Walk whatever depth the
        # checkpoint declares rather than tabulating it.
        branch0 = _conv(network, tensor, weights, f"{name}.branch0", dtype)
        branch1 = _chain(network, tensor, weights, f"{name}.branch1", dtype)
        branch2 = _chain(network, tensor, weights, f"{name}.branch2", dtype)
        branch3 = graph.average_pool(network, tensor, kernel=3, stride=1, padding=1)
        branch3 = _conv(network, branch3, weights, f"{name}.branch3.1", dtype)
        return graph.concatenate(network, [branch0, branch1, branch2, branch3])
    if kind == "reduction_a":
        branch0 = _conv(network, tensor, weights, f"{name}.branch0", dtype, stride=2, padding=0)
        branch1 = _conv(network, tensor, weights, f"{name}.branch1.0", dtype)
        branch1 = _conv(network, branch1, weights, f"{name}.branch1.1", dtype)
        branch1 = _conv(network, branch1, weights, f"{name}.branch1.2", dtype, stride=2, padding=0)
        pool = graph.max_pool(network, tensor, kernel=3, stride=2, padding=0)
        return graph.concatenate(network, [branch0, branch1, pool])
    if kind == "reduction_b":
        branch0 = _conv(network, tensor, weights, f"{name}.branch0.0", dtype)
        branch0 = _conv(network, branch0, weights, f"{name}.branch0.1", dtype, stride=2, padding=0)
        branch1 = _conv(network, tensor, weights, f"{name}.branch1.0", dtype)
        branch1 = _conv(network, branch1, weights, f"{name}.branch1.1", dtype)
        branch1 = _conv(network, branch1, weights, f"{name}.branch1.2", dtype)
        branch1 = _conv(network, branch1, weights, f"{name}.branch1.3", dtype, stride=2, padding=0)
        pool = graph.max_pool(network, tensor, kernel=3, stride=2, padding=0)
        return graph.concatenate(network, [branch0, branch1, pool])
    # inception_c: two branches split into asymmetric pairs and rejoin.
    branch0 = _conv(network, tensor, weights, f"{name}.branch0", dtype)
    branch1 = _conv(network, tensor, weights, f"{name}.branch1_0", dtype)
    branch1 = graph.concatenate(
        network,
        [
            _conv(network, branch1, weights, f"{name}.branch1_1a", dtype),
            _conv(network, branch1, weights, f"{name}.branch1_1b", dtype),
        ],
    )
    branch2 = _conv(network, tensor, weights, f"{name}.branch2_0", dtype)
    branch2 = _conv(network, branch2, weights, f"{name}.branch2_1", dtype)
    branch2 = _conv(network, branch2, weights, f"{name}.branch2_2", dtype)
    branch2 = graph.concatenate(
        network,
        [
            _conv(network, branch2, weights, f"{name}.branch2_3a", dtype),
            _conv(network, branch2, weights, f"{name}.branch2_3b", dtype),
        ],
    )
    branch3 = graph.average_pool(network, tensor, kernel=3, stride=1, padding=1)
    branch3 = _conv(network, branch3, weights, f"{name}.branch3.1", dtype)
    return graph.concatenate(network, [branch0, branch1, branch2, branch3])


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
        raise ValueError(f"unsupported timm Inception-v4 precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, numpy_dtype)
    if weights["classifier.weight"].shape[0] != config["num_classes"]:
        raise ValueError("Inception-v4 classifier dimensions do not match config.json")

    height = config["image_height"]
    width = config["image_width"]
    if verbose:
        print(
            "[trtmc build] timm_inception_v4: "
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
        raise RuntimeError("TensorRT rejected the Inception-v4 input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Inception-v4 input cast")
        hidden = cast.get_output(0)

    for block in blocks:
        index, kind = int(block["index"]), str(block["kind"])
        if kind == "stem":
            stride, padding = _STEM[index]
            hidden = _conv(
                network, hidden, weights, f"features.{index}", numpy_dtype,
                stride=stride, padding=padding,
            )
            continue
        hidden = _block(network, hidden, weights, index, kind, numpy_dtype)

    shape = hidden.shape
    hidden = graph.global_average_pool(network, hidden, int(shape[2]), int(shape[3]))
    logits = graph.classifier(
        network, hidden, weights["classifier.weight"], weights["classifier.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Inception-v4 output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm Inception-v4 engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm Inception-v4 image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_inception_v4 does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_inception_v4 does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_inception_v4 does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_inception_v4 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_inception_v4 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_inception_v4 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_inception_v4 does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_inception_v4 supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_inception_v4 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_inception_v4 does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_inception_v4", task=request.task, backend=request.backend)
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
