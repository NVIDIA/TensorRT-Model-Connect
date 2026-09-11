# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm Inception-ResNet-v2 classifiers.

The depth of each repeat group is read from the checkpoint. Two things are not
recoverable from weights and are stated as constants: the factor each group
scales its residual branch by, and the fact that the trailing block adds its
branch unscaled and skips the activation after the add.
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


# Inception-ResNet-v2 is a TensorFlow port and uses the TensorFlow batch-norm epsilon.
_BATCH_NORM_EPSILON = 1e-3

# A "same"-style branch pads according to its kernel; the factorised 1xN and Nx1
# convolutions pad on one axis only. Strided reduction convolutions pad zero and
# pass their padding explicitly.
_SAME_PADDING = {
    (1, 1): 0, (3, 3): 1, (5, 5): 2, (1, 7): (0, 3), (7, 1): (3, 0),
    (1, 3): (0, 1), (3, 1): (1, 0),
}

# The stem, in order. A "maxpool" entry is a 3x3 stride-2 pool with no weights.
_STEM = (
    ("conv2d_1a", 2, 0),
    ("conv2d_2a", 1, 0),
    ("conv2d_2b", 1, 1),
    ("maxpool", 0, 0),
    ("conv2d_3b", 1, 0),
    ("conv2d_4a", 1, 0),
    ("maxpool", 0, 0),
)

# The residual scale per repeat group. These are architecture constants; the
# checkpoint records nothing about them, and a wrong value still builds.
_GROUP_SCALE = {"repeat": 0.17, "repeat_1": 0.10, "repeat_2": 0.20}

# The trailing standalone block adds its branch unscaled and does not activate.
_FINAL_BLOCK_SCALE = 1.0



def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"InceptionResNet model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("InceptionResNet config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("inception_resnet"):
        raise ValueError(f"unsupported timm Inception-ResNet-v2 model identity: {identity!r}")
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
        raise ValueError("InceptionResNet pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("InceptionResNet image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("InceptionResNet image std must contain three values")
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
        raise ValueError("InceptionResNet preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> dict[str, int]:
    """Read each repeat group's length from the checkpoint."""
    names = checkpoint.names
    groups: dict[str, int] = {}
    for group in _GROUP_SCALE:
        pattern = re.compile(rf"^{group}\.(\d+)\.")
        indices = {int(match.group(1)) for match in map(pattern.match, names) if match}
        if not indices:
            raise ValueError(f"Inception-ResNet-v2 checkpoint has no {group} blocks")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Inception-ResNet-v2 {group} block indices are not contiguous")
        groups[group] = len(indices)
    for required in ("mixed_5b", "mixed_6a", "mixed_7a", "block8", "conv2d_7b"):
        if not any(name.startswith(required + ".") for name in names):
            raise ValueError(f"Inception-ResNet-v2 checkpoint is missing {required}")
    return groups


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
        raise ValueError(f"Inception-ResNet-v2 norm {prefix}.bn has mismatched shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"Inception-ResNet-v2 norm {prefix}.bn does not match its convolution")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"Inception-ResNet-v2 norm {prefix}.bn has a non-positive variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(checkpoint: Checkpoint, dtype: np.dtype) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in sorted(checkpoint.names):
        if name.endswith(".conv.weight"):
            prefix = name[: -len(".conv.weight")]
            weight, bias = _fold(checkpoint, prefix, dtype)
            result[f"{prefix}.weight"], result[f"{prefix}.bias"] = weight, bias
    # The 1x1 projection inside each residual block is a plain biased
    # convolution with no batch norm, unlike every other convolution here.
    for name in sorted(checkpoint.names):
        if name.endswith(".conv2d.weight"):
            prefix = name[: -len(".weight")]
            result[f"{prefix}.weight"] = checkpoint.tensor(f"{prefix}.weight").astype(dtype)
            result[f"{prefix}.bias"] = checkpoint.tensor(f"{prefix}.bias").astype(dtype)
    result["classifier.weight"] = checkpoint.tensor("classif.weight").astype(dtype)
    result["classifier.bias"] = checkpoint.tensor("classif.bias").astype(dtype)
    if result["classifier.weight"].ndim != 2 or result["classifier.bias"].shape != (
        result["classifier.weight"].shape[0],
    ):
        raise ValueError("Inception-ResNet-v2 classifier weights have incompatible shapes")
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
        raise RuntimeError("TensorRT rejected an Inception-ResNet-v2 convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (vertical, horizontal)
    return graph.relu(network, layer.get_output(0))


def _chain(network, tensor, weights, prefix: str, dtype):
    step = 0
    while f"{prefix}.{step}.weight" in weights:
        tensor = _conv(network, tensor, weights, f"{prefix}.{step}", dtype)
        step += 1
    if step == 0:
        raise ValueError(f"Inception-ResNet-v2 {prefix} has no convolutions")
    return tensor


def _strided_chain(network, tensor, weights, prefix: str, dtype):
    """A chain whose final convolution carries the stride and pads zero."""
    steps = 0
    while f"{prefix}.{steps}.weight" in weights:
        steps += 1
    for step in range(steps):
        last = step == steps - 1
        tensor = _conv(
            network, tensor, weights, f"{prefix}.{step}", dtype,
            stride=2 if last else 1, padding=0 if last else None,
        )
    return tensor


def _residual_block(network, tensor, weights, prefix: str, factor: float, dtype, *, activate: bool):
    """Concatenated branches, a biased 1x1 projection, then a scaled add."""
    branches = [_conv(network, tensor, weights, f"{prefix}.branch0", dtype)]
    for index in (1, 2):
        name = f"{prefix}.branch{index}"
        if f"{name}.0.weight" in weights:
            branches.append(_chain(network, tensor, weights, name, dtype))
    merged = graph.concatenate(network, branches)
    projection = weights[f"{prefix}.conv2d.weight"]
    layer = network.add_convolution_nd(
        merged,
        num_output_maps=int(projection.shape[0]),
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(projection, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(weights[f"{prefix}.conv2d.bias"], dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected an Inception-ResNet-v2 projection")
    scaled = graph.scale(network, layer.get_output(0), factor, dtype=dtype)
    summed = graph.add(network, scaled, tensor)
    return graph.relu(network, summed) if activate else summed


def _mixed_5b(network, tensor, weights, dtype):
    branch0 = _conv(network, tensor, weights, "mixed_5b.branch0", dtype)
    branch1 = _chain(network, tensor, weights, "mixed_5b.branch1", dtype)
    branch2 = _chain(network, tensor, weights, "mixed_5b.branch2", dtype)
    branch3 = graph.average_pool(network, tensor, kernel=3, stride=1, padding=1)
    branch3 = _conv(network, branch3, weights, "mixed_5b.branch3.1", dtype)
    return graph.concatenate(network, [branch0, branch1, branch2, branch3])


def _reduction(network, tensor, weights, prefix: str, dtype):
    """Strided branches beside a strided max pool."""
    if f"{prefix}.branch0.weight" in weights:
        branches = [_conv(network, tensor, weights, f"{prefix}.branch0", dtype, stride=2, padding=0)]
    else:
        branches = [_strided_chain(network, tensor, weights, f"{prefix}.branch0", dtype)]
    index = 1
    while f"{prefix}.branch{index}.0.weight" in weights:
        branches.append(_strided_chain(network, tensor, weights, f"{prefix}.branch{index}", dtype))
        index += 1
    branches.append(graph.max_pool(network, tensor, kernel=3, stride=2, padding=0))
    return graph.concatenate(network, branches)


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
        raise ValueError(f"unsupported timm Inception-ResNet-v2 precision: {precision}")
    config = _preprocess_config(raw)
    groups = _layout(checkpoint)
    weights = _weights(checkpoint, numpy_dtype)
    if weights["classifier.weight"].shape[0] != config["num_classes"]:
        raise ValueError("Inception-ResNet-v2 classifier dimensions do not match config.json")

    height = config["image_height"]
    width = config["image_width"]
    if verbose:
        print(
            "[trtmc build] timm_inception_resnet: "
            f"image={height}x{width}, groups={groups}, "
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
        raise RuntimeError("TensorRT rejected the Inception-ResNet-v2 input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Inception-ResNet-v2 input cast")
        hidden = cast.get_output(0)

    for name, stride, padding in _STEM:
        if name == "maxpool":
            hidden = graph.max_pool(network, hidden, kernel=3, stride=2, padding=0)
            continue
        hidden = _conv(network, hidden, weights, name, numpy_dtype, stride=stride, padding=padding)

    hidden = _mixed_5b(network, hidden, weights, numpy_dtype)
    for index in range(groups["repeat"]):
        hidden = _residual_block(
            network, hidden, weights, f"repeat.{index}",
            _GROUP_SCALE["repeat"], numpy_dtype, activate=True,
        )
    hidden = _reduction(network, hidden, weights, "mixed_6a", numpy_dtype)
    for index in range(groups["repeat_1"]):
        hidden = _residual_block(
            network, hidden, weights, f"repeat_1.{index}",
            _GROUP_SCALE["repeat_1"], numpy_dtype, activate=True,
        )
    hidden = _reduction(network, hidden, weights, "mixed_7a", numpy_dtype)
    for index in range(groups["repeat_2"]):
        hidden = _residual_block(
            network, hidden, weights, f"repeat_2.{index}",
            _GROUP_SCALE["repeat_2"], numpy_dtype, activate=True,
        )
    # The trailing block adds its branch unscaled and does not activate.
    hidden = _residual_block(
        network, hidden, weights, "block8", _FINAL_BLOCK_SCALE, numpy_dtype, activate=False
    )
    hidden = _conv(network, hidden, weights, "conv2d_7b", numpy_dtype)

    shape = hidden.shape
    hidden = graph.global_average_pool(network, hidden, int(shape[2]), int(shape[3]))
    logits = graph.classifier(
        network, hidden, weights["classifier.weight"], weights["classifier.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Inception-ResNet-v2 output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm Inception-ResNet-v2 engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm Inception-ResNet-v2 image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_inception_resnet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_inception_resnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_inception_resnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_inception_resnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_inception_resnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_inception_resnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_inception_resnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_inception_resnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_inception_resnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_inception_resnet does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_inception_resnet", task=request.task, backend=request.backend)
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
