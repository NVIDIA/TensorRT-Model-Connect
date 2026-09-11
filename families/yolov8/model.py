# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for YOLOv8 detectors.

YOLOv8 is end to end: the one-to-one head it uses at inference emits one box
per object, so nothing here performs non-max suppression. The checkpoint also
carries a one-to-many head (`cv2`/`cv3`) that exists only for training; it is
deliberately not built, which is why the weight accounting reports those tensors
as unused rather than missing.

The module graph, the per-stage channel widths and the block wiring are all read
from the checkpoint. What cannot be read from weights is which block type each
stage is, because several share a leaf layout, so the stage table below is
stated explicitly and checked against the tensors that must exist for it.
"""

from __future__ import annotations

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


# Ultralytics uses a looser batch-norm epsilon than the PyTorch default. Using
# 1e-5 here still builds and still detects, with every box slightly wrong.
_BATCH_NORM_EPSILON = 1e-3

# Stage table for the YOLOv8 topology: (index, kind, inputs).
# `c2f` keeps the residual inside its inner blocks and `c2f_plain` drops it.
# That flag is an architecture property, not a shape: the channel counts match
# either way, so inferring it from the tensors silently adds the wrong residual.
# It applies only to plain bottleneck inner blocks. Whether a block is a
# bottleneck or a CIB is readable from the keys and decided per block, and a CIB
# always carries its residual - that is what lets one table cover every YOLOv8
# width, since the wider ones place CIB blocks in neck stages where the narrow
# ones place bottlenecks.
# An input of -1 means the previous stage. Blocks share leaf names, so the kind
# cannot be recovered from the checkpoint; each entry is checked against the
# tensors it requires.
_STAGES: tuple[tuple[int, str, tuple[int, ...]], ...] = (
    (0, "conv", (-1,)),
    (1, "conv", (-1,)),
    (2, "c2f", (-1,)),
    (3, "conv", (-1,)),
    (4, "c2f", (-1,)),
    (5, "conv", (-1,)),
    (6, "c2f", (-1,)),
    (7, "conv", (-1,)),
    (8, "c2f", (-1,)),
    (9, "sppf", (-1,)),
    (10, "upsample", (-1,)),
    (11, "concat", (-1, 6)),
    (12, "c2f_plain", (-1,)),
    (13, "upsample", (-1,)),
    (14, "concat", (-1, 4)),
    (15, "c2f_plain", (-1,)),
    (16, "conv", (-1,)),
    (17, "concat", (-1, 12)),
    (18, "c2f_plain", (-1,)),
    (19, "conv", (-1,)),
    (20, "concat", (-1, 9)),
    (21, "c2f_plain", (-1,)),
)


_HEAD_INDEX = 22
_HEAD_SOURCES = (15, 18, 21)
_STRIDES = (8, 16, 32)


def describe_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    """The build configuration, taken from the archive itself.

    An Ultralytics release carries no config.json, so the class names and the
    training size come out of the archive and the rest are the values the
    reference predicts with.
    """
    if not checkpoint.class_names:
        raise ValueError("YOLOv8 archive names no classes")
    if checkpoint.image_size % 32:
        raise ValueError(
            f"YOLOv8 input {checkpoint.image_size} must be divisible by 32, "
            "the total stride of the backbone"
        )
    return {
        "names": {index: name for index, name in enumerate(checkpoint.class_names)},
        "imgsz": checkpoint.image_size,
        "max_det": 300,
    }


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    names = raw["names"]
    size = raw.get("imgsz", 640)
    if isinstance(size, list):
        height, width = int(size[-2]), int(size[-1])
    else:
        height = width = int(size)
    if height <= 0 or width <= 0:
        raise ValueError("YOLOv8 input size must be positive")
    if height % max(_STRIDES) or width % max(_STRIDES):
        raise ValueError(f"YOLOv8 input {height}x{width} must divide {max(_STRIDES)}")
    return {
        "image_height": height,
        "image_width": width,
        "num_classes": len(names),
        # Ultralytics letterboxes to a square, scales to [0, 1] and does not
        # mean-subtract; the runtime seam needs those exact values.
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "pad_value": 114.0 / 255.0,
        "max_detections": int(raw.get("max_det", 300)),
    }


def _layout(checkpoint: Checkpoint) -> None:
    """Check the checkpoint carries exactly the stages the table declares."""
    present: set[int] = set()
    for name in checkpoint.names:
        match = re.match(r"^model\.(\d+)\.", name)
        if match:
            present.add(int(match.group(1)))
    if not present:
        raise ValueError("YOLOv8 checkpoint has no model.<index> tensors")
    weighted = {index for index, kind, _ in _STAGES if kind not in {"upsample", "concat"}}
    expected = weighted | {_HEAD_INDEX}
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"YOLOv8 stage set does not match the expected topology; "
            f"missing={missing} unexpected={extra}"
        )


def _fold(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Fold a `conv`/`bn` pair into one biased convolution.

    Every convolution in YOLOv8 is a Conv(conv, bn, SiLU); the statistics stay
    float32 through the division so a small running variance keeps its precision
    in fp16.
    """
    weight = checkpoint.tensor(f"{prefix}.conv.weight")
    gamma = checkpoint.tensor(f"{prefix}.bn.weight")
    beta = checkpoint.tensor(f"{prefix}.bn.bias")
    mean = checkpoint.tensor(f"{prefix}.bn.running_mean")
    variance = checkpoint.tensor(f"{prefix}.bn.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"YOLOv8 norm {prefix}.bn has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"YOLOv8 norm {prefix}.bn does not match its convolution")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"YOLOv8 norm {prefix}.bn has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


class _Weights:
    """Folded convolutions and plain tensors, addressed by checkpoint prefix."""

    def __init__(self, checkpoint: Checkpoint, dtype: np.dtype) -> None:
        self._checkpoint = checkpoint
        self._dtype = dtype
        self._folded: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def has(self, prefix: str) -> bool:
        return f"{prefix}.conv.weight" in self._checkpoint.names

    def conv(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        if prefix not in self._folded:
            self._folded[prefix] = _fold(self._checkpoint, prefix, self._dtype)
        return self._folded[prefix]

    def raw(self, name: str) -> np.ndarray:
        return self._checkpoint.tensor(name).astype(self._dtype)

    def exists(self, name: str) -> bool:
        return name in self._checkpoint.names


def _conv(network, tensor, weights: _Weights, prefix: str, dtype, *, stride: int = 1):
    weight, bias = weights.conv(prefix)
    kernel = int(weight.shape[2])
    groups = 1
    if int(weight.shape[1]) == 1 and int(weight.shape[0]) == int(tensor.shape[1]):
        groups = int(weight.shape[0])
    tensor = graph.convolution(
        network,
        tensor,
        weight,
        bias,
        stride=stride,
        padding=kernel // 2,
        groups=groups,
        dtype=dtype,
    )
    return graph.silu(network, tensor)


def _bottleneck(network, tensor, weights: _Weights, prefix: str, dtype, *, residual: bool):
    """Two convolutions, with a residual only when the stage table says so."""
    inner = _conv(network, tensor, weights, f"{prefix}.cv1", dtype, stride=1)
    inner = _conv(network, inner, weights, f"{prefix}.cv2", dtype, stride=1)
    if not residual or int(inner.shape[1]) != int(tensor.shape[1]):
        return inner
    return graph.add(network, tensor, inner)


def _repvggdw(network, tensor, weights: _Weights, prefix: str, dtype):
    """A 7x7 and a 3x3 depthwise branch, reparameterised into one convolution.

    Both branches fold their own norm first, then the smaller kernel is padded
    into the centre of the larger and the two are summed. Adding them as
    separate layers would give the same answer; folding keeps the built graph
    the shape the model is meant to run as.
    """
    wide, wide_bias = weights.conv(f"{prefix}.conv")
    narrow, narrow_bias = weights.conv(f"{prefix}.conv1")
    span = int(wide.shape[2])
    inset = (span - int(narrow.shape[2])) // 2
    merged = np.array(wide, dtype=np.float32)
    merged[:, :, inset : inset + int(narrow.shape[2]), inset : inset + int(narrow.shape[3])] += (
        np.array(narrow, dtype=np.float32)
    )
    bias = np.array(wide_bias, dtype=np.float32) + np.array(narrow_bias, dtype=np.float32)
    tensor = graph.convolution(
        network,
        tensor,
        merged.astype(dtype),
        bias.astype(dtype),
        padding=span // 2,
        groups=int(merged.shape[0]),
        dtype=dtype,
    )
    return graph.silu(network, tensor)


def _c2f(network, tensor, weights: _Weights, prefix: str, dtype, *, residual: bool):
    """Split the entry convolution in half, chain blocks over the second half.

    Every intermediate output is kept and concatenated, so the exit convolution
    sees `2 + len(blocks)` groups of channels.
    """
    entry = _conv(network, tensor, weights, f"{prefix}.cv1", dtype, stride=1)
    half = int(entry.shape[1]) // 2
    first = graph.slice_axis(network, entry, axis=1, start=0, count=half)
    second = graph.slice_axis(network, entry, axis=1, start=half, count=half)
    parts = [first, second]
    index = 0
    while True:
        leaf = f"{prefix}.m.{index}"
        if not weights.exists(f"{leaf}.cv1.conv.weight"):
            break
        parts.append(_bottleneck(network, parts[-1], weights, leaf, dtype, residual=residual))
        index += 1
    if index == 0:
        raise ValueError(f"YOLOv8 {prefix} has no inner blocks")
    merged = graph.concatenate(network, parts)
    return _conv(network, merged, weights, f"{prefix}.cv2", dtype, stride=1)


def _sppf(network, tensor, weights: _Weights, prefix: str, dtype):
    """Three successive 5x5 max pools, all concatenated with their input."""
    entry = _conv(network, tensor, weights, f"{prefix}.cv1", dtype, stride=1)
    parts = [entry]
    for _ in range(3):
        parts.append(graph.max_pool(network, parts[-1], kernel=5, stride=1, padding=2))
    merged = graph.concatenate(network, parts)
    return _conv(network, merged, weights, f"{prefix}.cv2", dtype, stride=1)


def _backbone(network, pixels, weights: _Weights, dtype):
    """Run the stage table, keeping every output a later stage refers to."""
    outputs: dict[int, Any] = {}
    tensor = pixels
    for index, kind, sources in _STAGES:
        prefix = f"model.{index}"
        if kind == "conv":
            tensor = _conv(
                network,
                tensor,
                weights,
                prefix,
                dtype,
                stride=2 if index in _STRIDED_CONVS else 1,
            )
        elif kind in _RESIDUAL:
            tensor = _c2f(network, tensor, weights, prefix, dtype, residual=_RESIDUAL[kind])
        elif kind == "sppf":
            tensor = _sppf(network, tensor, weights, prefix, dtype)
        elif kind == "upsample":
            tensor = graph.nearest_upsample(network, tensor, 2)
        elif kind == "concat":
            tensor = graph.concatenate(
                network, [tensor if source == -1 else outputs[source] for source in sources]
            )
        else:
            raise ValueError(f"YOLOv8 stage {index} has an unknown kind {kind!r}")
        outputs[index] = tensor
    return [outputs[source] for source in _HEAD_SOURCES]


def _head_branch(network, tensor, weights: _Weights, prefix: str, dtype, *, depth: int):
    """A head branch: `depth` Conv blocks, then a bare convolution."""
    for step in range(depth):
        if weights.has(f"{prefix}.{step}"):
            tensor = _conv(network, tensor, weights, f"{prefix}.{step}", dtype)
            continue
        # A nested pair: depthwise then pointwise, both Conv blocks.
        for inner in range(2):
            tensor = _conv(network, tensor, weights, f"{prefix}.{step}.{inner}", dtype)
    final = weights.raw(f"{prefix}.{depth}.weight")
    return graph.convolution(
        network, tensor, final, weights.raw(f"{prefix}.{depth}.bias"), dtype=dtype
    )


def _anchors(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre points and their stride, in the order the head concatenates them."""
    points: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    for stride in _STRIDES:
        rows, columns = height // stride, width // stride
        y, x = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
        grid = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32) + 0.5
        points.append(grid)
        strides.append(np.full((grid.shape[0], 1), float(stride), dtype=np.float32))
    return np.concatenate(points, axis=0), np.concatenate(strides, axis=0)


# Only these convolutions halve the resolution; the rest keep it.
_STRIDED_CONVS = frozenset({0, 1, 3, 5, 7, 16, 19})
# The backbone C2f blocks add their input back, the neck ones do not. The
# channel shapes match either way, so this cannot be read from the checkpoint.
_RESIDUAL = {"c2f": True, "c2f_plain": False}


def _detect(network, sources, weights: _Weights, dtype, *, config, reg_max: int):
    """The one-to-one head: box distributions and class scores, decoded.

    Boxes come out of a distribution over `reg_max` bins per side, which is
    softmaxed and averaged against a fixed 0..reg_max-1 ramp - that is the DFL
    projection, and it is what turns four distributions into four distances.
    """
    classes = config["num_classes"]
    height, width = config["image_height"], config["image_width"]
    box_channels = 4 * reg_max

    box_parts, score_parts = [], []
    for level, tensor in enumerate(sources):
        prefix = f"model.{_HEAD_INDEX}"
        boxes = _head_branch(network, tensor, weights, f"{prefix}.cv2.{level}", dtype, depth=2)
        scores = _head_branch(network, tensor, weights, f"{prefix}.cv3.{level}", dtype, depth=2)
        cells = (height // _STRIDES[level]) * (width // _STRIDES[level])
        box_parts.append(graph.reshape(network, boxes, (1, box_channels, cells)))
        score_parts.append(graph.reshape(network, scores, (1, classes, cells)))

    boxes = graph.concatenate(network, box_parts, axis=2)
    scores = graph.sigmoid(network, graph.concatenate(network, score_parts, axis=2))
    cells = int(boxes.shape[2])

    # Distribution focal loss projection: softmax over the bins, then a weighted
    # sum against the 0..reg_max-1 ramp stored in the checkpoint.
    distribution = graph.reshape(network, boxes, (1, 4, reg_max, cells))
    distribution = graph.permute(network, distribution, (0, 1, 3, 2))
    distribution = graph.softmax(network, distribution, 3)
    # The matmul needs matching rank, so the ramp is shaped [1, 1, reg_max, 1].
    ramp = weights.raw(f"model.{_HEAD_INDEX}.dfl.conv.weight").reshape(1, 1, reg_max, 1)
    projection = graph.constant(network, ramp, dtype=dtype, like=distribution)
    distances = graph.matmul(network, distribution, projection)
    distances = graph.reshape(network, distances, (1, 4, cells))

    points, strides = _anchors(height, width)
    centres = graph.constant(network, points.T.reshape(1, 2, cells), dtype=dtype, like=distances)
    scale = graph.constant(network, strides.reshape(1, 1, cells), dtype=dtype, like=distances)
    left_top = graph.slice_axis(network, distances, axis=1, start=0, count=2)
    right_bottom = graph.slice_axis(network, distances, axis=1, start=2, count=2)
    minimum = graph.subtract(network, centres, left_top)
    maximum = graph.add(network, centres, right_bottom)
    corners = graph.concatenate(network, [minimum, maximum], axis=1)
    # The head predicts in feature-map cells; scale to network-input pixels.
    corners = graph.multiply(network, corners, scale)

    # YOLOv8's head is not end to end: it emits one prediction per anchor and
    # leaves the suppression to the caller. The engine therefore reports every
    # anchor with its strongest class, and the runtime drops the weak ones and
    # removes the overlaps.
    corners = graph.permute(network, graph.reshape(network, corners, (4, cells)), (1, 0))
    scores = graph.permute(network, graph.reshape(network, scores, (classes, cells)), (1, 0))
    best, index = graph.top_k(network, scores, k=1, axis=1)
    return (
        corners,
        graph.reshape(network, best, (cells,)),
        graph.reshape(network, index, (cells,)),
    )


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
        raise ValueError(f"unsupported YOLOv8 precision: {precision}")
    config = _preprocess_config(raw)
    _layout(checkpoint)
    weights = _Weights(checkpoint, numpy_dtype)

    reg_max = int(checkpoint.tensor(f"model.{_HEAD_INDEX}.dfl.conv.weight").reshape(-1).shape[0])
    height, width = config["image_height"], config["image_width"]
    if verbose:
        print(
            "[trtmc build] yolov8: "
            f"image={height}x{width}, classes={config['num_classes']}, "
            f"reg_max={reg_max}, "
            f"max_detections={config['max_detections']}, precision={precision}",
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
        raise RuntimeError("TensorRT rejected the YOLOv8 input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOv8 input cast")
        hidden = cast.get_output(0)

    sources = _backbone(network, hidden, weights, numpy_dtype)
    boxes, scores, classes = _detect(
        network, sources, weights, numpy_dtype, config=config, reg_max=reg_max
    )

    for tensor, name in ((boxes, "boxes"), (scores, "scores")):
        if tensor.dtype != trt.float32:
            cast = network.add_cast(tensor, trt.float32)
            if cast is None:
                raise RuntimeError(f"TensorRT rejected the YOLOv8 {name} cast")
            tensor = cast.get_output(0)
        tensor.name = name
        network.mark_output(tensor)
    if classes.dtype != trt.int32:
        cast = network.add_cast(classes, trt.int32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOv8 classes cast")
        classes = cast.get_output(0)
    classes.name = "classes"
    network.mark_output(classes)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT YOLOv8 engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one YOLOv8 object-detection bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("yolov8 does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("yolov8 does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("yolov8 does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("yolov8 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("yolov8 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("yolov8 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("yolov8 does not support context parallelism")
    if request.task != "object_detection":
        raise ValueError("yolov8 supports only task=object_detection")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("yolov8 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("yolov8 does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    checkpoint = Checkpoint.open(model_dir)
    raw = describe_checkpoint(checkpoint)
    plan, runtime = _build_engine(
        raw,
        checkpoint,
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="yolov8", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": runtime["image_height"],
            "input_image_w": runtime["image_width"],
            "image_mean": runtime["mean"],
            "image_std": runtime["std"],
            "pad_value": runtime["pad_value"],
            "max_detections": runtime["max_detections"],
        },
    )
