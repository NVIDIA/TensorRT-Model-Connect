# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for YOLOv5 detectors.

YOLOv5 predicts against anchor boxes: each of the three levels emits three
predictions per cell, and each prediction carries an objectness alongside the
class scores. That is the difference from the anchor-free YOLO generations -
the box comes out as an offset and a scale relative to a stored anchor rather
than as a distance from a cell centre, and the reported score is the product of
objectness and class probability. The head leaves overlaps in place, so the
runtime suppresses them.

The module graph, the per-stage channel widths and the block wiring are all
read from the checkpoint. What cannot be read from weights is which block type
each stage is, because several share a leaf layout, so the stage table below is
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


# YOLOv5 resets every norm to this epsilon after building the model. The value
# is pickled with the module but is not part of the state dict, so it is stated
# here. Using the PyTorch default of 1e-5 still builds and still detects, with
# every box slightly wrong and weak detections appearing that should not.
_BATCH_NORM_EPSILON = 1e-3

# The number of anchor boxes each level predicts against. Every published
# YOLOv5 release uses three, and the head convolution width confirms it.
_ANCHORS_PER_LEVEL = 3
# Objectness plus the four box values sit ahead of the class scores.
_BOX_VALUES = 4
_OBJECTNESS_VALUES = 1

# Stage table for the YOLOv5 topology: (index, kind, inputs).
# An input of -1 means the previous stage. Blocks share leaf names, so the kind
# cannot be recovered from the checkpoint; each entry is checked against the
# tensors it requires.
_STAGES: tuple[tuple[int, str, tuple[int, ...]], ...] = (
    (0, "conv", (-1,)),
    (1, "conv", (-1,)),
    (2, "c3", (-1,)),
    (3, "conv", (-1,)),
    (4, "c3", (-1,)),
    (5, "conv", (-1,)),
    (6, "c3", (-1,)),
    (7, "conv", (-1,)),
    (8, "c3", (-1,)),
    (9, "sppf", (-1,)),
    (10, "conv", (-1,)),
    (11, "upsample", (-1,)),
    (12, "concat", (-1, 6)),
    (13, "c3_plain", (-1,)),
    (14, "conv", (-1,)),
    (15, "upsample", (-1,)),
    (16, "concat", (-1, 4)),
    (17, "c3_plain", (-1,)),
    (18, "conv", (-1,)),
    (19, "concat", (-1, 14)),
    (20, "c3_plain", (-1,)),
    (21, "conv", (-1,)),
    (22, "concat", (-1, 10)),
    (23, "c3_plain", (-1,)),
)

_HEAD_INDEX = 24
_HEAD_SOURCES = (17, 20, 23)

# Only these convolutions halve the resolution; the rest keep it.
_STRIDED_CONVS = frozenset({0, 1, 3, 5, 7, 18, 21})
# The stem is the one convolution whose padding is not half its kernel: it uses
# a 6x6 kernel with a padding of 2, which is what makes the first stage halve
# the resolution exactly. Half of six would pad one column too many.
_STEM_PADDING = {0: 2}
# The backbone C3 blocks add their input back inside their bottlenecks, the
# neck ones do not. The channel shapes match either way, so this cannot be read
# from the checkpoint; it is taken from the published configuration, where
# every neck C3 is built with shortcut=False.
_RESIDUAL = {"c3": True, "c3_plain": False}


def describe_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    """The build configuration, taken from the archive itself.

    A YOLOv5 release carries no config.json, so the class names, the strides
    and the anchor boxes come out of the archive and the rest are the values
    the reference predicts with.
    """
    if not checkpoint.class_names:
        raise ValueError("YOLOv5 archive names no classes")
    total = max(checkpoint.strides)
    if checkpoint.image_size % total:
        raise ValueError(
            f"YOLOv5 input {checkpoint.image_size} must be divisible by {total}, "
            "the total stride of the backbone"
        )
    return {
        "names": {index: name for index, name in enumerate(checkpoint.class_names)},
        "imgsz": checkpoint.image_size,
        "strides": list(checkpoint.strides),
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
        raise ValueError("YOLOv5 input size must be positive")
    strides = tuple(int(value) for value in raw["strides"])
    if len(strides) != len(_HEAD_SOURCES):
        raise ValueError(f"YOLOv5 expects {len(_HEAD_SOURCES)} detection levels, got {strides}")
    if height % max(strides) or width % max(strides):
        raise ValueError(f"YOLOv5 input {height}x{width} must divide {max(strides)}")
    return {
        "image_height": height,
        "image_width": width,
        "num_classes": len(names),
        "strides": strides,
        # YOLOv5 letterboxes to a square, scales to [0, 1] and does not
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
        raise ValueError("YOLOv5 checkpoint has no model.<index> tensors")
    weighted = {index for index, kind, _ in _STAGES if kind not in {"upsample", "concat"}}
    expected = weighted | {_HEAD_INDEX}
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"YOLOv5 stage set does not match the expected topology; "
            f"missing={missing} unexpected={extra}"
        )


def _fold(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Fold a `conv`/`bn` pair into one biased convolution.

    Every convolution in YOLOv5 is a Conv(conv, bn, SiLU); the statistics stay
    float32 through the division so a small running variance keeps its
    precision in fp16.
    """
    weight = checkpoint.tensor(f"{prefix}.conv.weight")
    gamma = checkpoint.tensor(f"{prefix}.bn.weight")
    beta = checkpoint.tensor(f"{prefix}.bn.bias")
    mean = checkpoint.tensor(f"{prefix}.bn.running_mean")
    variance = checkpoint.tensor(f"{prefix}.bn.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"YOLOv5 norm {prefix}.bn has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"YOLOv5 norm {prefix}.bn does not match its convolution")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"YOLOv5 norm {prefix}.bn has a non-positive running variance")
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


def _conv(
    network,
    tensor,
    weights: _Weights,
    prefix: str,
    dtype,
    *,
    stride: int = 1,
    padding: int | None = None,
):
    weight, bias = weights.conv(prefix)
    kernel = int(weight.shape[2])
    tensor = graph.convolution(
        network,
        tensor,
        weight,
        bias,
        stride=stride,
        padding=kernel // 2 if padding is None else padding,
        dtype=dtype,
    )
    return graph.silu(network, tensor)


def _bottleneck(network, tensor, weights: _Weights, prefix: str, dtype, *, residual: bool):
    """Two convolutions, with a residual only when the stage table says so."""
    inner = _conv(network, tensor, weights, f"{prefix}.cv1", dtype)
    inner = _conv(network, inner, weights, f"{prefix}.cv2", dtype)
    if not residual or int(inner.shape[1]) != int(tensor.shape[1]):
        return inner
    return graph.add(network, tensor, inner)


def _c3(network, tensor, weights: _Weights, prefix: str, dtype, *, residual: bool):
    """A CSP block: two entry convolutions, a bottleneck chain on one of them.

    The chain runs on the `cv1` side only; `cv2` carries its half straight to
    the concatenation, and `cv3` is the exit.
    """
    left = _conv(network, tensor, weights, f"{prefix}.cv1", dtype)
    right = _conv(network, tensor, weights, f"{prefix}.cv2", dtype)
    index = 0
    while weights.exists(f"{prefix}.m.{index}.cv1.conv.weight"):
        left = _bottleneck(network, left, weights, f"{prefix}.m.{index}", dtype, residual=residual)
        index += 1
    if index == 0:
        raise ValueError(f"YOLOv5 {prefix} has no inner bottlenecks")
    merged = graph.concatenate(network, [left, right])
    return _conv(network, merged, weights, f"{prefix}.cv3", dtype)


def _sppf(network, tensor, weights: _Weights, prefix: str, dtype):
    """Three successive 5x5 max pools, all concatenated with their input."""
    entry = _conv(network, tensor, weights, f"{prefix}.cv1", dtype)
    parts = [entry]
    for _ in range(3):
        parts.append(graph.max_pool(network, parts[-1], kernel=5, stride=1, padding=2))
    merged = graph.concatenate(network, parts)
    return _conv(network, merged, weights, f"{prefix}.cv2", dtype)


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
                padding=_STEM_PADDING.get(index),
            )
        elif kind in _RESIDUAL:
            tensor = _c3(network, tensor, weights, prefix, dtype, residual=_RESIDUAL[kind])
        elif kind == "sppf":
            tensor = _sppf(network, tensor, weights, prefix, dtype)
        elif kind == "upsample":
            tensor = graph.nearest_upsample(network, tensor, 2)
        elif kind == "concat":
            tensor = graph.concatenate(
                network, [tensor if source == -1 else outputs[source] for source in sources]
            )
        else:
            raise ValueError(f"YOLOv5 stage {index} has an unknown kind {kind!r}")
        outputs[index] = tensor
    return [outputs[source] for source in _HEAD_SOURCES]


def _cell_grid(rows: int, columns: int) -> np.ndarray:
    """Cell coordinates as [x, y], in the order the head flattens them."""
    y, x = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    return np.stack([x.ravel(), y.ravel()], axis=0).astype(np.float32)


def _detect(network, sources, weights: _Weights, dtype, *, config, anchors: np.ndarray):
    """The anchor head: one 1x1 convolution per level, then the YOLOv5 decode.

    Every predicted value is passed through a sigmoid first. The centre is then
    an offset of up to half a cell either side of its own cell, and the size is
    a squared multiple of the anchor capped at four times it. Those two
    reparameterisations are what the `* 2 - 0.5` and the `(* 2) ** 2` are;
    getting either wrong still produces boxes, just in the wrong places.
    """
    classes = config["num_classes"]
    height, width = config["image_height"], config["image_width"]
    strides = config["strides"]
    values = _BOX_VALUES + _OBJECTNESS_VALUES + classes

    corner_parts, score_parts = [], []
    for level, tensor in enumerate(sources):
        stride = strides[level]
        rows, columns = height // stride, width // stride
        cells = rows * columns
        weight = weights.raw(f"model.{_HEAD_INDEX}.m.{level}.weight")
        bias = weights.raw(f"model.{_HEAD_INDEX}.m.{level}.bias")
        if int(weight.shape[0]) != _ANCHORS_PER_LEVEL * values:
            raise ValueError(
                f"YOLOv5 head level {level} predicts {int(weight.shape[0])} channels, "
                f"expected {_ANCHORS_PER_LEVEL * values}"
            )
        raw = graph.convolution(network, tensor, weight, bias, dtype=dtype)
        # The batch axis is dropped here: TensorRT picks worse slice tactics on
        # a rank-four tensor whose leading axis is one, and logs a build error
        # for each of them. Every value below is per anchor, per cell.
        raw = graph.reshape(network, raw, (_ANCHORS_PER_LEVEL, values, cells))
        raw = graph.sigmoid(network, raw)

        centre = graph.slice_axis(network, raw, axis=1, start=0, count=2)
        size = graph.slice_axis(network, raw, axis=1, start=2, count=2)
        objectness = graph.slice_axis(network, raw, axis=1, start=4, count=1)
        scores = graph.slice_axis(network, raw, axis=1, start=5, count=classes)

        # centre = (sigmoid * 2 - 0.5 + cell) * stride
        grid = _cell_grid(rows, columns).reshape(1, 2, cells) - 0.5
        offset = graph.constant(network, grid, dtype=dtype, like=centre)
        centre = graph.add(network, graph.scale(network, centre, 2.0, dtype=dtype), offset)
        centre = graph.scale(network, centre, float(stride), dtype=dtype)

        # size = (sigmoid * 2) ** 2 * anchor, with the anchor in input pixels.
        doubled = graph.scale(network, size, 2.0, dtype=dtype)
        squared = graph.multiply(network, doubled, doubled)
        boxes = anchors[level].reshape(_ANCHORS_PER_LEVEL, 2, 1) * float(stride)
        size = graph.multiply(
            network, squared, graph.constant(network, boxes, dtype=dtype, like=squared)
        )

        half = graph.scale(network, size, 0.5, dtype=dtype)
        corners = graph.concatenate(
            network,
            [graph.subtract(network, centre, half), graph.add(network, centre, half)],
            axis=1,
        )
        # Anchor first then cell, which is the order the reference flattens in.
        corners = graph.permute(network, corners, (0, 2, 1))
        corner_parts.append(
            graph.reshape(network, corners, (_ANCHORS_PER_LEVEL * cells, _BOX_VALUES))
        )
        # The reported score is objectness times class probability. Both sides
        # are flattened before the multiply: when one slice broadcasts against
        # another on the axis they were sliced on, TensorRT logs a build error
        # for every tactic it skips. The arithmetic is the same either way.
        flat = graph.reshape(
            network,
            graph.permute(network, scores, (0, 2, 1)),
            (_ANCHORS_PER_LEVEL * cells, classes),
        )
        confidence = graph.reshape(
            network,
            graph.permute(network, objectness, (0, 2, 1)),
            (_ANCHORS_PER_LEVEL * cells, 1),
        )
        score_parts.append(graph.multiply(network, flat, confidence))

    corners = graph.concatenate(network, corner_parts, axis=0)
    scores = graph.concatenate(network, score_parts, axis=0)
    total = int(corners.shape[0])
    # YOLOv5's head emits one prediction per anchor and leaves the suppression
    # to the caller. The engine therefore reports every anchor with its
    # strongest class, and the runtime drops the weak ones and removes the
    # overlaps.
    best, index = graph.top_k(network, scores, k=1, axis=1)
    return corners, graph.reshape(network, best, (total,)), graph.reshape(network, index, (total,))


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
        raise ValueError(f"unsupported YOLOv5 precision: {precision}")
    config = _preprocess_config(raw)
    _layout(checkpoint)
    weights = _Weights(checkpoint, numpy_dtype)

    # The archive stores the anchors already divided by their level's stride,
    # so they are in cells here and are scaled back to pixels in the head.
    anchors = checkpoint.tensor(f"model.{_HEAD_INDEX}.anchors")
    expected = (len(_HEAD_SOURCES), _ANCHORS_PER_LEVEL, 2)
    if anchors.shape != expected:
        raise ValueError(f"YOLOv5 anchors have shape {anchors.shape}, expected {expected}")
    if np.any(anchors <= 0.0):
        raise ValueError("YOLOv5 anchors must all be positive")

    height, width = config["image_height"], config["image_width"]
    if verbose:
        print(
            "[trtmc build] yolov5: "
            f"image={height}x{width}, classes={config['num_classes']}, "
            f"strides={list(config['strides'])}, anchors={_ANCHORS_PER_LEVEL}, "
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
        raise RuntimeError("TensorRT rejected the YOLOv5 input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOv5 input cast")
        hidden = cast.get_output(0)

    sources = _backbone(network, hidden, weights, numpy_dtype)
    boxes, scores, classes = _detect(
        network, sources, weights, numpy_dtype, config=config, anchors=anchors
    )

    for tensor, name in ((boxes, "boxes"), (scores, "scores")):
        if tensor.dtype != trt.float32:
            cast = network.add_cast(tensor, trt.float32)
            if cast is None:
                raise RuntimeError(f"TensorRT rejected the YOLOv5 {name} cast")
            tensor = cast.get_output(0)
        tensor.name = name
        network.mark_output(tensor)
    if classes.dtype != trt.int32:
        cast = network.add_cast(classes, trt.int32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOv5 classes cast")
        classes = cast.get_output(0)
    classes.name = "classes"
    network.mark_output(classes)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT YOLOv5 engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one YOLOv5 object-detection bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("yolov5 does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("yolov5 does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("yolov5 does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("yolov5 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("yolov5 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("yolov5 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("yolov5 does not support context parallelism")
    if request.task != "object_detection":
        raise ValueError("yolov5 supports only task=object_detection")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("yolov5 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("yolov5 does not support mixed-precision layers")
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
    writer.set_header(family="yolov5", task=request.task, backend=request.backend)
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
