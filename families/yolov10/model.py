# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for YOLOv10 detectors.

YOLOv10 is end to end: the one-to-one head it uses at inference emits one box
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


# Ultralytics uses a looser batch-norm epsilon than the PyTorch default. Using
# 1e-5 here still builds and still detects, with every box slightly wrong.
_BATCH_NORM_EPSILON = 1e-3

# Stage table for the YOLOv10 topology: (index, kind, inputs).
# `c2f` keeps the residual inside its inner blocks and `c2f_plain` drops it.
# That flag is an architecture property, not a shape: the channel counts match
# either way, so inferring it from the tensors silently adds the wrong residual.
# It applies only to plain bottleneck inner blocks. Whether a block is a
# bottleneck or a CIB is readable from the keys and decided per block, and a CIB
# always carries its residual - that is what lets one table cover every YOLOv10
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
    (5, "scdown", (-1,)),
    (6, "c2f", (-1,)),
    (7, "scdown", (-1,)),
    (8, "c2f", (-1,)),
    (9, "sppf", (-1,)),
    (10, "psa", (-1,)),
    (11, "upsample", (-1,)),
    (12, "concat", (-1, 6)),
    (13, "c2f_plain", (-1,)),
    (14, "upsample", (-1,)),
    (15, "concat", (-1, 4)),
    (16, "c2f_plain", (-1,)),
    (17, "conv", (-1,)),
    (18, "concat", (-1, 13)),
    (19, "c2f_plain", (-1,)),
    (20, "scdown", (-1,)),
    (21, "concat", (-1, 10)),
    (22, "c2f", (-1,)),
)

# PSA fixes its head size rather than its head count, so the number of heads
# follows from the attended width. Nothing in the checkpoint records this.
_PSA_HEAD_DIM = 64

_HEAD_INDEX = 23
_HEAD_SOURCES = (16, 19, 22)
_STRIDES = (8, 16, 32)


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"YOLOv10 model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YOLOv10 config.json must contain one object")
    model = value.get("model")
    if not isinstance(model, str) or not model.startswith("yolov10"):
        raise ValueError(f"unsupported YOLOv10 model identity: {model!r}")
    if value.get("task") != "detect":
        raise ValueError(f"unsupported YOLOv10 task: {value.get('task')!r}")
    names = value.get("names")
    if not isinstance(names, dict) or not names:
        raise ValueError("YOLOv10 config.json must name its classes")
    return value


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    names = raw["names"]
    size = raw.get("imgsz", 640)
    if isinstance(size, list):
        height, width = int(size[-2]), int(size[-1])
    else:
        height = width = int(size)
    if height <= 0 or width <= 0:
        raise ValueError("YOLOv10 input size must be positive")
    if height % max(_STRIDES) or width % max(_STRIDES):
        raise ValueError(f"YOLOv10 input {height}x{width} must divide {max(_STRIDES)}")
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
        match = re.match(r"^model\.model\.(\d+)\.", name)
        if match:
            present.add(int(match.group(1)))
    if not present:
        raise ValueError("YOLOv10 checkpoint has no model.model.<index> tensors")
    weighted = {index for index, kind, _ in _STAGES if kind not in {"upsample", "concat"}}
    expected = weighted | {_HEAD_INDEX}
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"YOLOv10 stage set does not match the expected topology; "
            f"missing={missing} unexpected={extra}"
        )


def _fold(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Fold a `conv`/`bn` pair into one biased convolution.

    Every convolution in YOLOv10 is a Conv(conv, bn, SiLU); the statistics stay
    float32 through the division so a small running variance keeps its precision
    in fp16.
    """
    weight = checkpoint.tensor(f"{prefix}.conv.weight")
    gamma = checkpoint.tensor(f"{prefix}.bn.weight")
    beta = checkpoint.tensor(f"{prefix}.bn.bias")
    mean = checkpoint.tensor(f"{prefix}.bn.running_mean")
    variance = checkpoint.tensor(f"{prefix}.bn.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"YOLOv10 norm {prefix}.bn has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"YOLOv10 norm {prefix}.bn does not match its convolution")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"YOLOv10 norm {prefix}.bn has a non-positive running variance")
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
        network, tensor, weight, bias,
        stride=stride, padding=kernel // 2, groups=groups, dtype=dtype,
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
    merged[:, :, inset:inset + int(narrow.shape[2]), inset:inset + int(narrow.shape[3])] += (
        np.array(narrow, dtype=np.float32)
    )
    bias = np.array(wide_bias, dtype=np.float32) + np.array(narrow_bias, dtype=np.float32)
    tensor = graph.convolution(
        network, tensor, merged.astype(dtype), bias.astype(dtype),
        padding=span // 2, groups=int(merged.shape[0]), dtype=dtype,
    )
    return graph.silu(network, tensor)


def _cib(network, tensor, weights: _Weights, prefix: str, dtype, *, residual: bool):
    """The compact inverted block: a five-step chain with a residual.

    The middle step is a reparameterisable depthwise pair, not a plain Conv, so
    it is dispatched on the keys the checkpoint actually carries.
    """
    inner = tensor
    for step in range(5):
        leaf = f"{prefix}.cv1.{step}"
        if weights.exists(f"{leaf}.conv1.conv.weight"):
            inner = _repvggdw(network, inner, weights, leaf, dtype)
        else:
            inner = _conv(network, inner, weights, leaf, dtype)
    if not residual or int(inner.shape[1]) != int(tensor.shape[1]):
        return inner
    return graph.add(network, tensor, inner)


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
        # A CIB stores its chain under cv1.0..4; a bottleneck has cv1 directly.
        if weights.exists(f"{leaf}.cv1.0.conv.weight"):
            parts.append(_cib(network, parts[-1], weights, leaf, dtype, residual=True))
        elif weights.exists(f"{leaf}.cv1.conv.weight"):
            parts.append(
                _bottleneck(network, parts[-1], weights, leaf, dtype, residual=residual)
            )
        else:
            break
        index += 1
    if index == 0:
        raise ValueError(f"YOLOv10 {prefix} has no inner blocks")
    merged = graph.concatenate(network, parts)
    return _conv(network, merged, weights, f"{prefix}.cv2", dtype, stride=1)


def _scdown(network, tensor, weights: _Weights, prefix: str, dtype):
    """A pointwise convolution, then a strided depthwise one that halves size."""
    tensor = _conv(network, tensor, weights, f"{prefix}.cv1", dtype, stride=1)
    weight, bias = weights.conv(f"{prefix}.cv2")
    tensor = graph.convolution(
        network, tensor, weight, bias,
        stride=2, padding=int(weight.shape[2]) // 2, groups=int(weight.shape[0]), dtype=dtype,
    )
    return tensor


def _sppf(network, tensor, weights: _Weights, prefix: str, dtype):
    """Three successive 5x5 max pools, all concatenated with their input."""
    entry = _conv(network, tensor, weights, f"{prefix}.cv1", dtype, stride=1)
    parts = [entry]
    for _ in range(3):
        parts.append(graph.max_pool(network, parts[-1], kernel=5, stride=1, padding=2))
    merged = graph.concatenate(network, parts)
    return _conv(network, merged, weights, f"{prefix}.cv2", dtype, stride=1)


def _attention(network, tensor, weights: _Weights, prefix: str, dtype, *, heads: int):
    """Spatial self-attention over the flattened feature map, plus a positional
    depthwise convolution applied to the values."""
    channels = int(tensor.shape[1])
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    tokens = height * width
    head_dim = channels // heads
    key_dim = head_dim // 2

    qkv, qkv_bias = weights.conv(f"{prefix}.qkv")
    fused = graph.convolution(network, tensor, qkv, qkv_bias, dtype=dtype)
    fused = graph.reshape(network, fused, (1, heads, key_dim * 2 + head_dim, tokens))
    query = graph.slice_axis(network, fused, axis=2, start=0, count=key_dim)
    key = graph.slice_axis(network, fused, axis=2, start=key_dim, count=key_dim)
    value = graph.slice_axis(network, fused, axis=2, start=key_dim * 2, count=head_dim)

    scores = graph.matmul(network, query, key, transpose_left=True)
    scores = graph.scale(network, scores, float(key_dim) ** -0.5, dtype=dtype)
    scores = graph.softmax(network, scores, 3)
    context = graph.matmul(network, value, scores, transpose_right=True)
    context = graph.reshape(network, context, (1, channels, height, width))

    spatial = graph.reshape(network, value, (1, channels, height, width))
    position, position_bias = weights.conv(f"{prefix}.pe")
    spatial = graph.convolution(
        network, spatial, position, position_bias,
        padding=int(position.shape[2]) // 2, groups=channels, dtype=dtype,
    )
    merged = graph.add(network, context, spatial)
    projection, projection_bias = weights.conv(f"{prefix}.proj")
    return graph.convolution(network, merged, projection, projection_bias, dtype=dtype)


def _psa(network, tensor, weights: _Weights, prefix: str, dtype, *, heads: int):
    """Attention applied to half the channels, the other half passed through."""
    entry = _conv(network, tensor, weights, f"{prefix}.cv1", dtype, stride=1)
    half = int(entry.shape[1]) // 2
    passthrough = graph.slice_axis(network, entry, axis=1, start=0, count=half)
    attended = graph.slice_axis(network, entry, axis=1, start=half, count=half)
    attended = graph.add(
        network, attended, _attention(network, attended, weights, f"{prefix}.attn", dtype, heads=heads)
    )
    feed = _conv(network, attended, weights, f"{prefix}.ffn.0", dtype, stride=1)
    weight, bias = weights.conv(f"{prefix}.ffn.1")
    feed = graph.convolution(network, feed, weight, bias, dtype=dtype)
    attended = graph.add(network, attended, feed)
    merged = graph.concatenate(network, [passthrough, attended])
    return _conv(network, merged, weights, f"{prefix}.cv2", dtype, stride=1)


# Conv stages that halve the resolution. SCDown stages always halve, inside cv2.
_STRIDED_CONVS = frozenset({0, 1, 3, 17})

_RESIDUAL = {"c2f": True, "c2f_plain": False}


def _backbone(network, pixels, weights: _Weights, dtype, *, heads: int):
    """Run the stage table, keeping every output a later stage refers to."""
    outputs: dict[int, Any] = {}
    tensor = pixels
    for index, kind, sources in _STAGES:
        prefix = f"model.model.{index}"
        if kind == "conv":
            tensor = _conv(
                network, tensor, weights, prefix, dtype,
                stride=2 if index in _STRIDED_CONVS else 1,
            )
        elif kind in _RESIDUAL:
            tensor = _c2f(
                network, tensor, weights, prefix, dtype, residual=_RESIDUAL[kind]
            )
        elif kind == "scdown":
            tensor = _scdown(network, tensor, weights, prefix, dtype)
        elif kind == "sppf":
            tensor = _sppf(network, tensor, weights, prefix, dtype)
        elif kind == "psa":
            tensor = _psa(network, tensor, weights, prefix, dtype, heads=heads)
        elif kind == "upsample":
            tensor = graph.nearest_upsample(network, tensor, 2)
        elif kind == "concat":
            tensor = graph.concatenate(
                network, [tensor if source == -1 else outputs[source] for source in sources]
            )
        else:
            raise ValueError(f"YOLOv10 stage {index} has an unknown kind {kind!r}")
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
        prefix = f"model.model.{_HEAD_INDEX}"
        boxes = _head_branch(
            network, tensor, weights, f"{prefix}.one2one_cv2.{level}", dtype, depth=2
        )
        scores = _head_branch(
            network, tensor, weights, f"{prefix}.one2one_cv3.{level}", dtype, depth=2
        )
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
    ramp = weights.raw(f"model.model.{_HEAD_INDEX}.dfl.conv.weight").reshape(1, 1, reg_max, 1)
    projection = graph.constant(network, ramp, dtype=dtype, like=distribution)
    distances = graph.matmul(network, distribution, projection)
    distances = graph.reshape(network, distances, (1, 4, cells))

    points, strides = _anchors(height, width)
    centres = graph.constant(
        network, points.T.reshape(1, 2, cells), dtype=dtype, like=distances
    )
    scale = graph.constant(
        network, strides.reshape(1, 1, cells), dtype=dtype, like=distances
    )
    left_top = graph.slice_axis(network, distances, axis=1, start=0, count=2)
    right_bottom = graph.slice_axis(network, distances, axis=1, start=2, count=2)
    minimum = graph.subtract(network, centres, left_top)
    maximum = graph.add(network, centres, right_bottom)
    corners = graph.concatenate(network, [minimum, maximum], axis=1)
    # The head predicts in feature-map cells; scale to network-input pixels.
    corners = graph.multiply(network, corners, scale)

    # End to end, matching the reference postprocess exactly. It is two stages,
    # and the flat index is anchor-major: `anchor * classes + class`. A single
    # stage, or a class-major index, still returns plausible boxes with the
    # wrong classes attached.
    keep = min(int(config["max_detections"]), cells)
    corners = graph.permute(network, graph.reshape(network, corners, (4, cells)), (1, 0))
    scores = graph.permute(network, graph.reshape(network, scores, (classes, cells)), (1, 0))

    # Stage one: the best `keep` anchors, ranked by their strongest class.
    # TensorRT's top-k needs at least two dimensions, so the ranked axis keeps
    # a trailing one and the indices are flattened for the gather.
    best = graph.reduce_max(network, scores, 1, keep_dims=True)
    _, ranked = graph.top_k(network, best, k=keep, axis=0)
    anchor_index = graph.reshape(network, ranked, (keep,))
    kept_scores = graph.gather(network, scores, anchor_index, axis=0)
    kept_boxes = graph.gather(network, corners, anchor_index, axis=0)

    # Stage two: the best `keep` class scores among those anchors.
    flat = graph.reshape(network, kept_scores, (1, keep * classes))
    top_scores, flat_ranked = graph.top_k(network, flat, k=keep, axis=1)
    flat_index = graph.reshape(network, flat_ranked, (keep,))
    divisor = graph.constant(
        network, np.full((keep,), classes, dtype=np.int32), dtype=np.int32, like=flat_index
    )
    selected = graph.floor_divide(network, flat_index, divisor)
    class_index = graph.subtract(
        network, flat_index, graph.multiply(network, selected, divisor)
    )
    return (
        graph.gather(network, kept_boxes, selected, axis=0),
        graph.reshape(network, top_scores, (keep,)),
        class_index,
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
        raise ValueError(f"unsupported YOLOv10 precision: {precision}")
    config = _preprocess_config(raw)
    _layout(checkpoint)
    weights = _Weights(checkpoint, numpy_dtype)

    reg_max = int(
        checkpoint.tensor(f"model.model.{_HEAD_INDEX}.dfl.conv.weight").reshape(-1).shape[0]
    )
    # PSA attends over half its entry channels, split into heads of _PSA_HEAD_DIM.
    attended = int(checkpoint.tensor("model.model.10.cv1.conv.weight").shape[0]) // 2
    qkv_width = int(checkpoint.tensor("model.model.10.attn.qkv.conv.weight").shape[0])
    # Each head emits head_dim values plus two key_dim halves, and key_dim is
    # head_dim / 2, so qkv is exactly twice the attended width. Checking that
    # catches a wrong head size, which otherwise still builds.
    if qkv_width != 2 * attended:
        raise ValueError(
            f"YOLOv10 PSA qkv width {qkv_width} does not match twice its attended "
            f"width {attended}"
        )
    heads = max(1, attended // _PSA_HEAD_DIM)

    height, width = config["image_height"], config["image_width"]
    if verbose:
        print(
            "[trtmc build] yolov10: "
            f"image={height}x{width}, classes={config['num_classes']}, "
            f"reg_max={reg_max}, psa_heads={heads}, "
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
        raise RuntimeError("TensorRT rejected the YOLOv10 input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOv10 input cast")
        hidden = cast.get_output(0)

    sources = _backbone(network, hidden, weights, numpy_dtype, heads=heads)
    boxes, scores, classes = _detect(
        network, sources, weights, numpy_dtype, config=config, reg_max=reg_max
    )

    for tensor, name in ((boxes, "boxes"), (scores, "scores")):
        if tensor.dtype != trt.float32:
            cast = network.add_cast(tensor, trt.float32)
            if cast is None:
                raise RuntimeError(f"TensorRT rejected the YOLOv10 {name} cast")
            tensor = cast.get_output(0)
        tensor.name = name
        network.mark_output(tensor)
    if classes.dtype != trt.int32:
        cast = network.add_cast(classes, trt.int32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOv10 classes cast")
        classes = cast.get_output(0)
    classes.name = "classes"
    network.mark_output(classes)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT YOLOv10 engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one YOLOv10 object-detection bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("yolov10 does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("yolov10 does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("yolov10 does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("yolov10 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("yolov10 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("yolov10 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("yolov10 does not support context parallelism")
    if request.task != "object_detection":
        raise ValueError("yolov10 supports only task=object_detection")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("yolov10 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("yolov10 does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="yolov10", task=request.task, backend=request.backend)
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
