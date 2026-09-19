# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YOLOX: CSPDarknet/PAFPN or Darknet/FPN and a decoupled anchor-free head.

The topology follows Megvii-BaseDetection/YOLOX at
6ddff4824372906469a7fae2dc3206c7aa4bbaee, exps/default/.
TensorRT owns lowering and execution; this family specifies the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph
from .checkpoint import Checkpoint

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


# Exp.get_model overrides the PyTorch default epsilon before loading weights.
_BATCH_NORM_EPSILON = 1e-3
_NUM_CLASSES = 80
_STRIDES = (8, 16, 32)


def _fold(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    weight = checkpoint.tensor(f"{prefix}.conv.weight")
    gamma = checkpoint.tensor(f"{prefix}.bn.weight")
    beta = checkpoint.tensor(f"{prefix}.bn.bias")
    mean = checkpoint.tensor(f"{prefix}.bn.running_mean")
    variance = checkpoint.tensor(f"{prefix}.bn.running_var")
    if weight.ndim != 4 or any(
        v.shape != (weight.shape[0],) for v in (gamma, beta, mean, variance)
    ):
        raise ValueError(f"YOLOX convolution/BatchNorm shape mismatch: {prefix}")
    if np.any(variance < 0):
        raise ValueError(f"YOLOX BatchNorm variance must be non-negative: {prefix}")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


class _Weights:
    """Folded convolutions and plain tensors, addressed by checkpoint prefix."""

    def __init__(self, checkpoint: Checkpoint, dtype: np.dtype) -> None:
        self._checkpoint = checkpoint
        self._dtype = dtype
        self._folded: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def conv(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        if prefix not in self._folded:
            self._folded[prefix] = _fold(self._checkpoint, prefix, self._dtype)
        return self._folded[prefix]

    def raw(self, name: str) -> np.ndarray:
        return self._checkpoint.tensor(name).astype(self._dtype)

    def exists(self, name: str) -> bool:
        return name in self._checkpoint.state


def _conv(network, tensor, weights: _Weights, prefix: str, dtype, *, stride: int = 1):
    if weights.exists(f"{prefix}.dconv.conv.weight"):
        # Rounding between depthwise and pointwise convolutions amplifies score error.
        if dtype == np.float16:
            cast = network.add_cast(tensor, trt.float32)
            if cast is None:
                raise RuntimeError("TensorRT rejected the YOLOX depthwise input cast")
            tensor = cast.get_output(0)
        tensor = _conv(network, tensor, weights, f"{prefix}.dconv", np.float32, stride=stride)
        tensor = _conv(network, tensor, weights, f"{prefix}.pconv", np.float32)
        if dtype == np.float16:
            cast = network.add_cast(tensor, trt.float16)
            if cast is None:
                raise RuntimeError("TensorRT rejected the YOLOX depthwise output cast")
            tensor = cast.get_output(0)
        return tensor
    weight, bias = weights.conv(prefix)
    groups = int(tensor.shape[1]) if prefix.endswith(".dconv") else 1
    if weight.shape[1] * groups != int(tensor.shape[1]):
        raise ValueError(f"YOLOX input channel mismatch: {prefix}")
    tensor = graph.convolution(
        network,
        tensor,
        weight,
        bias,
        stride=stride,
        padding=weight.shape[2] // 2,
        groups=groups,
        dtype=dtype,
    )
    if weights.exists("backbone.backbone.stem.0.conv.weight"):
        return graph.leaky_relu(network, tensor)
    return graph.silu(network, tensor)


def _csp(network, tensor, weights: _Weights, prefix: str, dtype, *, residual: bool):
    left = _conv(network, tensor, weights, f"{prefix}.conv1", dtype)
    right = _conv(network, tensor, weights, f"{prefix}.conv2", dtype)
    # Preserve the residual path around depthwise bottlenecks in FP32 as well.
    inner_dtype = np.float32 if weights.exists(f"{prefix}.m.0.conv2.dconv.conv.weight") else dtype
    if inner_dtype != dtype:
        cast = network.add_cast(left, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOX bottleneck input cast")
        left = cast.get_output(0)
    index = 0
    while weights.exists(f"{prefix}.m.{index}.conv1.conv.weight"):
        inner = _conv(network, left, weights, f"{prefix}.m.{index}.conv1", inner_dtype)
        inner = _conv(network, inner, weights, f"{prefix}.m.{index}.conv2", inner_dtype)
        left = graph.add(network, left, inner) if residual else inner
        index += 1
    if index == 0:
        raise ValueError(f"YOLOX CSP block has no bottlenecks: {prefix}")
    if inner_dtype != dtype:
        cast = network.add_cast(left, right.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOX bottleneck output cast")
        left = cast.get_output(0)
    return _conv(
        network, graph.concatenate(network, [left, right]), weights, f"{prefix}.conv3", dtype
    )


def _focus(network, tensor):
    # Order is top-left, bottom-left, top-right, bottom-right, not row-major.
    batch, channels, height, width = map(int, tensor.shape)
    parts = []
    for y, x in ((0, 0), (1, 0), (0, 1), (1, 1)):
        layer = network.add_slice(
            tensor, (0, 0, y, x), (batch, channels, height // 2, width // 2), (1, 1, 2, 2)
        )
        if layer is None:
            raise RuntimeError("TensorRT rejected YOLOX Focus")
        parts.append(layer.get_output(0))
    return graph.concatenate(network, parts)


def _spp(network, tensor, weights: _Weights, prefix: str, dtype):
    entry = _conv(network, tensor, weights, f"{prefix}.conv1", dtype)
    # Parallel SPP pools in the order used by the official implementation.
    parts = [entry] + [
        graph.max_pool(network, entry, kernel=k, stride=1, padding=k // 2) for k in (5, 9, 13)
    ]
    return _conv(network, graph.concatenate(network, parts), weights, f"{prefix}.conv2", dtype)


def _darknet(network, pixels, weights: _Weights, dtype):
    prefix = "backbone.backbone"
    tensor = _conv(network, pixels, weights, f"{prefix}.stem.0", np.float32)
    if dtype == np.float16:
        cast = network.add_cast(tensor, trt.float16)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOX Darknet feature cast")
        tensor = cast.get_output(0)
    outputs = []
    for stage in ("stem", "dark2", "dark3", "dark4", "dark5"):
        index = 1 if stage == "stem" else 0
        block = f"{prefix}.{stage}"
        tensor = _conv(network, tensor, weights, f"{block}.{index}", dtype, stride=2)
        index += 1
        while weights.exists(f"{block}.{index}.layer1.conv.weight"):
            inner = _conv(network, tensor, weights, f"{block}.{index}.layer1", dtype)
            inner = _conv(network, inner, weights, f"{block}.{index}.layer2", dtype)
            tensor = graph.add(network, tensor, inner)
            index += 1
        if stage == "dark5":
            tensor = _conv(network, tensor, weights, f"{block}.{index}", dtype)
            tensor = _conv(network, tensor, weights, f"{block}.{index + 1}", dtype)
            tensor = _spp(network, tensor, weights, f"{block}.{index + 2}", dtype)
            tensor = _conv(network, tensor, weights, f"{block}.{index + 3}", dtype)
            tensor = _conv(network, tensor, weights, f"{block}.{index + 4}", dtype)
        if stage in {"dark3", "dark4", "dark5"}:
            outputs.append(tensor)
    return outputs


def _backbone(network, pixels, weights: _Weights, dtype):
    if weights.exists("backbone.backbone.stem.0.conv.weight"):
        return _darknet(network, pixels, weights, dtype)
    prefix = "backbone.backbone"
    # Preserve small color differences in the unnormalized BGR byte input.
    tensor = _conv(network, _focus(network, pixels), weights, f"{prefix}.stem.conv", np.float32)
    if dtype == np.float16:
        cast = network.add_cast(tensor, trt.float16)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOX feature cast")
        tensor = cast.get_output(0)
    outputs = []
    for stage in range(2, 6):
        tensor = _conv(network, tensor, weights, f"{prefix}.dark{stage}.0", dtype, stride=2)
        if stage == 5:
            tensor = _spp(network, tensor, weights, f"{prefix}.dark5.1", dtype)
        tensor = _csp(
            network,
            tensor,
            weights,
            f"{prefix}.dark{stage}.{2 if stage == 5 else 1}",
            dtype,
            residual=stage != 5,
        )
        if stage >= 3:
            outputs.append(tensor)
    return outputs


def _fpn(network, sources, weights: _Weights, dtype):
    dark3, dark4, tensor = sources
    outputs = [tensor]
    for level, source in enumerate((dark4, dark3), start=1):
        tensor = _conv(network, tensor, weights, f"backbone.out{level}_cbl", dtype)
        tensor = graph.concatenate(network, [graph.nearest_upsample(network, tensor, 2), source])
        index = 0
        while weights.exists(f"backbone.out{level}.{index}.conv.weight"):
            tensor = _conv(network, tensor, weights, f"backbone.out{level}.{index}", dtype)
            index += 1
        outputs.append(tensor)
    return tuple(reversed(outputs))


def _neck(network, sources, weights: _Weights, dtype):
    # PAFPN and the head need FP32 to keep score error within 0.01.
    promoted = []
    for source in sources:
        if source.dtype != trt.float32:
            cast = network.add_cast(source, trt.float32)
            if cast is None:
                raise RuntimeError("TensorRT rejected the YOLOX PAFPN feature cast")
            source = cast.get_output(0)
        promoted.append(source)
    if weights.exists("backbone.out1_cbl.conv.weight"):
        return _fpn(network, promoted, weights, dtype)
    dark3, dark4, dark5 = promoted
    lateral = _conv(network, dark5, weights, "backbone.lateral_conv0", dtype)
    merged = graph.concatenate(network, [graph.nearest_upsample(network, lateral, 2), dark4])
    upper = _csp(network, merged, weights, "backbone.C3_p4", dtype, residual=False)
    reduced = _conv(network, upper, weights, "backbone.reduce_conv1", dtype)
    merged = graph.concatenate(network, [graph.nearest_upsample(network, reduced, 2), dark3])
    p3 = _csp(network, merged, weights, "backbone.C3_p3", dtype, residual=False)
    merged = graph.concatenate(
        network, [_conv(network, p3, weights, "backbone.bu_conv2", dtype, stride=2), reduced]
    )
    p4 = _csp(network, merged, weights, "backbone.C3_n3", dtype, residual=False)
    merged = graph.concatenate(
        network, [_conv(network, p4, weights, "backbone.bu_conv1", dtype, stride=2), lateral]
    )
    p5 = _csp(network, merged, weights, "backbone.C3_n4", dtype, residual=False)
    return p3, p4, p5


def _predict(network, tensor, weights: _Weights, prefix: str, dtype, *, channels: int):
    weight, bias = weights.raw(f"{prefix}.weight"), weights.raw(f"{prefix}.bias")
    if weight.shape != (channels, int(tensor.shape[1]), 1, 1) or bias.shape != (channels,):
        raise ValueError(f"Unsupported YOLOX prediction shape: {prefix}")
    return graph.convolution(network, tensor, weight, bias, dtype=dtype)


def _detect(network, sources, weights: _Weights, dtype):
    box_parts, score_parts = [], []
    for level, (tensor, stride) in enumerate(zip(sources, _STRIDES, strict=True)):
        stem = _conv(network, tensor, weights, f"head.stems.{level}", dtype)
        cls_feature, reg_feature = stem, stem
        for index in range(2):
            cls_feature = _conv(
                network, cls_feature, weights, f"head.cls_convs.{level}.{index}", dtype
            )
            reg_feature = _conv(
                network, reg_feature, weights, f"head.reg_convs.{level}.{index}", dtype
            )
        regression = _predict(
            network, reg_feature, weights, f"head.reg_preds.{level}", dtype, channels=4
        )
        objectness = _predict(
            network, reg_feature, weights, f"head.obj_preds.{level}", dtype, channels=1
        )
        classes = _predict(
            network, cls_feature, weights, f"head.cls_preds.{level}", dtype, channels=_NUM_CLASSES
        )
        rows, columns = map(int, regression.shape[2:])
        cells = rows * columns
        regression = graph.reshape(network, regression, (4, cells))
        offset = graph.slice_axis(network, regression, axis=0, start=0, count=2)
        log_size = graph.slice_axis(network, regression, axis=0, start=2, count=2)
        y, x = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
        grid = graph.constant(network, np.stack([x.ravel(), y.ravel()]), dtype=np.float32)
        centre = graph.scale(network, graph.add(network, offset, grid), stride, dtype=np.float32)
        exp = network.add_unary(log_size, trt.UnaryOperation.EXP)
        if exp is None:
            raise RuntimeError("TensorRT rejected YOLOX box exponential")
        half = graph.scale(network, exp.get_output(0), stride * 0.5, dtype=np.float32)
        corners = graph.concatenate(
            network,
            [graph.subtract(network, centre, half), graph.add(network, centre, half)],
            axis=0,
        )
        box_parts.append(graph.permute(network, corners, (1, 0)))
        probabilities = graph.reshape(
            network, graph.sigmoid(network, classes), (_NUM_CLASSES, cells)
        )
        confidence = graph.reshape(network, graph.sigmoid(network, objectness), (1, cells))
        score_parts.append(
            graph.permute(network, graph.multiply(network, probabilities, confidence), (1, 0))
        )
    boxes = graph.concatenate(network, box_parts, axis=0)
    scores = graph.concatenate(network, score_parts, axis=0)
    best, index = graph.top_k(network, scores, k=1, axis=1)
    total = int(boxes.shape[0])
    return (boxes, graph.reshape(network, best, (total,)), graph.reshape(network, index, (total,)))


def _build_engine(checkpoint: Checkpoint, precision: str, verbose: bool) -> bytes:
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported YOLOX precision: {precision}")
    numpy_dtype = np.float16 if precision == "fp16" else np.float32
    # Fold once in FP32; graph.convolution converts weights to each layer's dtype.
    weights = _Weights(checkpoint, np.float32)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    size = checkpoint.image_size
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the YOLOX input")
    sources = _backbone(network, pixels, weights, numpy_dtype)
    sources = _neck(network, sources, weights, np.float32)
    outputs = _detect(network, sources, weights, np.float32)
    checkpoint.assert_consumed()
    for tensor, name in zip(outputs, ("boxes", "scores", "classes"), strict=True):
        tensor.name = name
        network.mark_output(tensor)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT YOLOX engine build failed")
    return bytes(plan)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build an official 80-class YOLOX detector at its published input size."""
    if request.backend != "trt":
        raise NotImplementedError("yolox supports only backend=trt")
    if request.task != "object_detection":
        raise ValueError("yolox supports only task=object_detection")
    if request.dynamic_kv_cache:
        raise NotImplementedError("yolox does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("yolox does not support image_height or image_width overrides")
    if request.video_num_frames is not None:
        raise NotImplementedError("yolox does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("yolox does not support max_batch_size other than one")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("yolox does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("yolox does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("yolox does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("yolox does not support mixed-precision layer overrides")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("yolox does not support max_sequence_length")
    checkpoint = Checkpoint.open(Path(request.model_dir))
    plan = _build_engine(checkpoint, str(request.precision).lower(), bool(request.verbose))
    writer.set_header(family="yolox", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": checkpoint.image_size,
            "input_image_w": checkpoint.image_size,
            "pad_value": 114,
            "score_threshold": 0.25,
            "iou_threshold": 0.45,
            "num_classes": _NUM_CLASSES,
            "max_detections": sum((checkpoint.image_size // stride) ** 2 for stride in _STRIDES),
        },
    )
