# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse mixed NVFP4/MXFP8 math expressed solely through TensorRT Python APIs."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Projection:
    algorithm: str
    weight: np.ndarray
    output_width: int
    input_width: int
    block_scales: np.ndarray | None = None
    weight_scale: float | None = None
    input_scale: float | None = None

    def __post_init__(self):
        if self.algorithm not in {"dense", "NVFP4", "MXFP8"}:
            raise ValueError(f"Unknown Muse projection format: {self.algorithm}")
        for name in ("weight_scale", "input_scale"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"Muse {name} must be finite and positive")


def constant(network, array, keep_alive, dtype=None):
    import tensorrt as trt

    array = np.ascontiguousarray(array)
    keep_alive.append(array)
    weights = trt.Weights(array) if dtype is None else trt.Weights(dtype, array.ctypes.data, array.size)
    return network.add_constant(array.shape, weights).get_output(0)


def weight_tensor(network, projection: Projection, keep_alive, output_dtype):
    """Retain packed weights; perform complete reconstruction before FP16 rounding."""
    import tensorrt as trt

    p = projection
    if p.algorithm == "dense":
        w = constant(network, p.weight.astype(np.float32), keep_alive)
        return network.add_cast(w, output_dtype).get_output(0)
    if p.block_scales is None:
        raise ValueError("Muse packed projection requires block scales")
    if p.algorithm == "NVFP4":
        if p.weight_scale is None:
            raise ValueError("Muse NVFP4 requires the source global weight scale")
        keep_alive.append(p.weight)
        weights = trt.Weights(trt.DataType.FP4, p.weight.ctypes.data, p.output_width * p.input_width)
        codes = network.add_constant((p.output_width, p.input_width), weights).get_output(0)
        blocks = constant(network, p.block_scales, keep_alive, trt.DataType.FP8)
        global_scale = constant(network, np.asarray([p.weight_scale], np.float32), keep_alive)
        scale = network.add_dequantize(blocks, global_scale, trt.float32).get_output(0)
        decoded = network.add_dequantize(codes, scale, trt.float32)
    else:
        codes = constant(network, p.weight, keep_alive, trt.DataType.FP8)
        scale = constant(network, p.block_scales, keep_alive, trt.DataType.E8M0)
        decoded = network.add_dequantize(codes, scale, output_dtype)
    decoded.axis = 1
    value = decoded.get_output(0)
    return value if value.dtype == output_dtype else network.add_cast(value, output_dtype).get_output(0)


def activation_tensor(network, activation, projection: Projection, keep_alive):
    """Preserve source dynamic quantization, including reconstruction scale rounding."""
    import tensorrt as trt

    p = projection
    if p.algorithm == "dense":
        return activation
    if p.algorithm == "MXFP8":
        # The scale of an all-zero block is arbitrary, but its reconstructed
        # values must be zero. Avoid passing an undefined zero scale through
        # E8M0 conversion, without changing any nonzero block's quantization.
        blocked = network.add_shuffle(activation)
        blocked.reshape_dims = (-1, p.input_width // 32, 32)
        magnitude = network.add_unary(blocked.get_output(0), trt.UnaryOperation.ABS)
        maximum = network.add_reduce(magnitude.get_output(0), trt.ReduceOperation.MAX, 4, True)
        zero = constant(network, np.zeros((1, 1, 1), np.float16), keep_alive)
        one = constant(network, np.ones((1, 1, 1), np.float16), keep_alive)
        nonzero = network.add_elementwise(maximum.get_output(0), zero, trt.ElementWiseOperation.GREATER)
        safe = network.add_select(nonzero.get_output(0), blocked.get_output(0), one)
        rows = network.add_shuffle(safe.get_output(0))
        rows.reshape_dims = (-1, p.input_width)
        quant = network.add_dynamic_quantize(rows.get_output(0), 1, 32, trt.DataType.FP8, trt.DataType.E8M0)
        decoded = network.add_dequantize(quant.get_output(0), quant.get_output(1), activation.dtype)
        decoded.axis = 1
        result = network.add_shuffle(decoded.get_output(0))
        result.reshape_dims = (-1, p.input_width // 32, 32)
        result = network.add_select(nonzero.get_output(0), result.get_output(0), zero)
        rows = network.add_shuffle(result.get_output(0))
        rows.reshape_dims = (-1, p.input_width)
        return rows.get_output(0)
    else:
        if p.input_scale is None:
            raise ValueError("Muse NVFP4 linear requires the source calibrated input scale")
        scale = constant(network, np.asarray([p.input_scale], np.float32), keep_alive)
        quant = network.add_dynamic_quantize(activation, 1, 16, trt.DataType.FP4, trt.DataType.FP8)
        quant.set_input(1, scale)
        reconstruction = network.add_cast(scale, activation.dtype).get_output(0)
        blocks = network.add_dequantize(quant.get_output(1), reconstruction, activation.dtype)
        decoded = network.add_dequantize(quant.get_output(0), blocks.get_output(0), activation.dtype)
    decoded.axis = 1
    return decoded.get_output(0)


def linear(network, activation, projection: Projection, keep_alive):
    import tensorrt as trt

    if len(activation.shape) != 2 or activation.shape[1] != projection.input_width:
        raise ValueError("Muse linear expects token rows with the checkpoint input width")
    x = activation_tensor(network, activation, projection, keep_alive)
    w = weight_tensor(network, projection, keep_alive, activation.dtype)
    return network.add_matrix_multiply(x, trt.MatrixOperation.NONE, w,
                                       trt.MatrixOperation.TRANSPOSE).get_output(0)
