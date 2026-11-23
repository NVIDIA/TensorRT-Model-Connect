# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A single-token native Puzzle mixture-of-experts transition."""

import numpy as np

from .iq2 import selected_weights


def _const(n, value):
    value = np.ascontiguousarray(value)
    return n.add_constant(value.shape, value).get_output(0)


def _shape(n, x, shape):
    step = n.add_shuffle(x)
    step.reshape_dims = shape
    return step.get_output(0)


def _ew(n, a, b, operation):
    return n.add_elementwise(a, b, operation).get_output(0)


def _cast(n, x, dtype):
    return n.add_cast(x, dtype).get_output(0)


def _unary(n, x, op):
    return n.add_unary(x, op).get_output(0)


def q8_linear(n, x, packed, *, output_dtype=None):
    """Q8_0 weights by Q8_1 activations, with independent 32-element block scales."""
    import tensorrt as trt

    x_input_dtype = x.dtype
    rows, blocks, stride = packed.shape
    if packed.dtype != np.uint8 or stride != 34 or tuple(x.shape) != (1, blocks * 32):
        raise ValueError("Expected one hidden row and matching packed Q8_0 weights")
    codes = np.ascontiguousarray(packed[..., 2:].transpose(1, 0, 2)).view(np.int8)
    scales = np.ascontiguousarray(packed[..., :2]).view("<f2").transpose(1, 0, 2)
    if not np.isfinite(scales).all():
        raise ValueError("Nonfinite Q8_0 block scale")
    op = trt.ElementWiseOperation
    x = _shape(n, _cast(n, x, trt.float32), (blocks, 32, 1))
    peak = n.add_reduce(
        _unary(n, x, trt.UnaryOperation.ABS), trt.ReduceOperation.MAX, 2, True
    ).get_output(0)
    one = _const(n, np.ones((1, 1, 1), dtype=np.float32))
    zero = _const(n, np.zeros((1, 1, 1), dtype=np.float32))
    positive = _ew(n, peak, zero, op.GREATER)
    safe_peak = n.add_select(positive, peak, one).get_output(0)
    inverse = _ew(n, _const(n, np.full((1, 1, 1), 127, np.float32)), safe_peak, op.DIV)
    scale = n.add_select(positive, _ew(n, one, inverse, op.DIV), zero).get_output(0)
    # GGML rounds halfway away from zero, not to the nearest even integer.
    v = _ew(n, x, inverse, op.PROD)
    mag = _unary(n, v, trt.UnaryOperation.ABS)
    absolute = _unary(
        n,
        _ew(n, mag, _const(n, np.full((1, 1, 1), 0.5, np.float32)), op.SUM),
        trt.UnaryOperation.FLOOR,
    )
    q = _ew(n, _unary(n, v, trt.UnaryOperation.SIGN), absolute, op.PROD)
    w = n.add_dequantize(
        _const(n, codes), _const(n, np.array(1.0, dtype=np.float32)), trt.float32
    ).get_output(0)
    # Each 32-term integer dot fits exactly within FP32's integer precision.
    dots = n.add_matrix_multiply(
        w, trt.MatrixOperation.NONE, q, trt.MatrixOperation.NONE
    ).get_output(0)
    dots = _ew(n, dots, _cast(n, _const(n, scales), trt.float32), op.PROD)
    dots = _ew(n, dots, scale, op.PROD)
    result = n.add_reduce(dots, trt.ReduceOperation.SUM, 1, False).get_output(0)
    result = _shape(n, result, (1, rows))
    dtype = x_input_dtype if output_dtype is None else output_dtype
    if dtype not in (trt.float32, trt.float16, trt.bfloat16):
        raise ValueError("Unsupported Bolt Q8 output precision")
    return result if dtype == trt.float32 else _cast(n, result, dtype)


def _expert(n, x, up_blocks, down_blocks, indices):
    import tensorrt as trt

    up = selected_weights(n, up_blocks, indices)
    y = n.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, up, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    relu = n.add_activation(y, trt.ActivationType.RELU).get_output(0)
    y = _ew(n, relu, relu, trt.ElementWiseOperation.PROD)
    down = selected_weights(n, down_blocks, indices)
    return n.add_matrix_multiply(
        y, trt.MatrixOperation.NONE, down, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)


def puzzle_moe(n, hidden, weights, *, selected, normalise, routed_scale):
    """Mix a normalized BF16 token using this layer's actual heterogeneous geometry."""
    import tensorrt as trt

    if len(hidden.shape) != 2 or hidden.shape[0] != 1 or hidden.dtype != trt.bfloat16:
        raise ValueError("Expected a single normalized BF16 token")
    op = trt.ElementWiseOperation
    count, width = weights["gate"].shape
    if (
        not 0 < selected <= count
        or weights["correction"].shape != (count,)
        or width != hidden.shape[1]
    ):
        raise ValueError("Inconsistent Puzzle router geometry")
    logits = n.add_matrix_multiply(
        _cast(n, hidden, trt.float32),
        trt.MatrixOperation.NONE,
        _const(n, weights["gate"].astype(np.float32)),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = n.add_activation(logits, trt.ActivationType.SIGMOID).get_output(0)
    corrected = _ew(n, scores, _const(n, weights["correction"].reshape(1, -1)), op.SUM)
    indices = _shape(
        n, n.add_topk(corrected, trt.TopKOperation.MAX, selected, 2).get_output(1), (selected,)
    )
    # The additive correction chooses IDs, but does not enter their combination weights.
    mix = n.add_gather(_shape(n, scores, (count,)), indices, 0).get_output(0)
    if normalise:
        denom = n.add_reduce(mix, trt.ReduceOperation.SUM, 1, True).get_output(0)
        mix = _ew(n, mix, _ew(n, denom, _const(n, np.array([1e-20], np.float32)), op.SUM), op.DIV)
    mix = _ew(n, mix, _const(n, np.array([routed_scale], np.float32)), op.PROD)
    x = q8_linear(n, hidden, weights["latent_in"])
    latent = x.shape[-1]
    expert = _expert(n, _shape(n, x, (1, 1, latent)), weights["up"], weights["down"], indices)
    weighted = _ew(n, _cast(n, expert, trt.float32), _shape(n, mix, (selected, 1, 1)), op.PROD)
    combined = n.add_reduce(weighted, trt.ReduceOperation.SUM, 1, False).get_output(0)
    out = q8_linear(n, _cast(n, combined, trt.bfloat16), weights["latent_out"])
    shared = _expert(
        n,
        _shape(n, hidden, (1, 1, width)),
        weights["shared_up"][None],
        weights["shared_down"][None],
        _const(n, np.array([0], dtype=np.int32)),
    )
    return _ew(n, out, _shape(n, shared, (1, width)), op.SUM), indices, mix
