# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Flash-Next per-layer embedding graph."""

import math

import numpy as np
import tensorrt as trt

from .ple_weights import fp16_table_scale


def dequantize_ngram_rows(network, rows, scale):
    """Convert the selected FP8 rows [tokens, 16, 160] to FP16 token embeddings."""
    if rows.dtype != trt.fp8 or tuple(rows.shape)[1:] != (16, 160):
        raise ValueError("Flash-Next expects FP8 rows with 16 heads of width 160")
    factor = network.add_constant((), np.array(fp16_table_scale(scale))).get_output(0)
    value = network.add_dequantize(rows, factor, trt.float16).get_output(0)
    flatten = network.add_shuffle(value)
    flatten.reshape_dims = (0, 2560)
    return flatten.get_output(0)


def lexical_injection(network, embeddings, hyper_states, history, weights, epsilon):
    """Inject lexical features for one request; return the addition and new history.

    embeddings [tokens, 2560], hyper_states [tokens, 10240], and the preceding
    nine normalized rows [9, 10240] all use FP16. History is zero for a new
    sequence; the caller commits the new history only after a successful step.
    """
    n = network
    streams, hidden = 4, 2560
    channels = streams * hidden
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid Flash-Next normalization epsilon")

    def constant(array):
        return n.add_constant(array.shape, np.ascontiguousarray(array)).get_output(0)

    def weight(name, shape, dtype):
        w = weights[name]
        if w.shape != shape:
            raise ValueError(
                f"Invalid Flash-Next PLE weight {name}: expected {shape}, got {w.shape}"
            )
        return np.asarray(w, dtype=dtype)

    def cast(x, dtype):
        return n.add_cast(x, dtype).get_output(0)

    def reshape(x, shape):
        layer = n.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def binary(x, y, op):
        return n.add_elementwise(x, y, op).get_output(0)

    def unary(x, op):
        return n.add_unary(x, op).get_output(0)

    ew, un = trt.ElementWiseOperation, trt.UnaryOperation

    def linear(x, name, out):
        w = constant(weight(name, (out, hidden), np.float16))
        return n.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def norm(x, name):
        # HF accumulates in FP32 separately for each residual stream, applies
        # (1 + weight) before rounding back to the input dtype.
        a = reshape(cast(x, trt.float32), (0, streams, hidden))
        sq = binary(a, a, ew.PROD)
        mean = n.add_reduce(sq, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
        eps = constant(np.full((1, 1, 1), epsilon, np.float32))
        denom = unary(binary(mean, eps, ew.SUM), un.SQRT)
        scale = 1.0 + weight(name, (channels,), np.float32).reshape(1, streams, hidden)
        return cast(binary(binary(a, denom, ew.DIV), constant(scale), ew.PROD), trt.float16)

    key = norm(linear(embeddings, "key_proj.weight", channels), "norm_key.weight")
    query = norm(hyper_states, "norm_query.weight")
    value = reshape(linear(embeddings, "value_proj.weight", hidden), (0, 1, hidden))
    product = cast(binary(key, query, ew.PROD), trt.float32)
    dot = cast(
        n.add_reduce(product, trt.ReduceOperation.SUM, 1 << 2, True).get_output(0), trt.float16
    )
    gate = binary(dot, constant(np.full((1, 1, 1), math.sqrt(hidden), np.float16)), ew.DIV)
    bounded = binary(unary(gate, un.ABS), constant(np.full((1, 1, 1), 1e-6, np.float16)), ew.MAX)
    gate = binary(unary(bounded, un.SQRT), unary(gate, un.SIGN), ew.PROD)
    gate = n.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
    shared = reshape(binary(value, gate, ew.PROD), (0, channels))

    current = reshape(norm(shared, "norm_conv.weight"), (0, channels))
    concat = n.add_concatenation([history, current])
    concat.axis = 0
    timeline = concat.get_output(0)
    size = n.add_shape(current).get_output(0)
    kernel = weight("conv1d.weight", (channels, 1, 4), np.float32)
    acc = None
    for tap in range(4):
        # Cross-correlation reads offsets 0,3,6,9 (oldest first). Do not
        # convolve across different requests or discard history on decode.
        part = n.add_slice(timeline, (3 * tap, 0), (1, channels), (1, 1))
        part.set_input(2, size)
        term = binary(cast(part.get_output(0), trt.float32), constant(kernel[:, :, tap].T), ew.PROD)
        acc = term if acc is None else binary(acc, term, ew.SUM)
    conv = cast(acc, trt.float16)
    silu = binary(conv, n.add_activation(conv, trt.ActivationType.SIGMOID).get_output(0), ew.PROD)

    tokens = n.add_slice(size, (0,), (1,), (1,)).get_output(0)
    zero = constant(np.array([0], np.int64 if size.dtype == trt.int64 else np.int32))
    start = n.add_concatenation([tokens, zero])
    start.axis = 0
    tail = n.add_slice(timeline, (0, 0), (9, channels), (1, 1))
    tail.set_input(1, start.get_output(0))
    return binary(shared, silu, ew.SUM), tail.get_output(0)


def dequantize_packed_ngram_rows(network, words, scale):
    """Read original E4M3 bytes in INT32 transport words, including on pre-FP8 devices."""
    import ml_dtypes

    if words.dtype != trt.int32 or tuple(words.shape)[1:] != (640,):
        raise ValueError("Flash-Next expects 2,560 FP8 bytes per selected token")
    scale = fp16_table_scale(scale)
    n, ew = network, trt.ElementWiseOperation

    def const(a):
        return n.add_constant(a.shape, np.ascontiguousarray(a)).get_output(0)

    def op(x, y, kind):
        return n.add_elementwise(x, y, kind).get_output(0)

    sh = n.add_shuffle(words)
    sh.reshape_dims = (0, 640, 1)
    divisors = const(np.array([1, 256, 65536, 16777216], np.int32).reshape(1, 1, 4))
    shifted = op(sh.get_output(0), divisors, ew.FLOOR_DIV)
    radix = const(np.full((1, 1, 1), 256, np.int32))
    ids = op(shifted, op(op(shifted, radix, ew.FLOOR_DIV), radix, ew.PROD), ew.SUB)
    raw = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float16)
    # Same FP16 scale and DQ rounding as the FP8-typed TensorRT dequantizer.
    table = raw * scale
    out = n.add_shuffle(n.add_gather(const(table), ids, 0).get_output(0))
    out.reshape_dims = (0, 2560)
    return out.get_output(0)
