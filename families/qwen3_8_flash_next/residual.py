# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read, update, and collapse the four residual streams of the Flash-Next decoder."""

import math

import numpy as np
import tensorrt as trt


_STREAMS = 4
_WIDTH = 2560
_CHANNELS = _STREAMS * _WIDTH
_RANK = 320


def _reshape(network, tensor, dims):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = dims
    return layer.get_output(0)


def _read(network, hyper, checkpoint, prefix, eps, *, inject):
    if hyper.dtype != trt.float16 or tuple(hyper.shape)[1:] != (_CHANNELS,):
        raise ValueError("Flash-Next expects a matrix of four FP16 residual streams")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid Flash-Next residual normalization epsilon")
    n, ew = network, trt.ElementWiseOperation

    def const(array):
        return n.add_constant(array.shape, np.ascontiguousarray(array)).get_output(0)

    def elt(left, right, op):
        return n.add_elementwise(left, right, op).get_output(0)

    def sigmoid(x):
        return n.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)

    def linear(x, suffix, shape):
        weight = checkpoint.bf16(f"{prefix}.{suffix}.weight", shape)
        inputs = n.add_cast(x, trt.float32).get_output(0)
        weights = n.add_cast(const(weight), trt.float32).get_output(0)
        y = n.add_matrix_multiply(
            inputs, trt.MatrixOperation.NONE, weights, trt.MatrixOperation.TRANSPOSE
        )
        return n.add_cast(y.get_output(0), trt.float16).get_output(0)

    # RMS is independent per stream. The learned offset is applied in FP32
    # before rounding, not rounded as part of an FP16 normalization weight.
    streams = _reshape(n, n.add_cast(hyper, trt.float32).get_output(0), (0, _STREAMS, _WIDTH))
    sq = elt(streams, streams, ew.PROD)
    mean = n.add_reduce(sq, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
    den = n.add_unary(
        elt(mean, const(np.full((1, 1, 1), eps, np.float32)), ew.SUM), trt.UnaryOperation.SQRT
    ).get_output(0)
    gamma = checkpoint.bf16(f"{prefix}.hc_norm.weight", (_CHANNELS,), fp32=True)
    rms = elt(elt(streams, den, ew.DIV), const((1 + gamma).reshape(1, _STREAMS, _WIDTH)), ew.PROD)
    normed = n.add_cast(rms, trt.float16).get_output(0)
    flat = _reshape(n, normed, (0, _CHANNELS))
    quarter = const(np.full((1, 1), 1 / _STREAMS, np.float16))
    low = elt(linear(flat, "input_mix_weight_down", (_RANK, _CHANNELS)), quarter, ew.PROD)
    # SiLU computes in FP32 and rounds once, after multiplication.
    low = n.add_cast(low, trt.float32).get_output(0)
    low = n.add_cast(elt(low, sigmoid(low), ew.PROD), trt.float16).get_output(0)
    mix = sigmoid(linear(low, "input_mix_weight_up", (_CHANNELS, _RANK)))
    mix = elt(_reshape(n, mix, (0, _STREAMS, _WIDTH)), normed, ew.PROD)
    # Average the four already rounded FP16 products, returning an FP16 read.
    mixed = n.add_reduce(
        n.add_cast(mix, trt.float32).get_output(0), trt.ReduceOperation.AVG, 1 << 1, False
    ).get_output(0)
    mixed = n.add_cast(mixed, trt.float16).get_output(0)
    if not inject:
        return mixed, None
    logits = linear(flat, "block_inject_weight", (_STREAMS, _CHANNELS))
    scale = elt(
        sigmoid(elt(logits, quarter, ew.PROD)), const(np.full((1, 1), 2, np.float16)), ew.PROD
    )
    return mixed, scale


def gated_sublayer(network, hyper, checkpoint, prefix, eps, block):
    """Run a token block on the learned read, then inject it into the original streams.

    The callback returns (output, state). State is passed through unmodified:
    only the caller knows the cache format for a linear, sparse, or MoE block.
    """
    mixed, weights = read_for_block(network, hyper, checkpoint, prefix, eps)
    result, state = block(mixed)
    return inject_block(network, hyper, result, weights), state


def read_for_block(network, hyper, checkpoint, prefix, eps):
    """Return the normalized token read and the stream injection coefficients."""
    return _read(network, hyper, checkpoint, prefix, eps, inject=True)


def inject_block(network, hyper, result, weights):
    """Write one completed block output to its original, unnormalized streams."""
    if result.dtype != trt.float16 or tuple(result.shape)[1:] != (_WIDTH,):
        raise ValueError("Flash-Next block must return one FP16 vector per token")
    ew = trt.ElementWiseOperation
    scaled = network.add_elementwise(
        _reshape(network, result, (0, 1, _WIDTH)),
        _reshape(network, weights, (0, _STREAMS, 1)),
        ew.PROD,
    ).get_output(0)
    out = network.add_elementwise(hyper, _reshape(network, scaled, (0, _CHANNELS)), ew.SUM)
    return out.get_output(0)


def collapse_residual(network, hyper, checkpoint, eps):
    """The stack-exit mixer reads all four streams but has no injection matrix."""
    return _read(
        network, hyper, checkpoint, "model.language_model.hyper_connection_mixer", eps, inject=False
    )[0]
