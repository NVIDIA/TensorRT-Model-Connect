# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba-2 (SSD) mixer over ``S`` rows in one launch.

The single-token decoder applies, per head ``h`` and token ``t``::

    S_t = exp(dt_t A) S_{t-1} + dt_t x_t B_t^T      (state [P, N])
    y_t = S_t C_t + D x_t

With ``a_t = dt_t A`` and ``gamma_t = sum_{i<=t} a_i`` a chunk of ``S`` tokens
is, without any sequential step::

    y_t = exp(gamma_t) S_0 C_t + sum_{i<=t} exp(gamma_t - gamma_i) dt_i (C_t . B_i) x_i + D x_t
    S_S = exp(gamma_S) S_0 + sum_i exp(gamma_S - gamma_i) dt_i x_i B_i^T

Every exponent is ``<= 0``, so the chunk form stays finite in FP32.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

# Static size of the causal masks; it bounds the prefill rows.
MAX_ROWS = 128


def _const(network, values: np.ndarray) -> trt.ITensor:
    values = np.ascontiguousarray(values)
    return network.add_constant(values.shape, trt.Weights(values)).get_output(0)


def _i64(network, values) -> trt.ITensor:
    return _const(network, np.asarray(values, dtype=np.int64))


def _ew(network, a, b, op) -> trt.ITensor:
    return network.add_elementwise(a, b, op).get_output(0)


def _mm(network, a, b, transpose_a=False, transpose_b=False) -> trt.ITensor:
    op_a = trt.MatrixOperation.TRANSPOSE if transpose_a else trt.MatrixOperation.NONE
    op_b = trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE
    return network.add_matrix_multiply(a, op_a, b, op_b).get_output(0)


def shuffle(network, tensor, reshape=None, transpose=None) -> trt.ITensor:
    layer = network.add_shuffle(tensor)
    if transpose is not None:
        layer.first_transpose = transpose
    if reshape is not None:
        layer.reshape_dims = reshape
    return layer.get_output(0)


def row_count(network, tensor: trt.ITensor) -> trt.ITensor:
    """Length of axis 0 as an int64 ``[1]`` shape tensor."""
    shape = network.add_shape(tensor).get_output(0)
    return network.add_slice(shape, (0,), (1,), (1,)).get_output(0)


def _dims(network, parts) -> trt.ITensor:
    tensors = [p if isinstance(p, trt.ITensor) else _i64(network, [p]) for p in parts]
    joined = network.add_concatenation(tensors)
    joined.axis = 0
    return joined.get_output(0)


def dynamic_slice(network, tensor, start, size) -> trt.ITensor:
    """Slice with static or shape-tensor start and size entries."""
    rank = len(start)
    layer = network.add_slice(tensor, (0,) * rank, (1,) * rank, (1,) * rank)
    if any(isinstance(s, trt.ITensor) for s in start):
        layer.set_input(1, _dims(network, start))
    else:
        layer.start = tuple(start)
    layer.set_input(2, _dims(network, size))
    return layer.get_output(0)


def last_row(network, tensor: trt.ITensor, width: int) -> trt.ITensor:
    """``[S, width]`` -> ``[1, width]``, the last row."""
    last = _ew(
        network, row_count(network, tensor), _i64(network, [1]), trt.ElementWiseOperation.SUB
    )
    return dynamic_slice(network, tensor, (last, 0), (1, width))


def add_short_conv_rows(
    network: trt.INetworkDefinition,
    *,
    columns: trt.ITensor,
    conv_state: trt.ITensor,
    weight: trt.ITensor,
    channels: int,
    kernel: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Causal depthwise conv over ``columns`` ``[C, S]`` with a ``[C, kernel]`` history.

    The history holds the last ``kernel`` inputs (newest last). Output column
    ``t`` is ``sum_j weight[:, j] * window[:, t + j]`` over
    ``window = [history[:, 1:], columns]``; the new history is the last
    ``kernel`` columns of the window. Returns ``(outputs [C, S], history)``.
    """
    rows = row_count(network, shuffle(network, columns, transpose=(1, 0)))
    if kernel > 1:
        past = network.add_slice(conv_state, (0, 1), (channels, kernel - 1), (1, 1)).get_output(0)
        window = network.add_concatenation([past, columns])
        window.axis = 1
        window = window.get_output(0)
    else:
        window = columns
    total = None
    for tap in range(kernel):
        tap_weight = network.add_slice(weight, (0, tap), (channels, 1), (1, 1)).get_output(0)
        shifted = dynamic_slice(network, window, (0, tap), (channels, rows))
        term = _ew(network, shifted, tap_weight, trt.ElementWiseOperation.PROD)
        total = term if total is None else _ew(network, total, term, trt.ElementWiseOperation.SUM)
    last = _ew(network, rows, _i64(network, [1]), trt.ElementWiseOperation.SUB)
    history = dynamic_slice(network, window, (0, last), (channels, kernel))
    return total, history


def add_chunked_ssd(
    network: trt.INetworkDefinition,
    *,
    x: trt.ITensor,
    b: trt.ITensor,
    c: trt.ITensor,
    dt: trt.ITensor,
    a: np.ndarray,
    d: np.ndarray,
    state: trt.ITensor,
    num_heads: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    """SSD over one chunk, all tensors FP32.

    ``x`` is ``[H, S, P]``, ``b``/``c`` are ``[H, S, N]``, ``dt`` is ``[H, S]``
    (after softplus) and ``state`` is ``[H, P, N]``; ``a`` and ``d`` are the
    per-head ``A`` (negative) and ``D`` values. Returns ``(y [H, S, P], state)``.
    """
    sum_op, prod_op = trt.ElementWiseOperation.SUM, trt.ElementWiseOperation.PROD
    rows = row_count(network, shuffle(network, dt, transpose=(1, 0)))
    log_decay = _ew(
        network, dt, _const(network, np.asarray(a, np.float32).reshape(num_heads, 1)), prod_op
    )  # [H, S]
    axis = network.add_constant((), np.array(1, dtype=np.int32)).get_output(0)
    gamma = network.add_cumulative(
        log_decay, axis, trt.CumulativeOperation.SUM, False, False
    ).get_output(0)
    gamma_rows = shuffle(network, gamma, reshape=(0, 0, 1))  # [H, S, 1]
    diff = _ew(
        network,
        gamma_rows,
        shuffle(network, gamma, reshape=(0, 1, -1)),
        trt.ElementWiseOperation.SUB,
    )
    diff = _ew(
        network,
        diff,
        _const(network, np.zeros((1, 1, 1), np.float32)),
        trt.ElementWiseOperation.MIN,
    )
    causal = dynamic_slice(
        network,
        _const(network, np.tril(np.ones((MAX_ROWS, MAX_ROWS), np.float32))[None]),
        (0, 0, 0),
        (1, rows, rows),
    )
    decay = _ew(
        network, network.add_unary(diff, trt.UnaryOperation.EXP).get_output(0), causal, prod_op
    )
    scores = _ew(
        network,
        _ew(network, decay, _mm(network, c, b, transpose_b=True), prod_op),
        shuffle(network, dt, reshape=(0, 1, -1)),
        prod_op,
    )  # [H, t, i]
    skip = _const(network, np.asarray(d, np.float32).reshape(num_heads, 1, 1))
    y = _ew(
        network,
        _ew(
            network,
            _ew(
                network,
                network.add_unary(gamma_rows, trt.UnaryOperation.EXP).get_output(0),
                _mm(network, c, state, transpose_b=True),
                prod_op,
            ),
            _mm(network, scores, x),
            sum_op,
        ),
        _ew(network, skip, x, prod_op),
        sum_op,
    )

    # gamma_S is the last cumulative sum, not a separate reduction: summed in
    # another order it can exceed gamma_i by many ulps at large |dt A|, and
    # exp(gamma_S - gamma_i) then overflows into a non-finite state. A float
    # running sum of non-positive terms never increases, and the clamp keeps
    # the exponent at most zero regardless.
    last = _ew(network, rows, _i64(network, [1]), trt.ElementWiseOperation.SUB)
    total = dynamic_slice(network, gamma, (0, last), (num_heads, 1))  # [H, 1]
    tail = _ew(
        network,
        _ew(network, total, gamma, trt.ElementWiseOperation.SUB),
        _const(network, np.zeros((1, 1), np.float32)),
        trt.ElementWiseOperation.MIN,
    )
    weights = _ew(
        network,
        network.add_unary(tail, trt.UnaryOperation.EXP).get_output(0),
        dt,
        prod_op,
    )  # exp(gamma_S - gamma_i) dt_i: [H, S]
    carried = _ew(
        network,
        network.add_unary(
            shuffle(network, total, reshape=(0, 1, 1)), trt.UnaryOperation.EXP
        ).get_output(0),
        state,
        prod_op,
    )
    final_state = _ew(
        network,
        carried,
        _mm(
            network,
            x,
            _ew(network, b, shuffle(network, weights, reshape=(0, 0, 1)), prod_op),
            transpose_a=True,
        ),
        sum_op,
    )
    return y, final_state
