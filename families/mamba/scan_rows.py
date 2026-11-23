# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba selective scan over ``S`` rows in one launch.

The single-token decoder applies, per channel ``d`` and state ``n``::

    h_t = exp(dt_t[d] A[d, n]) h_{t-1} + dt_t[d] B_t[n] x_t[d]
    y_t[d] = sum_n C_t[n] h_t[d, n] + D[d] x_t[d]

With ``h_t = a_t h_{t-1} + b_t`` the rows are combined by a Hillis-Steele scan:
``ceil(log2(S))`` rounds of ``(a, b)_t <- (a_t a_{t-k}, a_t b_{t-k} + b_t)``
for ``k = 1, 2, 4, ...``. Afterwards ``a_t`` is the decay product since the
chunk start and ``b_t`` the zero-state recurrence, so ``h_t = a_t h_0 + b_t``.
Every factor is a decay in ``(0, 1]``, so the scan stays finite in FP32.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

# Prefill rows of one launch; the scan runs ceil(log2(MAX_ROWS)) rounds.
MAX_ROWS = 64


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


def _shift_rows(network, tensor, rows, shift: int, fill: float, width: tuple[int, int]):
    """``out[t] = tensor[t - shift]`` for ``t >= shift``, else ``fill`` (``[S, d, n]``)."""
    pad = _ew(
        network,
        _const(network, np.full((shift, 1, 1), fill, np.float32)),
        _const(network, np.zeros((1, *width), np.float32)),
        trt.ElementWiseOperation.SUM,
    )
    joined = network.add_concatenation([pad, tensor])
    joined.axis = 0
    return dynamic_slice(network, joined.get_output(0), (0, 0, 0), (rows, *width))


def add_selective_scan_rows(
    network: trt.INetworkDefinition,
    *,
    decay: trt.ITensor,
    update: trt.ITensor,
    state: trt.ITensor,
    width: tuple[int, int],
) -> tuple[trt.ITensor, trt.ITensor]:
    """``h_t = decay_t h_{t-1} + update_t`` over ``[S, d, n]`` FP32 rows from ``state [d, n]``.

    Returns ``(h [S, d, n], h_last [d, n])``.
    """
    prod_op, sum_op = trt.ElementWiseOperation.PROD, trt.ElementWiseOperation.SUM
    rows = row_count(network, decay)
    shift = 1
    while shift < MAX_ROWS:
        prior_decay = _shift_rows(network, decay, rows, shift, 1.0, width)
        prior_update = _shift_rows(network, update, rows, shift, 0.0, width)
        update = _ew(network, _ew(network, decay, prior_update, prod_op), update, sum_op)
        decay = _ew(network, decay, prior_decay, prod_op)
        shift *= 2
    start = shuffle(network, state, reshape=(1, *width))
    states = _ew(network, _ew(network, decay, start, prod_op), update, sum_op)
    last = _ew(network, rows, _i64(network, [1]), trt.ElementWiseOperation.SUB)
    final = dynamic_slice(network, states, (last, 0, 0), (1, *width))
    return states, shuffle(network, final, reshape=width)
