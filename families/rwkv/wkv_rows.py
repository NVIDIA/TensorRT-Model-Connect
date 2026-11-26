# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-4 time mixing over ``S`` rows in one launch.

The single-token decoder keeps, per channel, ``num * exp(max)`` and
``den * exp(max)`` of

    A_t = exp(w) A_{t-1} + exp(k_t) v_t,    B_t = exp(w) B_{t-1} + exp(k_t)

and outputs ``wkv_t = (A_{t-1} + exp(u + k_t) v_t) / (B_{t-1} + exp(u + k_t))``
(``w`` is the negative ``time_decay``, ``u`` the ``time_first`` bonus). The
decay does not depend on the token, so for a chunk with incoming state
``(num_0, den_0, max_0)`` row ``t`` sums three kinds of terms, with exponents

    max_0 + t w                 (the incoming state)
    k_i + (t - 1 - i) w,  i < t (earlier rows of the chunk)
    u + k_t                     (the bonus of row t)

Each row subtracts the largest exponent before ``exp``, as the decoder does.
Row ``S`` (no bonus) is the state after the chunk, kept as ``(num, den, max)``
with ``max`` that largest exponent: the same ``A`` and ``B`` at a possibly
different scale than token-by-token steps would keep.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

# Rows of one launch; the distance table below is sized for it.
MAX_ROWS = 64
# Exponent of terms that do not exist; exp() of it minus any real maximum is 0.
_ABSENT = -1.0e30


def _const(network, values: np.ndarray) -> trt.ITensor:
    values = np.ascontiguousarray(values)
    return network.add_constant(values.shape, trt.Weights(values)).get_output(0)


def _i64(network, values) -> trt.ITensor:
    return _const(network, np.asarray(values, dtype=np.int64))


def _ew(network, a, b, op) -> trt.ITensor:
    return network.add_elementwise(a, b, op).get_output(0)


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


def _plus_one(network, rows: trt.ITensor) -> trt.ITensor:
    return _ew(network, rows, _i64(network, [1]), trt.ElementWiseOperation.SUM)


def last_row(network, tensor: trt.ITensor, width: int) -> trt.ITensor:
    """``[S, width]`` -> ``[1, width]``, the last row."""
    last = _ew(
        network, row_count(network, tensor), _i64(network, [1]), trt.ElementWiseOperation.SUB
    )
    return dynamic_slice(network, tensor, (last, 0), (1, width))


def shifted_rows(network, rows_in: trt.ITensor, previous: trt.ITensor, width: int) -> trt.ITensor:
    """Row ``t`` of the result is row ``t - 1`` of ``rows_in`` (``[S, width]``); row 0 is ``previous``."""
    joined = network.add_concatenation([previous, rows_in])
    joined.axis = 0
    return dynamic_slice(
        network, joined.get_output(0), (0, 0), (row_count(network, rows_in), width)
    )


def _append_row(network, rows_in: trt.ITensor, value: float, width: int) -> trt.ITensor:
    """``[S, width]`` -> ``[S + 1, width]`` with a constant last row."""
    joined = network.add_concatenation(
        [rows_in, _const(network, np.full((1, width), value, np.float32))]
    )
    joined.axis = 0
    return joined.get_output(0)


def add_wkv_rows(
    network: trt.INetworkDefinition,
    *,
    k: trt.ITensor,
    v: trt.ITensor,
    time_decay: trt.ITensor,
    time_first: trt.ITensor,
    num_state: trt.ITensor,
    den_state: trt.ITensor,
    max_state: trt.ITensor,
    width: int,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor, trt.ITensor]:
    """WKV of ``S`` rows (FP32 ``[S, width]`` ``k`` and ``v``, ``[1, width]`` decay, bonus and state).

    Returns ``(wkv [S, width], num, den, max)``, the new state ``[1, width]`` each.
    """
    op = trt.ElementWiseOperation
    rows = row_count(network, k)
    rows1 = _plus_one(network, rows)

    # distance[t, i] = t - 1 - i: the decay steps from row i to the state read by row t.
    t_index = np.arange(MAX_ROWS + 1, dtype=np.float32)
    distance = dynamic_slice(
        network,
        _const(network, t_index[:, None] - 1.0 - np.arange(MAX_ROWS, dtype=np.float32)[None, :]),
        (0, 0),
        (rows1, rows),
    )
    steps = dynamic_slice(network, _const(network, t_index[:, None]), (0, 0), (rows1, 1))

    # Earlier rows: [S + 1, S, width]; rows i >= t do not exist.
    distance3 = shuffle(network, distance, reshape=(0, 0, 1))
    decay3 = shuffle(network, time_decay, reshape=(1, 1, width))
    earlier = _ew(
        network,
        shuffle(network, k, reshape=(1, -1, width)),
        _ew(network, distance3, decay3, op.PROD),
        op.SUM,
    )
    exists = _ew(
        network, distance3, _const(network, np.full((1, 1, 1), -0.5, np.float32)), op.GREATER
    )
    earlier = network.add_select(
        exists, earlier, _const(network, np.full((1, 1, 1), _ABSENT, np.float32))
    ).get_output(0)

    # Incoming state and bonus: [S + 1, width]; the state row S has no bonus.
    incoming = _ew(network, max_state, _ew(network, steps, time_decay, op.PROD), op.SUM)
    bonus = _append_row(network, _ew(network, time_first, k, op.SUM), _ABSENT, width)
    bonus_value = _append_row(network, v, 0.0, width)

    top = network.add_reduce(earlier, trt.ReduceOperation.MAX, 1 << 1, keep_dims=False).get_output(
        0
    )
    top = _ew(network, top, _ew(network, incoming, bonus, op.MAX), op.MAX)

    weight = network.add_unary(
        _ew(network, earlier, shuffle(network, top, reshape=(0, 1, width)), op.SUB),
        trt.UnaryOperation.EXP,
    ).get_output(0)
    value3 = shuffle(network, v, reshape=(1, -1, width))
    num = network.add_reduce(
        _ew(network, weight, value3, op.PROD), trt.ReduceOperation.SUM, 1 << 1, keep_dims=False
    ).get_output(0)
    den = network.add_reduce(weight, trt.ReduceOperation.SUM, 1 << 1, keep_dims=False).get_output(0)

    exp_bonus = network.add_unary(
        _ew(network, bonus, top, op.SUB), trt.UnaryOperation.EXP
    ).get_output(0)
    exp_state = network.add_unary(
        _ew(network, incoming, top, op.SUB), trt.UnaryOperation.EXP
    ).get_output(0)
    num = _ew(network, num, _ew(network, exp_bonus, bonus_value, op.PROD), op.SUM)
    num = _ew(network, num, _ew(network, exp_state, num_state, op.PROD), op.SUM)
    den = _ew(network, den, exp_bonus, op.SUM)
    den = _ew(network, den, _ew(network, exp_state, den_state, op.PROD), op.SUM)

    wkv = _ew(
        network,
        dynamic_slice(network, num, (0, 0), (rows, width)),
        dynamic_slice(network, den, (0, 0), (rows, width)),
        op.DIV,
    )
    new_state = [
        dynamic_slice(network, tensor, (rows, 0), (1, width)) for tensor in (num, den, top)
    ]
    return wkv, new_state[0], new_state[1], new_state[2]
