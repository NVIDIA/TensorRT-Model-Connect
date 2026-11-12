# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet mixer over ``S`` rows in one launch (chunked delta rule).

The single-token decoder applies, per head and token ``t``::

    S_t = exp(g_t) S_{t-1} + k_t u_t^T,   u_t = beta_t (v_t - exp(g_t) S_{t-1}^T k_t)
    o_t = S_t^T q_t

For a chunk of ``S`` tokens with cumulative decay ``gamma_t = sum_{i<=t} g_i``
and ``Gamma_t = exp(gamma_t)`` this is equivalent to::

    (I + A) U = beta V - diag(beta Gamma) K S_0,
        A_ij = beta_i exp(gamma_i - gamma_j) k_i.k_j   for j < i, else 0
    O   = diag(Gamma) Q S_0 + (exp(gamma_t - gamma_i) q_t.k_i)_{i<=t} U
    S_S = exp(gamma_S) S_0 + (K * exp(gamma_S - gamma))^T U

``I + A`` is unit lower triangular. It is inverted by 2x2 block inversion on
a fixed ``SOLVE_ROWS`` padding (the padded rows are identity), batched over
the blocks of each level. This stays accurate where a single Neumann product
over 32 or more rows does not (nearly identical keys with beta close to one).
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

# Static size of the padded triangular solve; it bounds the prefill rows.
SOLVE_ROWS = 64
# Diagonal blocks of this size are inverted with a Neumann product.
_SOLVE_LEAF = 8


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


def _shuffle(network, tensor, reshape=None, transpose=None) -> trt.ITensor:
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


def _block_rows(network, matrix, heads: int, blocks: int, size: int, pairs: np.ndarray):
    """Blocks ``(r, c)`` of ``[heads, blocks*size, blocks*size]`` with ``pairs[r, c] = 1``, summed over c.

    Returns ``[heads, blocks, size, size]``: row block ``r`` holds the selected block of that row.
    """
    split = _shuffle(network, matrix, reshape=(heads, blocks, size, blocks, size))
    mask = _const(network, pairs.astype(np.float32).reshape(1, blocks, 1, blocks, 1))
    selected = _ew(network, split, mask, trt.ElementWiseOperation.PROD)
    return network.add_reduce(
        selected, trt.ReduceOperation.SUM, 1 << 3, keep_dims=False
    ).get_output(0)


def _pair_half(network, tensor, heads: int, pairs: int, size: int, index: int) -> trt.ITensor:
    """``[heads, 2*pairs, size, size]`` -> block ``index`` (0 or 1) of each pair."""
    grouped = _shuffle(network, tensor, reshape=(heads, pairs, 2, size, size))
    half = network.add_slice(
        grouped, (0, 0, index, 0, 0), (heads, pairs, 1, size, size), (1, 1, 1, 1, 1)
    ).get_output(0)
    return _shuffle(network, half, reshape=(heads, pairs, size, size))


def _inverse_unit_lower(network, lower: trt.ITensor, heads: int, size: int) -> trt.ITensor:
    """Inverse of a static ``[heads, size, size]`` unit lower-triangular matrix.

    All ``_SOLVE_LEAF`` diagonal blocks are inverted at once with a Neumann
    product, then each level merges pairs of blocks with
    ``[[T11, 0], [-T22 L21 T11, T22]]``, batched over the pairs, so the
    graph has a handful of matmuls per level rather than per block.
    """
    sum_op = trt.ElementWiseOperation.SUM
    leaf = min(_SOLVE_LEAF, size)
    if size % leaf or (size // leaf) & (size // leaf - 1):
        raise ValueError("the triangular solve size must be a power-of-two multiple of the leaf")
    blocks = size // leaf
    diagonal = _block_rows(network, lower, heads, blocks, leaf, np.eye(blocks))
    identity = _const(network, np.eye(leaf, dtype=np.float32)[None, None])
    # (I - N)^-1 = (I + N)(I + N^2)(I + N^4)... with N = I - D nilpotent.
    nilpotent = _ew(network, identity, diagonal, trt.ElementWiseOperation.SUB)
    inverse = _ew(network, identity, nilpotent, sum_op)
    power = nilpotent
    for _ in range(max(0, math.ceil(math.log2(leaf)) - 1)):
        power = _mm(network, power, power)
        inverse = _mm(network, inverse, _ew(network, identity, power, sum_op))
    width = leaf
    while blocks > 1:
        pairs = blocks // 2
        below = np.zeros((blocks, blocks))
        below[np.arange(1, blocks, 2), np.arange(0, blocks, 2)] = 1.0
        lower_left = _pair_half(
            network,
            _block_rows(network, lower, heads, blocks, width, below),
            heads,
            pairs,
            width,
            1,
        )
        upper = _pair_half(network, inverse, heads, pairs, width, 0)
        bottom_right = _pair_half(network, inverse, heads, pairs, width, 1)
        corner = network.add_unary(
            _mm(network, _mm(network, bottom_right, lower_left), upper), trt.UnaryOperation.NEG
        ).get_output(0)
        zeros = _const(network, np.zeros((heads, pairs, width, width), dtype=np.float32))
        top = network.add_concatenation([upper, zeros])
        top.axis = 3
        bottom = network.add_concatenation([corner, bottom_right])
        bottom.axis = 3
        merged = network.add_concatenation([top.get_output(0), bottom.get_output(0)])
        merged.axis = 2
        inverse = merged.get_output(0)
        blocks, width = pairs, width * 2
    return _shuffle(network, inverse, reshape=(heads, size, size))


def add_chunked_delta_rule(
    network: trt.INetworkDefinition,
    *,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    g: trt.ITensor,
    beta: trt.ITensor,
    state: trt.ITensor,
    num_heads: int,
    head_dim: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Gated delta rule over one chunk, all tensors FP32.

    ``q``/``k``/``v`` are head-major ``[H, S, D]`` (``q`` already scaled),
    ``g``/``beta`` are ``[H, S]`` and ``state`` is ``[H, D_k, D_v]``.
    Returns ``(outputs [H, S, D_v], final state [H, D_k, D_v])``.
    """
    sum_op, prod_op = trt.ElementWiseOperation.SUM, trt.ElementWiseOperation.PROD
    rows = row_count(network, _shuffle(network, g, transpose=(1, 0)))

    axis = network.add_constant((), np.array(1, dtype=np.int32)).get_output(0)
    gamma = network.add_cumulative(g, axis, trt.CumulativeOperation.SUM, False, False).get_output(0)
    gamma_rows = _shuffle(network, gamma, reshape=(0, 0, 1))  # [H, S, 1]
    gamma_cols = _shuffle(network, gamma, reshape=(0, 1, -1))  # [H, 1, S]
    # exp(gamma_i - gamma_j) for j <= i; the exponent is clamped at zero so the
    # masked upper triangle cannot overflow before it is multiplied by zero.
    diff = _ew(network, gamma_rows, gamma_cols, trt.ElementWiseOperation.SUB)
    diff = _ew(
        network,
        diff,
        _const(network, np.zeros((1, 1, 1), np.float32)),
        trt.ElementWiseOperation.MIN,
    )
    decay = network.add_unary(diff, trt.UnaryOperation.EXP).get_output(0)

    def mask(offset: int) -> trt.ITensor:
        full = np.tril(np.ones((SOLVE_ROWS, SOLVE_ROWS), np.float32), offset)[None]
        return dynamic_slice(network, _const(network, full), (0, 0, 0), (1, rows, rows))

    decay_strict = _ew(network, decay, mask(-1), prod_op)
    decay_inclusive = _ew(network, decay, mask(0), prod_op)

    beta_rows = _shuffle(network, beta, reshape=(0, 0, 1))  # [H, S, 1]
    a_matrix = _ew(
        network,
        _ew(network, beta_rows, decay_strict, prod_op),
        _mm(network, k, k, transpose_b=True),
        prod_op,
    )
    padded = network.add_slice(a_matrix, (0, 0, 0), (num_heads, SOLVE_ROWS, SOLVE_ROWS), (1, 1, 1))
    padded.mode = trt.SampleMode.FILL
    padded.set_input(4, network.add_constant((), np.array(0.0, dtype=np.float32)).get_output(0))
    lower = _ew(
        network,
        _const(network, np.eye(SOLVE_ROWS, dtype=np.float32)[None]),
        padded.get_output(0),
        sum_op,
    )
    inverse = dynamic_slice(
        network,
        _inverse_unit_lower(network, lower, num_heads, SOLVE_ROWS),
        (0, 0, 0),
        (num_heads, rows, rows),
    )

    gamma_exp = network.add_unary(gamma_rows, trt.UnaryOperation.EXP).get_output(0)  # [H, S, 1]
    history = _mm(network, k, state)  # [H, S, D_v]: S_0^T k_i
    rhs = _ew(
        network,
        _ew(network, beta_rows, v, prod_op),
        _ew(network, _ew(network, beta_rows, gamma_exp, prod_op), history, prod_op),
        trt.ElementWiseOperation.SUB,
    )
    updates = _mm(network, inverse, rhs)  # U: [H, S, D_v]

    scores = _ew(network, decay_inclusive, _mm(network, q, k, transpose_b=True), prod_op)
    outputs = _ew(
        network,
        _ew(network, gamma_exp, _mm(network, q, state), prod_op),
        _mm(network, scores, updates),
        sum_op,
    )

    # gamma_S is the last cumulative sum, not a separate reduction: summed in
    # another order it can exceed gamma_i by many ulps at large |g|, and
    # exp(gamma_S - gamma_i) would then overflow. A float running sum of
    # non-positive terms never increases, and the clamp keeps the exponent at
    # most zero regardless.
    last = _ew(network, rows, _i64(network, [1]), trt.ElementWiseOperation.SUB)
    total = dynamic_slice(network, gamma, (_i64(network, [0]), last), (num_heads, 1))  # [H, 1]
    tail = network.add_unary(
        _ew(
            network,
            _ew(
                network,
                _shuffle(network, total, reshape=(0, 0, 1)),
                gamma_rows,
                trt.ElementWiseOperation.SUB,
            ),
            _const(network, np.zeros((1, 1, 1), np.float32)),
            trt.ElementWiseOperation.MIN,
        ),
        trt.UnaryOperation.EXP,
    ).get_output(0)  # exp(gamma_S - gamma_i): [H, S, 1]
    carried = _ew(
        network,
        network.add_unary(
            _shuffle(network, total, reshape=(0, 1, 1)), trt.UnaryOperation.EXP
        ).get_output(0),
        state,
        prod_op,
    )
    final_state = _ew(
        network,
        carried,
        _mm(network, _ew(network, k, tail, prod_op), updates, transpose_a=True),
        sum_op,
    )
    return outputs, final_state


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
    rows = row_count(network, _shuffle(network, columns, transpose=(1, 0)))
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
    history = dynamic_slice(network, window, (_i64(network, [0]), last), (channels, kernel))
    return total, history
