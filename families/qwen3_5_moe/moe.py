# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned softmax top-k routed/shared experts in standard TensorRT."""

from dataclasses import dataclass

import ml_dtypes
import numpy as np
import tensorrt as trt

from .quantization import PackedLinear


@dataclass(frozen=True)
class ExpertWeights:
    """Input-major original matrices or original packed projections."""

    gate: np.ndarray | PackedLinear
    up: np.ndarray | PackedLinear
    down: np.ndarray | PackedLinear


@dataclass(frozen=True)
class MoEWeights:
    router: np.ndarray
    experts: tuple[ExpertWeights, ...]
    shared: ExpertWeights
    shared_gate: np.ndarray
    top_k: int


def add_sparse_moe(network, hidden, weights: MoEWeights):
    """Preserve top-k routing and source precision with compact selected experts.

    Gather only the selected experts and accumulate in ascending expert order.
    Packed source bytes remain packed until selection; TensorRT owns execution.
    Mixed representations retain the individual projection graph.
    """
    hidden_size, count = weights.router.shape
    if (
        count != len(weights.experts)
        or not 1 <= weights.top_k <= count
        or weights.shared_gate.shape != (hidden_size, 1)
    ):
        raise ValueError(
            "Sparse expert dimensions differ from the hidden/router contract"
        )
    for item in (*weights.experts, weights.shared):
        if (
            item.gate.shape != item.up.shape
            or item.down.shape != item.gate.shape[::-1]
            or item.gate.shape[0] != hidden_size
        ):
            raise ValueError("Gated expert projection dimensions disagree")
    for name in ("gate", "up", "down"):
        matrices = [getattr(item, name) for item in weights.experts]
        original = all(isinstance(matrix, np.ndarray) for matrix in matrices)
        packed = all(
            isinstance(matrix, PackedLinear)
            and matrix.format in {"gptq", "nvfp4_a16"}
            and matrix.format == matrices[0].format
            for matrix in matrices
        )
        if not (original or packed):
            return _add_sparse_moe_individual(network, hidden, weights)
    return _add_sparse_moe_selected(network, hidden, weights)


def _add_sparse_moe_selected(network, hidden, weights: MoEWeights):
    """Gather compact integer storage, then decode the selected original weights."""
    dtype = hidden.dtype
    hidden_size, count = weights.router.shape
    topk = weights.top_k

    def cast(x, d):
        return x if x.dtype == d else network.add_cast(x, d).get_output(0)

    def const(x):
        x = np.array(x, copy=True, order="C")
        return network.add_constant(x.shape, x).get_output(0)

    def binary(a, b, op):
        return network.add_elementwise(a, b, op).get_output(0)

    def reshape(x, shape):
        op = network.add_shuffle(x)
        op.reshape_dims = shape
        return op.get_output(0)

    def linear(x, w):
        if isinstance(w, PackedLinear):
            return w.matmul(network, x, *w.shape)
        return network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, cast(const(w), dtype), trt.MatrixOperation.NONE
        ).get_output(0)

    def activated(x):
        x = cast(x, trt.float32)
        return cast(
            binary(
                x,
                network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0),
                trt.ElementWiseOperation.PROD,
            ),
            dtype,
        )

    router = network.add_matrix_multiply(
        cast(hidden, trt.float32),
        trt.MatrixOperation.NONE,
        cast(cast(const(weights.router), dtype), trt.float32),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    logits = cast(cast(router, dtype), trt.float32)
    sm = network.add_softmax(logits)
    sm.axes = 1 << 1
    top = network.add_topk(sm.get_output(0), trt.TopKOperation.MAX, topk, 1 << 1)
    probs, indices = top.get_output(0), top.get_output(1)
    denom = network.add_reduce(probs, trt.ReduceOperation.SUM, 1 << 1, True).get_output(
        0
    )
    probs = cast(binary(probs, denom, trt.ElementWiseOperation.DIV), dtype)
    sorted_ = network.add_topk(
        cast(indices, trt.float32), trt.TopKOperation.MIN, topk, 1 << 1
    )
    selected = cast(sorted_.get_output(0), trt.int32)
    matches = binary(
        reshape(selected, (-1, topk, 1)),
        reshape(indices, (-1, 1, topk)),
        trt.ElementWiseOperation.EQUAL,
    )
    scores = network.add_select(
        matches,
        reshape(probs, (-1, 1, topk)),
        cast(const(np.zeros((1, 1, 1), np.float32)), dtype),
    ).get_output(0)
    scores = network.add_reduce(
        scores, trt.ReduceOperation.SUM, 1 << 2, True
    ).get_output(0)

    def selected_projection(value, matrices):
        k, n = matrices[0].shape
        if not isinstance(matrices[0], PackedLinear):
            rhs = cast(const(np.stack([m.T for m in matrices])), dtype)
            rhs = network.add_gather(rhs, selected, 0).get_output(0)
        else:
            fmt = matrices[0].format
            if fmt not in ("gptq", "nvfp4_a16") or any(
                m.format != fmt or m.shape != (k, n) for m in matrices
            ):
                raise ValueError(
                    "Selected experts require matching packed precision and geometry"
                )
            # Integer words retain the exact compact bytes without claiming an
            # INT8 quantized floating-point tensor (which TRT restricts to DQ).
            packed = np.stack([m.packed for m in matrices]).view("<i4")
            words = network.add_gather(const(packed), selected, 0).get_output(0)
            sixteen = const(np.full((1, 1, 1, 1), 16, np.int32))
            digits = []
            for offset in range(8):
                shifted = (
                    words
                    if offset == 0
                    else binary(
                        words,
                        const(np.full((1, 1, 1, 1), 1 << (4 * offset), np.int32)),
                        trt.ElementWiseOperation.FLOOR_DIV,
                    )
                )
                high = binary(shifted, sixteen, trt.ElementWiseOperation.FLOOR_DIV)
                digit = binary(
                    shifted,
                    binary(high, sixteen, trt.ElementWiseOperation.PROD),
                    trt.ElementWiseOperation.SUB,
                )
                if fmt == "gptq":
                    sign = binary(
                        digit,
                        const(np.full((1, 1, 1, 1), 7, np.int32)),
                        trt.ElementWiseOperation.GREATER,
                    )
                    digit = network.add_select(
                        sign,
                        binary(digit, sixteen, trt.ElementWiseOperation.SUB),
                        digit,
                    ).get_output(0)
                digits.append(reshape(digit, (-1, topk, n, k // 8, 1)))
            interleave = network.add_concatenation(digits)
            interleave.axis = 4
            decoded = reshape(interleave.get_output(0), (-1, topk, n, k))
            group = 128 if fmt == "gptq" else 16
            if fmt == "gptq":
                scales = const(np.stack([m.scales for m in matrices]))
                decoded = cast(decoded, dtype)
            else:
                levels = const(
                    np.array(
                        [
                            0,
                            0.5,
                            1,
                            1.5,
                            2,
                            3,
                            4,
                            6,
                            -0.0,
                            -0.5,
                            -1,
                            -1.5,
                            -2,
                            -3,
                            -4,
                            -6,
                        ],
                        np.float32,
                    )
                )
                decoded = network.add_gather(levels, decoded, 0).get_output(0)
                sf = (
                    np.stack([m.scales for m in matrices])
                    .view(ml_dtypes.float8_e4m3fn)
                    .astype(np.float32)
                )
                # Match the source FP32 block-scale/global-scale multiplication order.
                sf = sf * np.asarray(
                    [m.global_scale for m in matrices], np.float32
                ).reshape(count, 1, 1)
                scales = const(sf)
            scales = network.add_gather(scales, selected, 0).get_output(0)
            decoded = reshape(decoded, (-1, topk, n, k // group, group))
            scales = reshape(scales, (-1, topk, n, k // group, 1))
            rhs = reshape(
                cast(
                    binary(
                        decoded,
                        cast(scales, decoded.dtype),
                        trt.ElementWiseOperation.PROD,
                    ),
                    dtype,
                ),
                (-1, topk, n, k),
            )
        return network.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    rows = reshape(hidden, (-1, 1, 1, hidden_size))
    gate = selected_projection(rows, [e.gate for e in weights.experts])
    up = selected_projection(rows, [e.up for e in weights.experts])
    contribution = selected_projection(
        binary(activated(gate), up, trt.ElementWiseOperation.PROD),
        [e.down for e in weights.experts],
    )
    contribution = binary(
        reshape(contribution, (-1, topk, hidden_size)),
        scores,
        trt.ElementWiseOperation.PROD,
    )
    routed = binary(
        hidden,
        cast(const(np.zeros((1, 1), np.float32)), dtype),
        trt.ElementWiseOperation.PROD,
    )
    for i in range(topk):
        piece = network.add_gather(
            contribution, const(np.asarray(i, np.int32)), 1
        ).get_output(0)
        routed = binary(routed, piece, trt.ElementWiseOperation.SUM)
    gate = linear(hidden, weights.shared.gate)
    up = linear(hidden, weights.shared.up)
    shared = linear(
        binary(activated(gate), up, trt.ElementWiseOperation.PROD), weights.shared.down
    )
    shared_gate = cast(
        network.add_activation(
            cast(linear(hidden, weights.shared_gate), trt.float32),
            trt.ActivationType.SIGMOID,
        ).get_output(0),
        dtype,
    )
    return binary(
        routed,
        binary(shared, shared_gate, trt.ElementWiseOperation.PROD),
        trt.ElementWiseOperation.SUM,
    )


def _add_sparse_moe_individual(network, hidden, weights: MoEWeights):
    """Return MoE output for rank-two token rows, with the original compute dtype.

    Routing normalizes FP32 probabilities after the compute-precision router
    projection, then restores compute precision before weighting each expert.
    Expert accumulation follows ascending expert index, as in the source loop.
    Selection is expressed as tensor arithmetic rather than conditional network
    execution. All expert projections remain in the graph; TensorRT owns their
    lowering and scheduling. No packed weight or source quantizer is replaced.
    """
    dtype = hidden.dtype
    hidden_size, num_experts = weights.router.shape
    # Rank-two token rows are established by the owning decoder.
    if (
        num_experts != len(weights.experts)
        or not 1 <= weights.top_k <= num_experts
        or weights.shared_gate.shape != (hidden_size, 1)
    ):
        raise ValueError(
            "Sparse expert dimensions differ from the hidden/router contract"
        )

    def cast(value, target):
        return (
            value
            if value.dtype == target
            else network.add_cast(value, target).get_output(0)
        )

    def constant(values):
        values = np.array(values, copy=True, order="C")
        return network.add_constant(values.shape, values).get_output(0)

    def binary(a, b, op):
        return network.add_elementwise(a, b, op).get_output(0)

    def linear(value, weight):
        if isinstance(weight, PackedLinear):
            return weight.matmul(network, value, *weight.shape)
        weight = np.asarray(weight)
        if weight.ndim != 2:
            raise ValueError("Expert projection must be a matrix")
        rhs = cast(constant(weight), value.dtype)
        return network.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
        ).get_output(0)

    def expert(value, item):
        if item.gate.shape != item.up.shape or item.down.shape != item.gate.shape[::-1]:
            raise ValueError("Gated expert projection dimensions disagree")
        gate, up = linear(value, item.gate), linear(value, item.up)
        # Source SiLU computes internally in FP32 before its FP16 result.
        gate32 = cast(gate, trt.float32)
        sigmoid = network.add_activation(gate32, trt.ActivationType.SIGMOID).get_output(
            0
        )
        activated = cast(binary(gate32, sigmoid, trt.ElementWiseOperation.PROD), dtype)
        return linear(binary(activated, up, trt.ElementWiseOperation.PROD), item.down)

    logits = cast(linear(hidden, weights.router), trt.float32)
    softmax = network.add_softmax(logits)
    softmax.axes = 1 << 1
    top = network.add_topk(
        softmax.get_output(0), trt.TopKOperation.MAX, weights.top_k, 1 << 1
    )
    probabilities, indices = top.get_output(0), top.get_output(1)
    denominator = network.add_reduce(
        probabilities, trt.ReduceOperation.SUM, 1 << 1, True
    ).get_output(0)
    probabilities = cast(
        binary(probabilities, denominator, trt.ElementWiseOperation.DIV), dtype
    )
    zero = cast(constant(np.zeros((1, 1), np.float32)), dtype)
    zero_row = cast(constant(np.zeros((1, hidden_size), np.float32)), dtype)
    empty_rows = binary(hidden, zero, trt.ElementWiseOperation.PROD)
    routed = empty_rows
    for number, item in enumerate(weights.experts):
        matches = binary(
            indices,
            constant(np.asarray([[number]], np.int32)),
            trt.ElementWiseOperation.EQUAL,
        )
        count = network.add_reduce(
            cast(matches, trt.int32), trt.ReduceOperation.SUM, 1 << 1, True
        ).get_output(0)
        member = binary(
            count,
            constant(np.zeros((1, 1), np.int32)),
            trt.ElementWiseOperation.GREATER,
        )
        scores = network.add_select(matches, probabilities, zero).get_output(0)
        scores = network.add_reduce(
            scores, trt.ReduceOperation.SUM, 1 << 1, True
        ).get_output(0)
        masked = network.add_select(member, hidden, zero_row).get_output(0)
        contribution = expert(masked, item)
        # Select, rather than multiply, so unused overflow cannot produce inf*0.
        contribution = network.add_select(member, contribution, zero_row).get_output(0)
        contribution = binary(contribution, scores, trt.ElementWiseOperation.PROD)
        routed = binary(routed, contribution, trt.ElementWiseOperation.SUM)
    shared = expert(hidden, weights.shared)
    gate = cast(linear(hidden, weights.shared_gate), trt.float32)
    gate = cast(
        network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0), dtype
    )
    return binary(
        routed,
        binary(shared, gate, trt.ElementWiseOperation.PROD),
        trt.ElementWiseOperation.SUM,
    )
