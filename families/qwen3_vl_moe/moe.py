# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routed SwiGLU experts with the official router and accumulation boundaries."""

import numpy as np
import tensorrt as trt
from . import graph_ops


def add_moe(network, hidden, weights, prefix, *, top_k, dtype):
    width = hidden.shape[-1]
    gate = weights[prefix + ".experts.w_gate"]
    expert_count = gate.shape[0]
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden, width, expert_count, weights[prefix + ".router"], dtype=dtype
    )
    # HF normalizes in FP32, then rounds router probabilities to the work dtype.
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    probability = network.add_softmax(logits)
    probability.axes = 1 << 1
    top = network.add_topk(probability.get_output(0), trt.TopKOperation.MAX, top_k, 1 << 1)
    values, indices = top.get_output(0), top.get_output(1)
    total = network.add_reduce(values, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    normalized = network.add_elementwise(values, total, trt.ElementWiseOperation.DIV).get_output(0)
    if normalized.dtype != hidden.dtype:
        normalized = network.add_cast(normalized, hidden.dtype).get_output(0)

    def constant(value, storage=dtype, runtime=hidden.dtype):
        tensor = graph_ops.add_constant(network, value.shape, value, dtype=storage)
        return (
            tensor if tensor.dtype == runtime else network.add_cast(tensor, runtime).get_output(0)
        )

    reshaped = network.add_shuffle(hidden)
    reshaped.reshape_dims = (1, -1, width)

    # TensorRT broadcasts the input batch over expert weight matrices.
    def matmul(lhs, rhs):
        return network.add_matrix_multiply(
            lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
        ).get_output(0)

    projected_gate = matmul(reshaped.get_output(0), constant(gate))
    projected_up = matmul(reshaped.get_output(0), constant(weights[prefix + ".experts.w_up"]))
    sigmoid = network.add_activation(projected_gate, trt.ActivationType.SIGMOID).get_output(0)
    swish = network.add_elementwise(
        projected_gate, sigmoid, trt.ElementWiseOperation.PROD
    ).get_output(0)
    activation = network.add_elementwise(
        swish, projected_up, trt.ElementWiseOperation.PROD
    ).get_output(0)
    output = matmul(activation, constant(weights[prefix + ".experts.w_down"]))

    # The reference visits experts in ascending ID order and updates the BF16
    # accumulator after each contribution. Gather the selected experts in that
    # order rather than reducing a dense expert axis in FP32.
    sort = network.add_topk(
        network.add_cast(indices, trt.float32).get_output(0),
        trt.TopKOperation.MIN,
        top_k,
        1 << 1,
    )
    sorted_ids = network.add_cast(sort.get_output(0), trt.int32).get_output(0)
    probabilities = network.add_gather(normalized, sort.get_output(1), 1)
    probabilities.num_elementwise_dims = 1
    scale = network.add_shuffle(probabilities.get_output(0))
    scale.reshape_dims = (0, top_k, 1)
    transpose = network.add_shuffle(output)
    transpose.first_transpose = (1, 0, 2)
    selected = network.add_gather(transpose.get_output(0), sorted_ids, 1)
    selected.num_elementwise_dims = 1
    weighted_output = network.add_elementwise(
        selected.get_output(0), scale.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    accumulated = None
    for offset in range(top_k):
        index = constant(np.array([offset], dtype=np.int32), np.int32, trt.int32)
        contribution = network.add_gather(weighted_output, index, 1).get_output(0)
        accumulated = (
            contribution
            if accumulated is None
            else network.add_elementwise(
                accumulated, contribution, trt.ElementWiseOperation.SUM
            ).get_output(0)
        )
    result = network.add_shuffle(accumulated)
    result.reshape_dims = (-1, width)
    return result.get_output(0)
