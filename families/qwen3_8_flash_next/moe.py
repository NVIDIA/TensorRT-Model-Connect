# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flash-Next sparse MoE stages surrounding the selected-expert execution."""

import numpy as np
import tensorrt as trt

from .residual import inject_block, read_for_block


def route_experts(network, hyper, checkpoint, layer, eps):
    """Return token rows, ten distinct indices and weights, shared output and stream gates."""
    n, ew, mat = network, trt.ElementWiseOperation, trt.MatrixOperation
    prefix = f"model.language_model.layers.{layer}"
    hidden, inject = read_for_block(n, hyper, checkpoint, prefix + ".mlp_hyper_connection", eps)

    def linear(x, name, shape):
        w = checkpoint.bf16(f"{prefix}.mlp.{name}.weight", shape)
        weights = n.add_constant(shape, np.ascontiguousarray(w)).get_output(0)
        inputs = n.add_cast(x, trt.float32).get_output(0)
        weights = n.add_cast(weights, trt.float32).get_output(0)
        y = n.add_matrix_multiply(inputs, mat.NONE, weights, mat.TRANSPOSE)
        return n.add_cast(y.get_output(0), trt.float16).get_output(0)

    def mul(x, y):
        return n.add_elementwise(x, y, ew.PROD).get_output(0)

    def sigmoid(x):
        return n.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)

    logits = linear(hidden, "gate", (512, 2560))
    probs = n.add_softmax(n.add_cast(logits, trt.float32).get_output(0))
    probs.axes = 1 << 1
    top = n.add_topk(probs.get_output(0), trt.TopKOperation.MAX, 10, 1 << 1)
    values = top.get_output(0)
    z = n.add_reduce(values, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    weights = n.add_cast(
        n.add_elementwise(values, z, ew.DIV).get_output(0), trt.float16
    ).get_output(0)

    up = linear(hidden, "shared_expert.up_proj", (640, 2560))
    gate = n.add_cast(
        linear(hidden, "shared_expert.gate_proj", (640, 2560)), trt.float32
    ).get_output(0)
    silu = n.add_cast(mul(gate, sigmoid(gate)), trt.float16).get_output(0)
    shared = linear(mul(silu, up), "shared_expert.down_proj", (2560, 640))
    shared = mul(shared, sigmoid(linear(hidden, "shared_expert_gate", (1, 2560))))
    return hidden, top.get_output(1), weights, shared, inject


def combine_experts(network, hyper, inject, shared, coefficients, expert_outputs):
    """Combine the selected paths in token/rank order, then update all four streams."""
    n, ew = network, trt.ElementWiseOperation
    if tuple(expert_outputs.shape)[1:] != (10, 2560) or tuple(coefficients.shape)[1:] != (10,):
        raise ValueError("Invalid Flash-Next selected expert dimensions")
    sh = n.add_shuffle(coefficients)
    sh.reshape_dims = (0, 10, 1)
    weighted = n.add_elementwise(expert_outputs, sh.get_output(0), ew.PROD).get_output(0)
    acc = n.add_reduce(
        n.add_cast(weighted, trt.float32).get_output(0), trt.ReduceOperation.SUM, 1 << 1, False
    )
    routed = n.add_cast(acc.get_output(0), trt.float16).get_output(0)
    mixed = n.add_elementwise(routed, shared, ew.SUM).get_output(0)
    return inject_block(n, hyper, mixed, inject)
