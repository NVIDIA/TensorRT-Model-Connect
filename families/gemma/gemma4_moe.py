# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 routed GELU experts expressed with ordinary TensorRT operators."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def expert_geometry(raw):
    experts = raw.get("num_experts")
    selected = raw.get("top_k_experts")
    width = raw.get("moe_intermediate_size")
    if (any(type(x) is not int for x in (experts, selected, width))
            or not 0 < selected <= experts or width <= 0):
        raise ValueError("Gemma4 routed expert geometry is invalid")
    return experts, selected, width


def add_experts(graph, residual, dense_output, prefix):
    """Combine independently normalized dense and routed branches.

    A TensorRT loop accumulates experts in source order and bounds decoded
    weight storage to one expert. TensorRT owns execution and fusion; this
    graph has no sparse GPU helper, SKU rule, or sparse-performance claim.
    """
    n, raw = graph.net, graph.raw
    experts, selected, width = expert_geometry(raw)
    hidden = raw["hidden_size"]

    # Routing uses the post-attention residual, not the dense MLP's norm.
    router = graph.norm(residual)
    router = graph.op(router, graph.reshape(graph.weight(prefix + ".router.scale"), (1, hidden)),
                      trt.ElementWiseOperation.PROD)
    router = graph.scale(router, hidden**-0.5)
    scores = graph.cast(graph.linear(router, prefix + ".router.proj"), trt.float32)
    probabilities = n.add_softmax(scores)
    probabilities.axes = 1 << 1
    top = n.add_topk(probabilities.get_output(0), trt.TopKOperation.MAX, selected, 1 << 1)
    weights, indices = top.get_output(0), top.get_output(1)
    total = n.add_reduce(weights, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    weights = graph.op(weights, total, trt.ElementWiseOperation.DIV)
    scales = n.add_gather(graph.weight(prefix + ".router.per_expert_scale", dtype=trt.float32),
                          indices, 0).get_output(0)
    weights = graph.op(weights, scales, trt.ElementWiseOperation.PROD)

    normalized = graph.norm(residual, prefix + ".pre_feedforward_layernorm_2")
    loop = n.add_loop()
    loop.add_trip_limit(graph.reshape(graph.const(np.array([experts], np.int32)), ()),
                        trt.TripLimit.COUNT)
    expert_id = loop.add_iterator(graph.const(np.arange(experts, dtype=np.int32)),
                                  0, False).get_output(0)
    from .checkpoint_mapper import _has_tensor

    if _has_tensor(graph.readers, f"{graph.prefix}.{prefix}.experts.gate_up_proj"):
        gate_up = loop.add_iterator(graph.weight(prefix + ".experts.gate_up_proj"),
                                    0, False).get_output(0)
        down = loop.add_iterator(graph.weight(prefix + ".experts.down_proj"),
                                 0, False).get_output(0)
    else:
        from .gemma4_quant import expert_tensor

        gate_up = expert_tensor(graph, prefix, "gate_up_proj", loop)
        down = expert_tensor(graph, prefix, "down_proj", loop)
    if gate_up.shape != (2 * width, hidden) or down.shape != (hidden, width):
        raise ValueError("Gemma4 expert weight shape disagrees with its configuration")
    # Explicit FP32 accumulation preserves the rounded source weights.
    # The FP16 accumulator differed despite bit-identical decoded weights;
    # this boundary restores same-precision reference parity without clamps.
    projected = n.add_matrix_multiply(graph.cast(normalized, trt.float32), trt.MatrixOperation.NONE,
                                      graph.cast(gate_up, trt.float32),
                                      trt.MatrixOperation.TRANSPOSE).get_output(0)
    projected = graph.cast(projected, graph.dtype)
    gate = n.add_gather(projected, graph.const(np.arange(width, dtype=np.int32)), 1).get_output(0)
    up = n.add_gather(projected, graph.const(np.arange(width, 2 * width, dtype=np.int32)),
                       1).get_output(0)
    # Preserve the same nonlinear-product/down-projection FP32 boundary as
    # the dense Gemma4 MLP, without clipping the original model's values.
    product = graph.op(graph.activation(graph.cast(gate, trt.float32)),
                       graph.cast(up, trt.float32), trt.ElementWiseOperation.PROD)
    outputs = n.add_matrix_multiply(product, trt.MatrixOperation.NONE,
                                    graph.cast(down, trt.float32),
                                    trt.MatrixOperation.TRANSPOSE).get_output(0)
    chosen = graph.op(graph.reshape(expert_id, (1, 1)), indices,
                      trt.ElementWiseOperation.EQUAL)
    selected_weights = n.add_select(chosen, weights,
                                    graph.const(np.zeros((1, 1), np.float32))).get_output(0)
    coefficients = n.add_reduce(selected_weights, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    weighted = graph.op(outputs, coefficients, trt.ElementWiseOperation.PROD)
    # Do not let an unused expert's overflow poison selected expert output.
    active = n.add_reduce(graph.cast(chosen, trt.int32), trt.ReduceOperation.MAX,
                          1 << 1, True).get_output(0)
    weighted = n.add_select(graph.cast(active, trt.bool), weighted,
                            graph.const(np.zeros((1, 1), np.float32))).get_output(0)
    # Match the source's index_add into compute-precision activations, in
    # ascending expert order, rather than introducing a new reduction order.
    accumulation = loop.add_recurrence(graph.scale(normalized, 0.0))
    updated = graph.op(accumulation.get_output(0), graph.cast(weighted, graph.dtype),
                        trt.ElementWiseOperation.SUM)
    accumulation.set_input(1, updated)
    combined = loop.add_loop_output(accumulation.get_output(0),
                                    trt.LoopOutput.LAST_VALUE, 0).get_output(0)
    routed = graph.norm(combined, prefix + ".post_feedforward_layernorm_2")
    dense = graph.norm(dense_output, prefix + ".post_feedforward_layernorm_1")
    return graph.op(dense, graph.cast(routed, dense.dtype), trt.ElementWiseOperation.SUM)
