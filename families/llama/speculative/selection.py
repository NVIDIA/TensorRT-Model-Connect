# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless greedy selection graph; independent of attention lowering.

The separate engine consumes existing FP32 logits by device binding without
changing the target/draft graphs or their state ABI.
"""

import numpy as np
import tensorrt as trt


def build_selection(vocab: int, max_rows: int, ranks: int, *, verbose=False) -> bytes:
    if ranks not in (1, 2) or vocab < ranks or max_rows < 1:
        raise ValueError("selection requires positive rows and one or two vocabulary ranks")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    logits = network.add_input("logits", trt.float32, (-1, vocab))
    profile = builder.create_optimization_profile()
    profile.set_shape("logits", (1, vocab), (min(5, max_rows), vocab), (max_rows, vocab))
    config.add_optimization_profile(profile)

    def constant(value):
        value = np.ascontiguousarray(value)
        return network.add_constant(value.shape, value).get_output(0)

    def binary(a, b, op):
        return network.add_elementwise(a, b, op).get_output(0)

    def reduce(value, op):
        return network.add_reduce(value, op, 1 << 1, True).get_output(0)

    # abs(x) < inf rejects NaN and either infinity. Keep status per row: only
    # rows consumed by the policy need to be finite.
    absolute = network.add_unary(logits, trt.UnaryOperation.ABS).get_output(0)
    finite = binary(absolute, constant(np.array([[np.inf]], np.float32)),
                    trt.ElementWiseOperation.LESS)
    finite = network.add_cast(finite, trt.int32).get_output(0)
    finite = reduce(finite, trt.ReduceOperation.MIN)
    indices = constant(np.arange(vocab, dtype=np.int32).reshape(1, vocab))
    sentinel = constant(np.array([[vocab]], np.int32))
    negative_inf = constant(np.array([[-np.inf]], np.float32))
    current = logits
    ids = []
    for _ in range(ranks):
        maximum = reduce(current, trt.ReduceOperation.MAX)
        tied = binary(current, maximum, trt.ElementWiseOperation.EQUAL)
        candidates = network.add_select(tied, indices, sentinel).get_output(0)
        best = reduce(candidates, trt.ReduceOperation.MIN)
        ids.append(best)
        # Explicit lowest-index tie breaking, including signed zero. TopK's
        # tie order is not part of this contract.
        chosen = binary(indices, best, trt.ElementWiseOperation.EQUAL)
        current = network.add_select(chosen, negative_inf, current).get_output(0)
    output = network.add_concatenation(ids + [finite])
    output.axis = 1
    output.get_output(0).name = "selection"
    network.mark_output(output.get_output(0))
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("failed to build greedy selection engine")
    return bytes(plan)


def add_selection_plans(writer, target, draft, *, verbose=False):
    for role, contract, ranks in (("target", target, 1), ("draft", draft, 2)):
        rows = (max(p.logits[2] for p in contract.execution_profiles)
                if contract.execution_profiles else contract.max_query)
        writer.add_bytes(f"{role}_selection.plan",
                         build_selection(contract.vocab_size, rows, ranks, verbose=verbose))
