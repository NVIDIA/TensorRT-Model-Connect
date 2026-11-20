# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H sigmoid routing and non-gated experts, expressed in TensorRT."""

import numpy as np
import tensorrt as trt

from . import graph_ops
from .prepared_quantization import PreparedLinear


def add_moe_layer(network, hidden, eps_tensor, *, weights, prefix, raw, dtype, quant_ctx):
    """One recurrent token: grouped top-k, conditional experts, shared MLP, residual.

    TensorRT owns execution of the conditional expert networks. The family
    expresses routing semantics, not a target-specific kernel or schedule.
    Routing and accumulation remain FP32 as in the source implementation.
    """
    hidden_size = int(raw["hidden_size"])
    experts = int(raw["n_routed_experts"])
    topk = int(raw["num_experts_per_tok"])
    groups = int(raw["n_group"])
    selected_groups = int(raw["topk_group"])
    width = int(raw["moe_intermediate_size"])
    shared_width = int(raw["moe_shared_expert_intermediate_size"])
    latent = raw.get("moe_latent_size")
    routed_width = hidden_size if latent is None else int(latent)
    if (
        experts % groups
        or not 1 <= selected_groups <= groups
        or not 1 <= topk <= selected_groups * (experts // groups)
        or raw.get("n_shared_experts", 1) != 1
    ):
        raise ValueError("Invalid Nemotron-H expert routing dimensions")
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    def constant(value, np_dtype=np.float32):
        value = np.asarray(value, dtype=np_dtype)
        return graph_ops.add_constant(network, value.shape, value, dtype=np_dtype)

    def reshape(value, shape):
        layer = network.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def binary(left, right, operation):
        return network.add_elementwise(left, right, operation).get_output(0)

    router_input = normed
    if router_input.dtype != trt.float32:
        router_input = network.add_cast(router_input, trt.float32).get_output(0)
    logits = graph_ops.add_matmul_rhs_constant(
        network, router_input, hidden_size, experts, weights[f"{prefix}.router"], dtype=np.float32, rank=2
    )
    scores = network.add_activation(logits, trt.ActivationType.SIGMOID).get_output(0)
    choice = binary(
        scores,
        constant(weights[f"{prefix}.router_bias"].reshape(1, experts)),
        trt.ElementWiseOperation.SUM,
    )
    if selected_groups != groups:
        per_group = experts // groups
        if per_group < 2:
            raise ValueError("Nemotron-H grouped routing requires at least two experts per group")
        grouped = reshape(choice, (groups, per_group))
        best_two = network.add_topk(grouped, trt.TopKOperation.MAX, 2, 1 << 1).get_output(0)
        group_scores = network.add_reduce(
            best_two, trt.ReduceOperation.SUM, 1 << 1, False
        ).get_output(0)
        group_indices = network.add_topk(
            reshape(group_scores, (1, groups)), trt.TopKOperation.MAX, selected_groups, 1 << 1
        ).get_output(1)
        matches = binary(
            constant(np.arange(groups).reshape(groups, 1), np.int32),
            reshape(group_indices, (1, selected_groups)),
            trt.ElementWiseOperation.EQUAL,
        )
        matches = network.add_cast(matches, trt.float32).get_output(0)
        chosen = network.add_reduce(matches, trt.ReduceOperation.MAX, 1 << 1, True).get_output(0)
        mask = binary(chosen, constant(np.ones((1, per_group))), trt.ElementWiseOperation.PROD)
        # The reference masks excluded groups to zero, not minus infinity.
        choice = binary(choice, reshape(mask, (1, experts)), trt.ElementWiseOperation.PROD)
    indices = network.add_topk(choice, trt.TopKOperation.MAX, topk, 1 << 1).get_output(1)
    matches = binary(
        constant(np.arange(experts).reshape(experts, 1), np.int32),
        indices,
        trt.ElementWiseOperation.EQUAL,
    )
    membership = network.add_cast(matches, trt.float32).get_output(0)
    membership = network.add_reduce(membership, trt.ReduceOperation.MAX, 1 << 1, True).get_output(0)
    routed_scores = binary(scores, reshape(membership, (1, experts)), trt.ElementWiseOperation.PROD)
    if raw.get("norm_topk_prob", True):
        denominator = network.add_reduce(
            routed_scores, trt.ReduceOperation.SUM, 1 << 1, True
        ).get_output(0)
        denominator = binary(denominator, constant([[1e-20]]), trt.ElementWiseOperation.SUM)
        routed_scores = binary(routed_scores, denominator, trt.ElementWiseOperation.DIV)
    routed_scores = binary(
        routed_scores,
        constant([[raw.get("routed_scaling_factor", 1.0)]]),
        trt.ElementWiseOperation.PROD,
    )

    def matmul(lhs, input_width, output_width, weight, name):
        # Recurrent MoE projections are rank two by construction. Conditional
        # output dimensions are not necessarily queryable before graph build.
        if isinstance(weight, PreparedLinear):
            if quant_ctx is None:
                raise ValueError("Prepared MoE projection requires its quantization context")
            return weight.matmul(network, lhs, input_width, output_width)
        rhs = graph_ops.add_constant(network, (input_width, output_width), weight, dtype=dtype)
        return graph_ops._add_matrix_multiply_with_fp32_accumulation(
            network, lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
        )

    routed_input = normed
    if latent is not None:
        routed_input = matmul(
            normed, hidden_size, routed_width, weights[f"{prefix}.latent_in"], f"{prefix}.latent_in"
        )
    routed = constant(np.zeros((1, routed_width)))
    for expert in range(experts):
        selected = network.add_slice(membership, (expert, 0), (1, 1), (1, 1)).get_output(0)
        active = binary(selected, constant([[0.0]]), trt.ElementWiseOperation.GREATER)
        conditional = network.add_if_conditional()
        conditional.set_condition(reshape(active, ()))
        payload = conditional.add_input(routed_input).get_output(0)
        expert_prefix = f"{prefix}.expert.{expert}"
        up = matmul(
            payload, routed_width, width, weights[expert_prefix + ".up"], expert_prefix + ".up"
        )
        activated = graph_ops.add_activation(network, up, "relu2", dtype=dtype)
        down = matmul(
            activated,
            width,
            routed_width,
            weights[expert_prefix + ".down"],
            expert_prefix + ".down",
        )
        if down.dtype != trt.float32:
            down = network.add_cast(down, trt.float32).get_output(0)
        zero = constant(np.zeros((1, routed_width)))
        output = conditional.add_output(down, zero).get_output(0)
        probability = network.add_slice(routed_scores, (0, expert), (1, 1), (1, 1)).get_output(0)
        routed = binary(
            routed,
            binary(output, probability, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM,
        )
    routed = reshape(routed, (1, routed_width))
    if routed.dtype != normed.dtype:
        routed = network.add_cast(routed, normed.dtype).get_output(0)
    if latent is not None:
        routed = matmul(
            routed,
            routed_width,
            hidden_size,
            weights[f"{prefix}.latent_out"],
            f"{prefix}.latent_out",
        )
    shared = matmul(
        normed, hidden_size, shared_width, weights[f"{prefix}.shared.up"], f"{prefix}.shared.up"
    )
    shared = graph_ops.add_activation(network, shared, "relu2", dtype=dtype)
    shared = matmul(
        shared, shared_width, hidden_size, weights[f"{prefix}.shared.down"], f"{prefix}.shared.down"
    )
    return binary(
        hidden, binary(routed, shared, trt.ElementWiseOperation.SUM), trt.ElementWiseOperation.SUM
    )
