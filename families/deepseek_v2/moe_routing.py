# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2/V3 mixture-of-experts routing graph helpers."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_ops




def validate_router_contract(
    *,
    scoring_func: str,
    topk_method: str,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
) -> None:
    """Reject router configurations that this family cannot reproduce."""
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(
            f"Unsupported DeepSeek MoE scoring_func: {scoring_func}")
    if topk_method not in ("greedy", "group_limited_greedy", "noaux_tc"):
        raise ValueError(
            f"Unsupported DeepSeek MoE topk_method: {topk_method}")
    if not 0 < num_experts_per_tok <= n_routed_experts:
        raise ValueError(
            "num_experts_per_tok must be in [1, n_routed_experts]")

    if topk_method == "greedy":
        return
    if n_group <= 0 or n_routed_experts % n_group != 0:
        raise ValueError(
            "Grouped DeepSeek MoE routing requires n_routed_experts "
            "divisible by n_group")
    if not 0 < topk_group <= n_group:
        raise ValueError("topk_group must be in [1, n_group]")
    candidates = topk_group * (n_routed_experts // n_group)
    if num_experts_per_tok > candidates:
        raise ValueError(
            "Grouped DeepSeek MoE routing selects fewer candidates than "
            "num_experts_per_tok")
    if topk_method == "noaux_tc":
        experts_per_group = n_routed_experts // n_group
        if scoring_func != "sigmoid":
            raise ValueError("noaux_tc routing requires sigmoid scoring")
        if experts_per_group < 2:
            raise ValueError(
                "noaux_tc routing requires at least two experts per group")


def _reshape(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    shape: tuple[int, ...],
) -> trt.ITensor:
    shuffle = network.add_shuffle(tensor)
    shuffle.reshape_dims = shape
    return shuffle.get_output(0)


def _group_limited_topk(
    network: trt.INetworkDefinition,
    choice_scores: trt.ITensor,
    *,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    group_score_topk: int,
) -> trt.ITensor:
    """Select global expert IDs from the highest-scoring expert groups."""
    experts_per_group = n_routed_experts // n_group
    grouped_scores = _reshape(
        network, choice_scores, (n_group, experts_per_group))

    group_topk = network.add_topk(
        grouped_scores,
        trt.TopKOperation.MAX,
        group_score_topk,
        1 << 1,
    )
    group_scores = network.add_reduce(
        group_topk.get_output(0),
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=False,
    )
    group_scores_2d = _reshape(
        network, group_scores.get_output(0), (1, n_group))
    selected_groups = network.add_topk(
        group_scores_2d,
        trt.TopKOperation.MAX,
        topk_group,
        1 << 1,
    ).get_output(1)

    expert_group_ids = np.repeat(
        np.arange(n_group, dtype=np.int32), experts_per_group).reshape(
            1, n_routed_experts)
    expert_group_ids_tensor = graph_ops.add_constant(
        network,
        expert_group_ids.shape,
        expert_group_ids,
        dtype=np.int32,
    )
    masked_scores = graph_ops.add_constant(
        network,
        (1, n_routed_experts),
        np.zeros((1, n_routed_experts), dtype=np.float32),
        dtype=np.float32,
    )
    for group_position in range(topk_group):
        selected_group = network.add_slice(
            selected_groups,
            start=(0, group_position),
            shape=(1, 1),
            stride=(1, 1),
        ).get_output(0)
        in_selected_group = network.add_elementwise(
            expert_group_ids_tensor,
            selected_group,
            trt.ElementWiseOperation.EQUAL,
        ).get_output(0)
        masked_scores = network.add_select(
            in_selected_group,
            choice_scores,
            masked_scores,
        ).get_output(0)

    selected_experts = network.add_topk(
        masked_scores,
        trt.TopKOperation.MAX,
        num_experts_per_tok,
        1 << 1,
    )
    return selected_experts.get_output(1)


def add_router(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    router_weight: np.ndarray,
    *,
    hidden_size: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    scoring_func: str,
    topk_method: str,
    correction_bias: np.ndarray | None,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Return selected expert IDs and their DeepSeek routing weights."""
    validate_router_contract(
        scoring_func=scoring_func,
        topk_method=topk_method,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
    )

    router_input = inp
    if router_input.dtype != trt.float32:
        router_input = network.add_cast(
            router_input, trt.float32).get_output(0)
    router_logits = graph_ops.add_matmul_rhs_constant(
        network,
        router_input,
        hidden_size,
        n_routed_experts,
        router_weight,
        dtype=np.float32,
    )

    if scoring_func == "sigmoid":
        scores = network.add_activation(
            router_logits, trt.ActivationType.SIGMOID).get_output(0)
    else:
        softmax = network.add_softmax(router_logits)
        softmax.axes = 1 << 1
        scores = softmax.get_output(0)

    if topk_method == "greedy":
        selected = network.add_topk(
            scores,
            trt.TopKOperation.MAX,
            num_experts_per_tok,
            1 << 1,
        )
        top_indices = selected.get_output(1)
        top_weights = selected.get_output(0)
    else:
        choice_scores = scores
        group_score_topk = 1
        if topk_method == "noaux_tc":
            if correction_bias is None:
                correction_bias = np.zeros(
                    n_routed_experts, dtype=np.float32)
            bias = graph_ops.add_constant(
                network,
                (1, n_routed_experts),
                np.asarray(correction_bias, dtype=np.float32).reshape(
                    1, n_routed_experts),
                dtype=np.float32,
            )
            choice_scores = network.add_elementwise(
                scores, bias, trt.ElementWiseOperation.SUM).get_output(0)
            group_score_topk = 2

        top_indices = _group_limited_topk(
            network,
            choice_scores,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            group_score_topk=group_score_topk,
        )
        scores_1d = _reshape(network, scores, (n_routed_experts,))
        indices_1d = _reshape(
            network, top_indices, (num_experts_per_tok,))
        top_weights_1d = network.add_gather(
            scores_1d, indices_1d, 0).get_output(0)
        top_weights = _reshape(
            network, top_weights_1d, (1, num_experts_per_tok))

    if norm_topk_prob:
        denominator = network.add_reduce(
            top_weights,
            trt.ReduceOperation.SUM,
            1 << 1,
            keep_dims=True,
        ).get_output(0)
        epsilon = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([[1e-20]], dtype=np.float32),
            dtype=np.float32,
        )
        denominator = network.add_elementwise(
            denominator,
            epsilon,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        top_weights = network.add_elementwise(
            top_weights,
            denominator,
            trt.ElementWiseOperation.DIV,
        ).get_output(0)

    if routed_scaling_factor != 1.0:
        scale = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([[routed_scaling_factor]], dtype=np.float32),
            dtype=np.float32,
        )
        top_weights = network.add_elementwise(
            top_weights,
            scale,
            trt.ElementWiseOperation.PROD,
        ).get_output(0)

    return top_indices, top_weights
