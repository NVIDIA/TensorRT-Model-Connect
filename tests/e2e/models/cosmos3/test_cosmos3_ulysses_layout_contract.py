# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-array layout contract for packed Cosmos3 Ulysses exchanges."""

from __future__ import annotations

import numpy as np


def _all_to_all(rank_inputs: list[np.ndarray]) -> list[np.ndarray]:
    world_size = len(rank_inputs)
    return [
        np.stack(
            [rank_inputs[source_rank][destination_rank] for source_rank in range(world_size)]
        )
        for destination_rank in range(world_size)
    ]


def _coalesced_all_to_all(
    routes_by_tensor: list[list[np.ndarray]],
) -> list[list[np.ndarray]]:
    world_size = len(routes_by_tensor[0])
    widths = [int(np.prod(routes[0].shape[1:])) for routes in routes_by_tensor]
    packed_inputs = [
        np.concatenate(
            [routes[source_rank].reshape(world_size, -1) for routes in routes_by_tensor],
            axis=1,
        )
        for source_rank in range(world_size)
    ]
    packed_outputs = _all_to_all(packed_inputs)

    outputs = []
    offset = 0
    for routes, width in zip(routes_by_tensor, widths):
        outputs.append(
            [
                packed[:, offset : offset + width].reshape(routes[0].shape)
                for packed in packed_outputs
            ]
        )
        offset += width
    return outputs


def test_coalesced_ulysses_preserves_rank_major_layout_and_round_trip() -> None:
    world_size = 8
    head_dim = 2
    # q_text, q_vision, k_text, v_text, k_vision, v_vision.
    specs = ((2, 32), (3, 32), (2, 8), (2, 8), (3, 8), (3, 8))
    inputs = []
    next_value = 0
    for local_length, total_heads in specs:
        rank_inputs = []
        for _ in range(world_size):
            size = local_length * total_heads * head_dim
            rank_inputs.append(
                np.arange(next_value, next_value + size).reshape(
                    local_length, total_heads * head_dim
                )
            )
            next_value += size
        inputs.append(rank_inputs)

    sequence_routes = []
    for rank_inputs, (local_length, total_heads) in zip(inputs, specs):
        local_heads = total_heads // world_size
        sequence_routes.append(
            [
                value.reshape(
                    local_length, world_size, local_heads, head_dim
                ).transpose(1, 0, 2, 3)
                for value in rank_inputs
            ]
        )

    individual_sequence_outputs = [_all_to_all(routes) for routes in sequence_routes]
    packed_sequence_outputs = _coalesced_all_to_all(sequence_routes)
    for individual, packed in zip(individual_sequence_outputs, packed_sequence_outputs):
        for individual_rank, packed_rank in zip(individual, packed):
            np.testing.assert_array_equal(packed_rank, individual_rank)

    full_head_tensors = []
    for packed_outputs, (local_length, total_heads) in zip(
        packed_sequence_outputs, specs
    ):
        local_heads = total_heads // world_size
        full_head_tensors.append(
            [
                value.transpose(2, 0, 1, 3).reshape(
                    1, local_heads, local_length * world_size, head_dim
                )
                for value in packed_outputs
            ]
        )

    return_routes = []
    for rank_inputs, (local_length, total_heads) in zip(full_head_tensors[:2], specs[:2]):
        local_heads = total_heads // world_size
        return_routes.append(
            [
                value.reshape(
                    local_heads, world_size, local_length, head_dim
                ).transpose(1, 0, 2, 3)
                for value in rank_inputs
            ]
        )

    individual_return_outputs = [_all_to_all(routes) for routes in return_routes]
    packed_return_outputs = _coalesced_all_to_all(return_routes)
    for tensor_index, (individual, packed) in enumerate(
        zip(individual_return_outputs, packed_return_outputs)
    ):
        local_length, total_heads = specs[tensor_index]
        for rank, (individual_rank, packed_rank) in enumerate(zip(individual, packed)):
            np.testing.assert_array_equal(packed_rank, individual_rank)
            local_rows = packed_rank.transpose(2, 0, 1, 3).reshape(
                local_length, total_heads * head_dim
            )
            np.testing.assert_array_equal(local_rows, inputs[tensor_index][rank])
