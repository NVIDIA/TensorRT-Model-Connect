# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-selection contracts for Cosmos3 B200 context parallel attention."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.cosmos3.transformer_builder import (
    select_attention_decomposition,
    select_cp_rank_local_sharding,
    select_cp_vision_query_chunk_size,
)


@pytest.mark.parametrize(
    ("compute_capability", "cp_size"),
    [
        ((8, 0), 8),
        ((9, 0), 8),
        ((10, 0), 1),
        ((10, 3), 8),
        ((11, 0), 8),
    ],
)
def test_cp_vision_query_chunking_does_not_change_unqualified_targets(
    compute_capability: tuple[int, int],
    cp_size: int,
) -> None:
    assert (
        select_cp_vision_query_chunk_size(
            compute_capability,
            cp_size=cp_size,
            local_vision_length=5_520,
        )
        is None
    )


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_cp_vision_query_chunking_uses_full_query_on_b200_distributed(
    cp_size: int,
) -> None:
    assert (
        select_cp_vision_query_chunk_size(
            (10, 0),
            cp_size=cp_size,
            local_vision_length=44_160 // cp_size,
        )
        == 44_160
    )


@pytest.mark.parametrize("cp_size", [1, 8])
def test_attention_decomposition_is_qualified_for_b200(
    cp_size: int,
) -> None:
    assert select_attention_decomposition((10, 0), cp_size=cp_size)


@pytest.mark.parametrize(
    ("compute_capability", "cp_size"),
    [
        ((8, 0), 1),
        ((8, 0), 8),
        ((9, 0), 1),
        ((9, 0), 8),
        ((10, 0), 2),
        ((10, 0), 4),
        ((10, 3), 1),
        ((10, 3), 8),
        ((11, 0), 1),
        ((11, 0), 8),
    ],
)
def test_attention_decomposition_preserves_unqualified_tactic_policy(
    compute_capability: tuple[int, int],
    cp_size: int,
) -> None:
    assert not select_attention_decomposition(
        compute_capability,
        cp_size=cp_size,
    )


@pytest.mark.parametrize(
    ("compute_capability", "cp_size"),
    [
        ((8, 0), 8),
        ((9, 0), 8),
        ((10, 0), 2),
        ((10, 3), 8),
        ((11, 0), 8),
    ],
)
def test_rank_local_sharding_does_not_change_unqualified_targets(
    compute_capability: tuple[int, int],
    cp_size: int,
) -> None:
    assert not select_cp_rank_local_sharding(
        compute_capability,
        cp_size=cp_size,
    )


@pytest.mark.parametrize("requested_cp_size", [4, 8])
def test_rank_local_sharding_is_qualified_for_b200_cfg_split_cp_graphs(
    requested_cp_size: int,
) -> None:
    assert select_cp_rank_local_sharding((10, 0), cp_size=requested_cp_size)
