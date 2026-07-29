# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the FLUX.1 Ulysses context-parallel layout."""

from __future__ import annotations

from pathlib import Path


Tensor3D = list[list[list[int]]]


def _reference_seq_to_head(shards: list[Tensor3D]) -> list[Tensor3D]:
    """AITune layout: [S/CP,H,D] per rank -> [H/CP,S,D] per rank."""
    cp_size = len(shards)
    num_heads = len(shards[0][0])
    local_heads = num_heads // cp_size
    outputs = []
    for destination in range(cp_size):
        head_start = destination * local_heads
        head_end = head_start + local_heads
        outputs.append([
            token[head_start:head_end]
            for shard in shards
            for token in shard
        ])
    return outputs


def _reference_head_to_seq(head_shards: list[Tensor3D]) -> list[Tensor3D]:
    """Inverse AITune layout: [H/CP,S,D] per rank -> [S/CP,H,D]."""
    cp_size = len(head_shards)
    full_seq = len(head_shards[0])
    local_seq = full_seq // cp_size
    outputs = []
    for destination in range(cp_size):
        seq_start = destination * local_seq
        seq_end = seq_start + local_seq
        outputs.append([
            [head for shard in head_shards for head in shard[token_index]]
            for token_index in range(seq_start, seq_end)
        ])
    return outputs


def test_cp4_ulysses_sequence_head_exchange_is_exactly_invertible() -> None:
    cp_size = 4
    local_seq = 5
    num_heads = 24
    head_dim = 3
    shards = [[
        [[rank * 100_000 + token * 1_000 + head * 10 + dim
          for dim in range(head_dim)]
         for head in range(num_heads)]
        for token in range(local_seq)
    ] for rank in range(cp_size)]

    head_shards = _reference_seq_to_head(shards)

    assert [
        (len(value), len(value[0]), len(value[0][0]))
        for value in head_shards
    ] == [
        (local_seq * cp_size, num_heads // cp_size, head_dim)
    ] * cp_size
    restored = _reference_head_to_seq(head_shards)
    for expected, actual in zip(shards, restored, strict=True):
        assert actual == expected


def test_cp_builder_keeps_aitune_permutations_and_split_rotary() -> None:
    builder_path = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/flux/flux_dit_cp_builder.py"
    )
    source = builder_path.read_text()

    assert "Permutation([1, 0, 2, 3])" in source
    assert "Permutation([2, 0, 1, 3])" in source
    assert "_reduce_scatter_replicated(network, hidden_inp, cp_size)" in source
    assert "_reduce_scatter_replicated(network, txt_cos_full, cp_size)" in source
    assert "_reduce_scatter_replicated(network, img_cos_full, cp_size)" in source
    assert "local_cos = _concat_rows(network, txt_cos, img_cos)" in source
    assert "CollectiveOperation.ALL_TO_ALL" in source
    assert "CollectiveOperation.ALL_GATHER" in source
