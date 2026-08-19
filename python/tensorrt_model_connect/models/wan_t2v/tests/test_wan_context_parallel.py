# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for Wan2.1 Ulysses context parallelism."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_cp8_contiguous_patch_shards_reconstruct_full_sequence() -> None:
    cp_size = 8
    num_patches = 2016
    local_patches = num_patches // cp_size
    full = list(range(num_patches))

    shards = [full[rank * local_patches : (rank + 1) * local_patches] for rank in range(cp_size)]

    assert local_patches == 252
    assert [patch for shard in shards for patch in shard] == full


def test_cp8_routes_zero_padded_heads_without_changing_model_width() -> None:
    model_heads = 12
    cp_size = 8
    routed_heads = ((model_heads + cp_size - 1) // cp_size) * cp_size

    assert routed_heads == 16
    assert routed_heads // cp_size == 2


def test_cp_builder_uses_padded_head_ulysses_attention() -> None:
    source = (
        Path(__file__).resolve().parents[5]
        / "python/tensorrt_model_connect/models/wan_t2v/standard_dit_cp_builder.py"
    ).read_text()

    assert "CollectiveOperation.REDUCE_SCATTER" in source
    assert "CollectiveOperation.ALL_TO_ALL" in source
    assert "_pad_attention_heads(" in source
    assert "routed_num_heads = _round_up_to_multiple(num_heads, cp_size)" in source
    assert "CollectiveOperation.ALL_GATHER" in source
    assert "num_heads % parallel.cp_size" not in source


def test_cp4_manifest_declares_context_parallel_world_size() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "wan21-t2v-1.3b-l0-cp4.json"
    case = load_manifest(manifest_path)

    assert case.name == "wan21-t2v-1.3b-l0-cp4"
    assert case.metadata["build_args"]["parallel"] == {
        "mode": "context_parallel",
        "cp_size": 4,
    }
    assert case.metadata["distributed_runtime"]["world_size"] == 4
    assert case.reference_family == "diffusers_video_gen"
