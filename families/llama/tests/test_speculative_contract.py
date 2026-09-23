# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import replace

from ..speculative.contract import EngineContract, ExecutionProfile, tree_visibility


def test_siblings_are_hidden_and_ancestors_visible():
    mask = tree_visibility([-1, 0, 0, 1, 2])
    assert mask[3] == [1, 1, 0, 1, 0]
    assert mask[4] == [1, 0, 1, 0, 1]


@pytest.mark.parametrize("parents", [[], [0], [-1, 1], [-1, -1], [-1, 2, 0]])
def test_invalid_tree_rejected(parents):
    with pytest.raises(ValueError):
        tree_visibility(parents)


def test_contract_rejects_incompatible_precision_and_capacity():
    fields = dict(role="target", layers=32, hidden_size=4096, kv_heads=8,
                  head_dim=128, vocab_size=128256, capacity=2048, feature_width=12288)
    with pytest.raises(ValueError):
        EngineContract(**fields, precision="bf16")
    with pytest.raises(ValueError):
        EngineContract(**fields, max_query=2049)
    with pytest.raises(ValueError):
        EngineContract(**fields, greedy_selection="unknown")
    assert EngineContract(**fields, greedy_selection="device_v1").to_dict()["greedy_selection"] == "device_v1"


def test_contract_rejects_unsupported_state_semantics():
    contract = EngineContract("target", 32, 4096, 8, 128, 128256, 2048, feature_width=12288)
    for changes in ({"version": 2}, {"cache_layout": "unknown"},
                    {"cache_update": "unknown"}, {"alias_contract": "unknown"}):
        with pytest.raises(ValueError):
            replace(contract, **changes)


def test_phase_profiles_separate_query_and_selected_rows():
    profiles = (ExecutionProfile("prefill", (1, 1024, 1024), (1, 1, 1)),
                ExecutionProfile("decode", (1, 5, 9), (1, 5, 9)))
    contract = EngineContract("target", 32, 4096, 8, 128, 128256, 2048,
                              max_query=1024, feature_width=12288, execution_profiles=profiles)
    assert contract.profile_shapes("attention_mask", (-1, 2048)) == [
        ((1, 2048), (1024, 2048), (1024, 2048)), ((1, 2048), (5, 2048), (9, 2048))]
    assert contract.profile_shapes("logits_indices", (-1,)) == [
        ((1,), (1,), (1,)), ((1,), (5,), (9,))]
    assert contract.profile_shapes("token_id", (-1,))[1] == ((1,), (5,), (9,))
    for changes in ({"execution_profiles": profiles[::-1]}, {"max_query": 2048},
                    {"role": "draft"}, {"execution_profiles": profiles[:1]}):
        with pytest.raises(ValueError):
            replace(contract, **changes)
    legacy = replace(contract, execution_profiles=(), max_query=64)
    assert legacy.profile_shapes("logits_indices", (-1,)) == [((1,), (16,), (64,))]


@pytest.mark.parametrize("query,logits", [((0, 5, 9), (1, 1, 1)), ((1, 9, 5), (1, 1, 1)),
                                         ((1, 5, 9), (1, 6, 9)), ((1, 5, 9), (1, 5, 10))])
def test_invalid_profile_bounds(query, logits):
    with pytest.raises(ValueError):
        ExecutionProfile("decode", query, logits)
