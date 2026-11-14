# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT attention and RoPE config contract tests."""

from __future__ import annotations

import json

import pytest

from families.modernbert.config import ModelConfig, resolve_attention_contract


def _config(**overrides) -> ModelConfig:
    raw = {
        "model_type": "modernbert",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 7,
        "num_attention_heads": 4,
        **overrides,
    }
    return ModelConfig.from_json(json.dumps(raw))


def test_resolves_released_checkpoint_attention_layout() -> None:
    contract = resolve_attention_contract(
        _config(
            global_attn_every_n_layers=3,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            layer_types=None,
            rope_parameters=None,
        )
    )

    assert contract.layer_types == (
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )
    assert contract.full_rope_theta == 160000.0
    assert contract.sliding_rope_theta == 10000.0


def test_prefers_materialized_attention_layout() -> None:
    contract = resolve_attention_contract(
        _config(
            num_hidden_layers=2,
            global_attn_every_n_layers=3,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "full_attention": {"rope_theta": 200000.0},
                "sliding_attention": {"rope_theta": 20000.0},
            },
        )
    )

    assert contract.layer_types == (
        "sliding_attention",
        "full_attention",
    )
    assert contract.full_rope_theta == 200000.0
    assert contract.sliding_rope_theta == 20000.0


def test_rejects_non_positive_global_attention_interval() -> None:
    with pytest.raises(ValueError, match="global_attn_every_n_layers must be positive"):
        resolve_attention_contract(_config(global_attn_every_n_layers=0, layer_types=None))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({}, 64),
        ({"local_attention": 128}, 64),
        ({"local_attention": 5}, 2),
        ({"local_attention": 8, "sliding_window": 1}, 1),
        ({"sliding_window": 0}, 0),
        ({"sliding_window": 2147483646}, 2147483646),
        ({"local_attention": 4294967293}, 2147483646),
    ],
)
def test_resolves_inclusive_local_half_window(raw, expected):
    assert resolve_attention_contract(_config(**raw)).sliding_window == expected


@pytest.mark.parametrize("key", ["local_attention", "sliding_window"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "4"])
def test_rejects_invalid_local_window(key, value):
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_attention_contract(_config(**{key: value}))


@pytest.mark.parametrize("raw", [{"sliding_window": 2147483647}, {"local_attention": 4294967294}])
def test_rejects_local_window_outside_graph_int32_range(raw):
    with pytest.raises(ValueError, match="half-window must be at most 2147483646"):
        resolve_attention_contract(_config(**raw))
