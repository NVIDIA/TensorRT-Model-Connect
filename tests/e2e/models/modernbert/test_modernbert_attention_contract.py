# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT attention and RoPE config contract tests."""

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.families.modernbert.config import (
    ModelConfig,
    resolve_attention_contract,
)


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
    with pytest.raises(
        ValueError, match="global_attn_every_n_layers must be positive"
    ):
        resolve_attention_contract(
            _config(global_attn_every_n_layers=0, layer_types=None)
        )
