# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Qwen-owned TriAttention selection helpers."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.models.qwen.triattention_runtime import (
    TriAttentionRuntimeConfig,
    TriAttentionSelector,
)


def _make_stats_payload() -> dict:
    return {
        "version": 1,
        "head_dim": 4,
        "rope_style": "half",
        "rope_theta": 10000.0,
        "sampled_heads": [[0, 0]],
        "stats": {
            "layer00_head00": {
                "q_mean_real": [0.5, 0.25],
                "q_mean_imag": [0.1, -0.2],
                "q_abs_mean": [0.6, 0.7],
            }
        },
    }


def test_selector_keeps_budget_and_recent_window():
    selector = TriAttentionSelector(
        _make_stats_payload(),
        TriAttentionRuntimeConfig(
            kv_budget=4,
            recent_window=2,
            score_aggregation="mean",
        ),
    )
    layer_cache = np.asarray([
        [0.1, 0.0, 0.2, 0.0],
        [0.3, 0.1, 0.4, 0.2],
        [0.5, 0.1, 0.3, 0.4],
        [0.8, 0.2, 0.5, 0.3],
        [0.7, 0.4, 0.6, 0.2],
        [0.9, 0.3, 0.8, 0.4],
    ], dtype=np.float32)

    keep = selector.select_keep_indices(
        {0: layer_cache},
        cache_positions=[0, 1, 2, 3, 4, 5],
        next_position=6,
    )

    assert keep.shape == (3,)
    assert keep[-1] == 5


def test_selector_protects_prefill_by_absolute_position():
    selector = TriAttentionSelector(
        _make_stats_payload(),
        TriAttentionRuntimeConfig(
            kv_budget=3,
            recent_window=1,
            protect_prefill=True,
        ),
    )
    layer_cache = np.asarray([
        [0.1, 0.0, 0.2, 0.0],
        [0.2, 0.1, 0.3, 0.1],
        [0.3, 0.2, 0.4, 0.2],
        [0.4, 0.3, 0.5, 0.3],
        [0.5, 0.4, 0.6, 0.4],
    ], dtype=np.float32)

    keep = selector.select_keep_indices(
        {0: layer_cache},
        cache_positions=[0, 4, 5, 1, 6],
        next_position=7,
        prefix_length=2,
    )

    assert keep.tolist() == [0, 3]


def test_selector_maps_score_heads_to_gqa_cache_groups():
    selector = TriAttentionSelector(
        {
            "version": 2,
            "head_dim": 4,
            "rope_style": "half",
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "stats_head_count": 2,
            "sampled_heads": [[0, 1]],
            "stats": {
                "layer00_head01": {
                    "q_mean_real": [1.0, 0.0],
                    "q_mean_imag": [0.0, 0.0],
                    "q_abs_mean": [1.0, 0.0],
                }
            },
        },
        TriAttentionRuntimeConfig(
            kv_budget=3,
            recent_window=1,
            disable_trig=True,
            disable_mlr=True,
        ),
    )
    layer_cache = np.asarray([
        [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.0,
        ],
    ], dtype=np.float32)

    keep = selector.select_keep_indices(
        {0: layer_cache},
        cache_positions=[0, 1, 2, 3],
        next_position=4,
    )

    assert keep.tolist() == [1, 2]


def test_runtime_config_reads_count_prompt_tokens():
    cfg = TriAttentionRuntimeConfig.from_bundle_config(
        {
            "kv_budget": 5,
            "recent_window": 2,
            "count_prompt_tokens": False,
        },
        rope_style="half",
        max_cache_length=16,
    )

    assert cfg.kv_budget == 5
    assert cfg.count_prompt_tokens is False
