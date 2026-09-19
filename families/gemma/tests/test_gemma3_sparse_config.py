# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3 checkpoints that state only geometry must still build correctly.

google/gemma-3-12b-it names hidden_size, the layer and head counts,
intermediate_size, sliding_window and rope_scaling, and nothing else; every
other value comes from Gemma3TextConfig. Mirrors write the full set out, which
is why a build against a mirror can pass while the same build against the
official checkpoint silently produces a different model.
"""

from __future__ import annotations

import pytest

from families.gemma import graph_blocks
from families.gemma.model import _GEMMA3_CONFIG_DEFAULTS, _apply_gemma3_config_defaults


class _Config:
    """Minimal stand-in for the parsed config, matching the sparse official shape."""

    def __init__(self, **raw):
        self.model_type = raw.pop("model_type", "gemma3")
        self.hidden_size = raw.get("hidden_size", 3840)
        self.num_hidden_layers = raw.get("num_hidden_layers", 48)
        self.num_attention_heads = raw.get("num_attention_heads", 16)
        self.num_key_value_heads = raw.get("num_key_value_heads", 8)
        # The parser's own fallback, which is the local base rather than the
        # global one; the defaults must overwrite it.
        self.rope_theta = 10000.0
        self.rms_norm_eps = None
        self.hidden_act = None
        self.max_position_embeddings = None
        self._head_dim = None
        self.raw = dict(raw)

    @property
    def head_dim(self):
        if self._head_dim:
            return self._head_dim
        return self.hidden_size // self.num_attention_heads


class _Readers:
    tensor_map: dict = {}


def _official_12b() -> _Config:
    return _Config(
        hidden_size=3840,
        intermediate_size=15360,
        num_attention_heads=16,
        num_hidden_layers=48,
        num_key_value_heads=8,
        sliding_window=1024,
        rope_scaling={"factor": 8.0, "rope_type": "linear"},
    )


def test_sparse_config_resolves_the_gemma3_rope_bases():
    config = _official_12b()
    _apply_gemma3_config_defaults(config, _Readers(), "model")

    assert config.head_dim == 256
    assert config.rope_theta == 1000000.0, "global layers must not inherit the local base"
    schedule = graph_blocks.gemma3_attention_schedule(config, config.num_hidden_layers)
    assert schedule["window"] == 1024
    assert schedule["local_theta"] == 10000.0


def test_sparse_config_places_local_attention_five_in_six():
    config = _official_12b()
    _apply_gemma3_config_defaults(config, _Readers(), "model")
    schedule = graph_blocks.gemma3_attention_schedule(config, config.num_hidden_layers)

    # transformers resolves layer_types for this config to full attention at
    # indices 5, 11, 17, 23, 29, 35, 41 and 47.
    assert sum(schedule["is_local"]) == 40
    assert [i for i, local in enumerate(schedule["is_local"]) if not local] == [
        5,
        11,
        17,
        23,
        29,
        35,
        41,
        47,
    ]


def test_gemma3_refuses_to_fall_back_to_all_global():
    """The old behavior built cleanly and generated fluent, wrong text."""
    config = _official_12b()
    # Defaults deliberately not applied, so the schedule inputs are missing.
    with pytest.raises(ValueError, match="rope_local_base_freq"):
        graph_blocks.gemma3_attention_schedule(config, config.num_hidden_layers)


def test_a_stated_value_still_wins_over_the_default():
    config = _official_12b()
    config.raw["rope_local_base_freq"] = 12345.0
    config.raw["sliding_window_pattern"] = 4
    _apply_gemma3_config_defaults(config, _Readers(), "model")
    schedule = graph_blocks.gemma3_attention_schedule(config, config.num_hidden_layers)

    assert schedule["local_theta"] == 12345.0
    assert [i for i, local in enumerate(schedule["is_local"]) if not local][:3] == [3, 7, 11]


def test_gemma2_is_untouched_by_the_gemma3_defaults():
    config = _Config(
        model_type="gemma2",
        hidden_size=2304,
        num_attention_heads=8,
        num_hidden_layers=26,
        sliding_window=4096,
    )
    _apply_gemma3_config_defaults(config, _Readers(), "model")

    assert config.rope_theta == 10000.0
    assert "rope_local_base_freq" not in config.raw
    schedule = graph_blocks.gemma3_attention_schedule(config, config.num_hidden_layers)
    assert not any(schedule["is_local"])


def test_defaults_cover_every_field_the_schedule_and_scale_read():
    for key in (
        "rope_theta",
        "rope_local_base_freq",
        "sliding_window_pattern",
        "query_pre_attn_scalar",
        "head_dim",
    ):
        assert key in _GEMMA3_CONFIG_DEFAULTS
