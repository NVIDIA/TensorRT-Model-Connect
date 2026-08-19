# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-Neo local/global attention contract tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def test_plugin_forwards_hf_attention_pattern_to_decoder_builder(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.gpt_neo.model"
    )

    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return b"engine"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        fake_build,
    )
    config = SimpleNamespace(
        num_hidden_layers=4,
        raw={
            "attention_layers": ["global", "local", "global", "local"],
            "window_size": 256,
        },
    )

    result = plugin_module.build_engine(
        config,
        {},
        max_cache_length=512,
    )

    assert result == b"engine"
    assert captured["attention_layer_types"] == (
        "global",
        "local",
        "global",
        "local",
    )
    assert captured["local_attention_window"] == 256


def test_attention_types_expand_to_all_layers() -> None:
    from tensorrt_model_connect.families.gpt_neo.attention_contract import (
        resolve_attention_layer_types,
    )

    assert resolve_attention_layer_types(
        {"attention_types": [[["global", "local"], 3]]},
        num_layers=6,
    ) == (
        "global",
        "local",
        "global",
        "local",
        "global",
        "local",
    )
