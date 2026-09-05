# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-Neo local/global attention contracts."""

from __future__ import annotations

from types import SimpleNamespace

from families.gpt_neo import model
from families.gpt_neo.attention_contract import resolve_attention_layer_types


def test_model_forwards_hf_attention_pattern_to_decoder_builder(monkeypatch) -> None:
    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return b"engine"

    monkeypatch.setattr(model, "build_standard_decoder_engine", fake_build)
    config = SimpleNamespace(
        num_hidden_layers=4,
        raw={
            "attention_layers": ["global", "local", "global", "local"],
            "window_size": 256,
        },
    )

    result = model._GPTNeoModel().build_engine(
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
