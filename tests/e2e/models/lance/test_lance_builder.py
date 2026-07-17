# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the LANCE decoder builder."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


def test_lance_embed_input_dispatches_to_dual_profile_builder(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.default_decoder")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"lance-dual-profile-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = type("Config", (), {"raw": {"_decoder_engine_role": "decode"}})()
    result = module.build_standard_decoder_engine(
        config, {}, 31, precision="fp16", embed_input=True)

    assert result == b"lance-dual-profile-plan"
    assert calls["build"][3]["embed_input"] is True
