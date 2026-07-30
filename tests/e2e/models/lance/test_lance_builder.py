# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the LANCE decoder builder."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

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


def test_lance_vl_config_matches_official_x2t_image_framing() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.plugin")
    config = SimpleNamespace(
        raw={
            "vision_config": {
                "patch_size": 14,
                "spatial_merge_size": 2,
            },
            "image_token_id": 151655,
            "video_token_id": 151656,
        },
        hidden_size=2048,
    )

    vl_config = module.LancePlugin().get_vl_config(config)

    assert vl_config is not None
    assert vl_config["fixed_image_size"] == 448
    assert vl_config["num_image_pad_tokens"] == 256
    assert vl_config["image_token_id"] == 151656
    assert vl_config["image_token_str"] == "<|video_pad|>"
    assert vl_config["vl_prompt_template"] == (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|>{image_pads}<|vision_end|>"
        "{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
