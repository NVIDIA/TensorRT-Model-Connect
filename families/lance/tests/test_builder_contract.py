# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the family-owned LANCE decoder builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.lance import default_decoder, graph_ops  # noqa: E402
from families.lance import model as model_module  # noqa: E402


def test_embed_input_dispatches_to_dual_profile_builder(monkeypatch) -> None:
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["call"] = (config, weights, max_cache_length, kwargs)
        return b"lance-dual-profile-plan"

    monkeypatch.setattr(default_decoder, "build_dual_profile_decoder_engine", fake_build)
    config = SimpleNamespace(raw={"_decoder_engine_role": "prefill"})

    result = default_decoder.build_standard_decoder_engine(
        config,
        {},
        31,
        precision="fp16",
        embed_input=True,
    )

    assert result == b"lance-dual-profile-plan"
    assert captured["call"][3]["embed_input"] is True
    assert captured["call"][3]["profile_mode"] == "prefill"


def test_bf16_build_rounds_rope_inv_freq_like_official_reference(monkeypatch) -> None:
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured.update(kwargs)
        return b"lance-bf16-plan"

    monkeypatch.setattr(model_module, "build_standard_decoder_engine", fake_build)
    result = model_module._LanceModel().build_engine(
        SimpleNamespace(),
        {},
        512,
        precision="bf16",
    )

    assert result == b"lance-bf16-plan"
    assert captured["round_rope_inv_freq_to_bf16"] is True


def test_rope_table_matches_bf16_inv_freq_buffer() -> None:
    regular = graph_ops.make_rope_table_half_dim(388, 128, 1_000_000.0, True)
    official_bf16 = graph_ops.make_rope_table_half_dim(
        388,
        128,
        1_000_000.0,
        True,
        round_inv_freq_to_bf16=True,
    )

    np.testing.assert_array_equal(official_bf16[0], regular[0])
    assert np.max(np.abs(official_bf16[387] - regular[387])) > 0.25
    np.testing.assert_allclose(
        official_bf16[387, :4],
        np.array([-0.83420676, -0.92246085, 0.92788374, 0.06237314], dtype=np.float32),
        atol=1e-6,
    )


def test_vl_config_matches_official_x2t_image_framing() -> None:
    config = SimpleNamespace(
        raw={
            "vision_config": {"patch_size": 14, "spatial_merge_size": 2},
            "image_token_id": 151655,
            "video_token_id": 151656,
        },
        hidden_size=2048,
    )

    vl_config = model_module._LanceModel().get_vl_config(config)

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
