# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the LANCE decoder builder."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


def test_lance_decoder_dispatches_fixed_specialization(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.lance.default_decoder")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"lance-dual-profile-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = type("Config", (), {"raw": {"_decoder_engine_role": "decode"}})()
    result = module.build_standard_decoder_engine(
        config, {}, 31, precision="fp16")

    assert result == b"lance-dual-profile-plan"
    assert calls["build"][3] == {
        "precision": "fp16",
        "quant_ctx": None,
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "position_type": "rope",
        "activation": "silu",
        "partial_rotary_factor": 1.0,
        "interleaved_rope": False,
        "parallel_residual": False,
        "scale_attn_weights": True,
        "round_rope_inv_freq_to_bf16": False,
        "verbose": False,
        "profile_mode": "dual_profile",
    }


def test_lance_decoder_specialization_always_owns_embed_inputs() -> None:
    modules_and_builders = (
        (
            "tensorrt_model_connect.models.lance.default_decoder",
            "build_standard_decoder_engine",
        ),
        (
            "tensorrt_model_connect.models.lance.default_dual_profile_decoder",
            "build_dual_profile_decoder_engine",
        ),
    )

    for module_name, builder_name in modules_and_builders:
        builder = getattr(importlib.import_module(module_name), builder_name)
        assert "embed_input" not in inspect.signature(builder).parameters
        source = inspect.getsource(builder)
        assert "network.add_input('input_embed'" in source
        assert "network.add_input('use_input_embed'" in source


def test_lance_bf16_build_rounds_rope_inv_freq_like_official_reference(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.lance.model")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["kwargs"] = kwargs
        return b"lance-bf16-plan"

    monkeypatch.setattr(module, "build_standard_decoder_engine", fake_build)
    result = module.build_engine(
        SimpleNamespace(), {}, 512, precision="bf16")

    assert result == b"lance-bf16-plan"
    assert calls["kwargs"]["round_rope_inv_freq_to_bf16"] is True


def test_lance_rope_table_can_match_bf16_inv_freq_buffer() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.lance.graph_ops")
    regular = module.make_rope_table_half_dim(
        388, 128, 1_000_000.0, True)
    official_bf16 = module.make_rope_table_half_dim(
        388,
        128,
        1_000_000.0,
        True,
        round_inv_freq_to_bf16=True,
    )

    # Position zero is invariant; later positions expose the BF16 frequency
    # quantization performed by the official reference's model.to(bfloat16).
    np.testing.assert_array_equal(official_bf16[0], regular[0])
    assert np.max(np.abs(official_bf16[387] - regular[387])) > 0.25
    np.testing.assert_allclose(
        official_bf16[387, :4],
        np.array(
            [-0.83420676, -0.92246085, 0.92788374, 0.06237314],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_lance_vl_config_matches_official_x2t_image_framing() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.models.lance.model")
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

    vl_config = module.get_vl_config(config)

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
