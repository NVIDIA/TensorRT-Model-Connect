# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Qwen-VL tensor-parallel dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="Qwen-VL builder tests require TensorRT")

try:
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(*, qwen3: bool = False) -> SimpleNamespace:
    raw = {"vision_config": {"deepstack_visual_indexes": [5, 11, 17]}} if qwen3 else {}
    return SimpleNamespace(
        raw=raw,
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        attention_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


@pytest.mark.parametrize(
    ("qwen3", "tp_size", "deepstack_num_levels"),
    [
        (False, 2, 0),
        (True, 4, 3),
    ],
)
def test_qwen_vl_plugin_routes_parallel_builds(
    monkeypatch, qwen3: bool, tp_size: int, deepstack_num_levels: int
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-tp-plan"

    monkeypatch.setattr(module, "build_qwen_vl_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=tp_size, rank=1)
    result = module.QwenVLPlugin().build_engine(
        _config(qwen3=qwen3),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"qwen-vl-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs["parallel_config"] == parallel
    assert kwargs["embed_input"] is True
    assert kwargs["deepstack_num_levels"] == deepstack_num_levels
    assert kwargs["verbose"] is True


def test_qwen25_vl_plugin_forwards_precision_to_standard_builder(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-plan"

    monkeypatch.setattr(module, "build_standard_decoder_engine", fake_build)

    result = module.QwenVLPlugin().build_engine(
        _config(qwen3=False),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        31,
        precision="bf16",
        verbose=True,
    )

    assert result == b"qwen-vl-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 31
    assert kwargs["precision"] == "bf16"
    assert kwargs["embed_input"] is True
    assert kwargs["verbose"] is True


def test_qwen25_vl_split_decode_uses_decode_profile(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.default_decoder")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    plugin = plugin_module.QwenVLPlugin()
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-dual-profile-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = _config(qwen3=False)
    config.raw["_decoder_engine_role"] = "decode"
    config.raw["_active_split_decoder_build"] = True
    assert plugin.supports_split_embed_input is True
    assert plugin.supports_split_decoder_roles(config) is True
    result = module.build_standard_decoder_engine(
        config, {}, 31, precision="fp16", embed_input=True)

    assert result == b"qwen-vl-dual-profile-plan"
    assert calls["build"][3]["embed_input"] is True
    assert calls["build"][3]["profile_mode"] == "decode"


def test_qwen25_vl_lora_keeps_dual_profile_prefill(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.default_decoder")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-lora-dual-profile-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = _config(qwen3=False)
    config.raw["_decoder_engine_role"] = "decode"
    config.raw["_family_build_options"] = {
        "qwen_vl_lora": {
            "enabled": True,
            "max_rank": 16,
            "target_modules": "q_proj,v_proj",
        }
    }

    result = module.build_standard_decoder_engine(
        config, {}, 31, precision="fp16", embed_input=True)

    assert result == b"qwen-vl-lora-dual-profile-plan"
    assert calls["build"][3]["embed_input"] is True


def test_qwen3_vl_vision_component_can_stay_fp32(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    vision_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.qwen_vl_vision_builder")
    calls: dict[str, object] = {}

    def fake_load(_model_dir, _config):
        return {"vision": "weights"}

    def fake_build(vision_config, weights, **kwargs):
        calls["build"] = (vision_config, weights, kwargs)
        return b"qwen3-vl-vision-plan"

    monkeypatch.setattr(module, "_load_vision_weights", fake_load)
    monkeypatch.setattr(vision_module, "build_qwen3_vl_vision_engine", fake_build)

    config = _config(qwen3=True)
    config.raw["_fp32_layers"] = [
        module._VISION_COMPONENT,
        module._VISION_LAYER_OFFSET + 5,
    ]
    result = module.QwenVLPlugin().build_vision_engine(
        "/tmp/model", config, {}, precision="fp16")

    assert result == b"qwen3-vl-vision-plan"
    assert calls["build"][2]["precision"] == "fp32"
    assert calls["build"][2]["fp32_layers"] == {5}


def test_qwen3_vl_text_decoder_component_can_stay_fp32(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen3-vl-text-plan"

    monkeypatch.setattr(module, "_build_qwen3_vl_decoder", fake_build)
    config = _config(qwen3=True)
    config.raw["_fp32_layers"] = [module._TEXT_DECODER_COMPONENT]

    result = module.QwenVLPlugin().build_engine(
        config,
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        31,
        precision="fp16",
    )

    assert result == b"qwen3-vl-text-plan"
    assert calls["build"][3]["precision"] == "fp32"
