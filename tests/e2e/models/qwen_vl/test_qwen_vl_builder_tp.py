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


def _config(
    *, qwen3: bool = False, context: int = 31, role: str = "decode",
) -> SimpleNamespace:
    model_type = "qwen3_vl" if qwen3 else "qwen2_5_vl"
    architecture = (
        "Qwen3VLForConditionalGeneration"
        if qwen3 else "Qwen2_5_VLForConditionalGeneration"
    )
    raw = {
        "_decoder_engine_layout": "split",
        "_decoder_engine_role": role,
        "vision_config": {
            "deepstack_visual_indexes": [5, 11, 17] if qwen3 else [],
        },
        "text_config": {
            "head_dim": 128,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
                "mrope_interleaved": qwen3,
            },
        },
    }
    return SimpleNamespace(
        raw=raw,
        model_type=model_type,
        architectures=[architecture],
        hidden_size=512,
        vocab_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=128,
        attention_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=context,
        hidden_act="silu",
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
    monkeypatch.setattr(module, "validate_native_kv_weights", lambda *_: None)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=tp_size, rank=1)
    result = module.QwenVLPlugin().build_engine(
        _config(qwen3=qwen3, context=23),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        precision="bf16",
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


def test_qwen25_vl_plugin_builds_native_split_role(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    monkeypatch.setattr(module, "validate_native_kv_weights", lambda *_: None)

    result = module.QwenVLPlugin().build_engine(
        _config(qwen3=False, context=31, role="prefill"),
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
    assert kwargs["profile_mode"] == "prefill"
    assert kwargs["verbose"] is True


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_standard_entrypoint_routes_to_native_split_role(
    monkeypatch, role: str,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.default_decoder")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-native-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    monkeypatch.setattr(module, "validate_native_kv_weights", lambda *_: None)
    config = _config(qwen3=False, role=role)
    result = module.build_standard_decoder_engine(
        config, {}, 31, precision="bf16", embed_input=True)

    assert result == b"qwen-vl-native-plan"
    assert calls["build"][3]["embed_input"] is True
    assert calls["build"][3]["profile_mode"] == role


def test_standard_entrypoint_requires_explicit_split_role() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.default_decoder")
    with pytest.raises(ValueError, match="explicit split"):
        module.build_standard_decoder_engine(
            _config(qwen3=False, role=""), {}, 31,
            precision="bf16", embed_input=True)


@pytest.mark.parametrize(
    ("precision", "family_options", "reason"),
    [
        ("fp16", {}, "requires BF16"),
        (
            "bf16",
            {
                "qwen_vl_lora": {
                    "enabled": True,
                    "max_rank": 16,
                    "target_modules": "q_proj,v_proj",
                },
            },
            "dynamic LoRA",
        ),
    ],
)
def test_standard_entrypoint_has_no_legacy_fallback(
    precision: str, family_options: dict, reason: str,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.default_decoder")
    config = _config(qwen3=False)
    config.raw["_family_build_options"] = family_options

    with pytest.raises(ValueError, match=reason):
        module.build_standard_decoder_engine(
            config, {}, 31, precision=precision, embed_input=True)


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


def test_qwen3_vl_text_fp32_override_has_no_legacy_fallback(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    config = _config(qwen3=True)
    config.raw["_fp32_layers"] = [0]

    with pytest.raises(ValueError, match="FP32 layer"):
        module.QwenVLPlugin().build_engine(
            config,
            {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
            31,
            precision="bf16",
        )


def test_generic_dynamic_kv_request_cannot_bypass_native_split_builder(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    config = _config(qwen3=False)
    config.raw["dynamic_kv_cache"] = True
    monkeypatch.setattr(
        module,
        "build_dual_profile_decoder_engine",
        lambda *_args, **_kwargs: pytest.fail("native builder must not run"),
    )

    with pytest.raises(ValueError, match="fixed physical KV capacity"):
        module.QwenVLPlugin().build_engine(
            config,
            {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
            31,
            precision="bf16",
        )
