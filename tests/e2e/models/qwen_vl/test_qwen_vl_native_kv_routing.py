# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for Qwen2/2.5/3-VL native TensorRT KV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tensorrt_model_connect.families.qwen_vl.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    native_mrope_settings,
    prefer_native_default,
)
from tensorrt_model_connect.families.qwen_vl.config import ModelConfig
from tensorrt_model_connect.families.qwen_vl.native_kv_contract import (
    validate_native_kv_weights,
)


_FAMILIES = (
    (
        "qwen2_vl",
        "Qwen2VLForConditionalGeneration",
        1536,
        28,
        12,
        2,
        32768,
        (16, 24, 24),
        False,
    ),
    (
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
        2048,
        36,
        16,
        2,
        128000,
        (16, 24, 24),
        False,
    ),
    (
        "qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        2048,
        28,
        16,
        8,
        262144,
        (24, 20, 20),
        True,
    ),
)


def _config(
    family_index: int = 1,
    *,
    raw_updates: dict | None = None,
    **overrides,
) -> ModelConfig:
    (
        model_type,
        architecture,
        hidden,
        layers,
        heads,
        kv_heads,
        context,
        section,
        interleaved,
    ) = _FAMILIES[family_index]
    values = {
        "model_type": model_type,
        "architectures": [architecture],
        "vocab_size": 151936,
        "hidden_size": hidden,
        "intermediate_size": hidden * 4,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": context,
        "hidden_act": "silu",
        "_head_dim": 128,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "vision_config": {
            "deepstack_visual_indexes": [5, 11, 17] if model_type == "qwen3_vl" else [],
        },
        "text_config": {
            "head_dim": 128,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": list(section),
                "mrope_interleaved": interleaved,
            },
        },
        "use_sliding_window": False,
    }
    raw.update(raw_updates or {})
    values["raw"] = raw
    return ModelConfig(**values)


@pytest.mark.parametrize("family_index", range(len(_FAMILIES)))
def test_all_official_qwen_vl_generations_use_full_context_native_kv(family_index):
    config = _config(family_index)
    capability = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(
        config, config.max_position_embeddings
    )

    assert capability.eligible, capability.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert row_bytes == 2 * config.num_hidden_layers * config.num_key_value_heads * 128 * 2
    assert cache_bytes == config.max_position_embeddings * row_bytes


def test_qwen3_interleaved_mrope_contract_is_preserved():
    assert native_mrope_settings(_config(2)) == ((24, 20, 20), True)
    assert native_mrope_settings(_config(1)) == ((16, 24, 24), False)


def test_qwen3_requires_its_deepstack_graph_contract():
    decision = native_kv_architecture_capability(
        _config(2, raw_updates={"vision_config": {}})
    )

    assert not decision.eligible
    assert "DeepStack" in decision.reason


def test_tp_cache_geometry_is_rank_local():
    config = _config(2)
    single_row, single_cache = native_kv_cache_geometry(
        config, config.max_position_embeddings
    )
    rank_row, rank_cache = native_kv_cache_geometry(
        config, config.max_position_embeddings, tp_size=4
    )

    assert rank_row * 4 == single_row
    assert rank_cache * 4 == single_cache


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"architectures": ["OtherForConditionalGeneration"]}, {}, "architectures"),
        ({"_head_dim": 64}, {"text_config": {}}, "head_dim=128"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        (
            {},
            {
                "text_config": {
                    "head_dim": 128,
                    "use_sliding_window": True,
                    "rope_parameters": {
                        "rope_type": "default",
                        "mrope_section": [16, 24, 24],
                        "mrope_interleaved": False,
                    },
                }
            },
            "sliding-window",
        ),
        (
            {},
            {
                "text_config": {
                    "head_dim": 128,
                    "num_experts": 8,
                    "rope_parameters": {
                        "rope_type": "default",
                        "mrope_section": [16, 24, 24],
                        "mrope_interleaved": False,
                    },
                }
            },
            "num_experts",
        ),
        (
            {},
            {
                "text_config": {
                    "head_dim": 128,
                    "partial_rotary_factor": 0.5,
                    "rope_parameters": {
                        "rope_type": "default",
                        "mrope_section": [16, 24, 24],
                        "mrope_interleaved": False,
                    },
                }
            },
            "full rotary",
        ),
        (
            {},
            {
                "text_config": {
                    "head_dim": 128,
                    "rope_parameters": {
                        "rope_type": "default",
                        "mrope_section": [16, 24, 24],
                        "mrope_interleaved": True,
                    },
                }
            },
            "interleaved mRoPE",
        ),
        (
            {},
            {
                "text_config": {
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 4.0,
                        "mrope_section": [16, 24, 24],
                    }
                }
            },
            "scaled mRoPE",
        ),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_updates, reason):
    decision = native_kv_architecture_capability(
        _config(raw_updates=raw_updates, **overrides)
    )

    assert decision.applicable
    assert not decision.eligible
    assert reason in decision.reason


@pytest.mark.parametrize(
    ("kwargs", "raw_updates", "reason"),
    [
        ({"precision": "fp16"}, {}, "BF16"),
        ({"max_cache_length": 127999}, {}, "max_cache_length"),
        ({"dynamic_kv_cache": True}, {}, "fixed physical"),
        ({}, {"_runtime_dynamic_kv_requested": True}, "fixed physical"),
        ({}, {"dynamic_kv_cache": True}, "fixed physical"),
        ({"quantized": True}, {}, "quantized"),
        ({"debug_layer_outputs": True}, {}, "debug"),
        ({"lora_enabled": True}, {}, "LoRA"),
        ({}, {"_fp32_layers": [0]}, "FP32 layer"),
        ({}, {"_decoder_engine_layout": "dual_profile"}, "split"),
        ({}, {"_rtx_build_requested": True}, "standard TensorRT"),
    ],
)
def test_unsupported_build_modes_have_no_legacy_fallback(kwargs, raw_updates, reason):
    decision = native_kv_build_capability(
        _config(raw_updates=raw_updates),
        **kwargs,
    )

    assert not decision.eligible
    assert reason in decision.reason


def test_tp_requires_kv_head_divisibility():
    config = _config(1)
    assert native_kv_build_capability(config, tp_size=2).eligible
    rejected = native_kv_build_capability(config, tp_size=4)
    assert not rejected.eligible
    assert "divisible" in rejected.reason


def test_declared_vision_engine_fails_closed_before_kv_admission():
    source = (
        Path(__file__).resolve().parents[4]
        / "src/runtime/models/qwen_vl/plugin.cpp"
    ).read_text(encoding="utf-8")

    assert 'extract_json_bool(ctx.config_json, "has_vision_engine", false)' in source
    assert "declared_in_config || plan != nullptr" in source
    assert "declared || plan != nullptr" in source
    assert 'throw std::runtime_error("Bundle missing vision_engine_plan")' in source
    assert "if (required)\n            throw;" in source
    assert "Bundle declares vision engine but" not in source

    create_lanes = source.index(
        "std::vector<std::unique_ptr<IPipeline>> create_lanes"
    )
    load_vision = source.index("load_vision_lane_modules(", create_lanes)
    admission = source.index("admit_native_kv_allocation(", create_lanes)
    allocation = source.index("make_pipeline_lanes(", create_lanes)
    assert load_vision < admission < allocation


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def _weights(config: ModelConfig) -> dict[str, object]:
    hidden = config.hidden_size
    attention = config.num_attention_heads * 128
    kv_attention = config.num_key_value_heads * 128
    mlp = config.intermediate_size
    weights: dict[str, object] = {
        "embedding": _Tensor((config.vocab_size, hidden)),
        "final_norm": _Tensor((hidden,)),
        "w_out": _Tensor((hidden, config.vocab_size)),
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_mlp_size": mlp,
    }
    for layer in range(config.num_hidden_layers):
        for name, shape in (
            ("input_norm", (hidden,)),
            ("w_q", (hidden, attention)),
            ("w_k", (hidden, kv_attention)),
            ("w_v", (hidden, kv_attention)),
            ("w_o", (attention, hidden)),
            ("post_attn_norm", (hidden,)),
            ("w_gate", (hidden, mlp)),
            ("w_up", (hidden, mlp)),
            ("w_down", (mlp, hidden)),
        ):
            weights[f"layer.{layer}.{name}"] = _Tensor(shape)
    return weights


def test_qwen2_5_weight_contract_is_size_generic():
    config = _config(
        1,
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=256,
    )
    weights = _weights(config)
    validate_native_kv_weights(config, weights)

    weights.pop("layer.1.w_k")
    with pytest.raises(ValueError, match="continuous layer indices|missing.*w_k"):
        validate_native_kv_weights(config, weights)
