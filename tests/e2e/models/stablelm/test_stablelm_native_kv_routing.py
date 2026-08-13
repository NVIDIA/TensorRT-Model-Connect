# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for StableLM's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import pytest

from tensorrt_model_connect.families.stablelm.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
    resolved_rotary_dim,
)
from tensorrt_model_connect.families.stablelm.config import ModelConfig
from tensorrt_model_connect.families.stablelm.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(*, raw_updates: dict | None = None, **overrides) -> ModelConfig:
    values = {
        "model_type": "stablelm",
        "architectures": ["StableLmForCausalLM"],
        "vocab_size": 100352,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "_head_dim": 0,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "max_position_embeddings": values["max_position_embeddings"],
        "partial_rotary_factor": 0.25,
        "use_qkv_bias": True,
    }
    raw.update(raw_updates or {})
    values["raw"] = raw
    return ModelConfig(**values)


def test_stablelm_2_1_6b_uses_full_context_native_kv():
    config = _config()

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(config, 4096)

    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert resolved_head_dim(config) == 64
    assert resolved_rotary_dim(config) == 16
    assert row_bytes == 2 * 24 * 32 * 64 * 2
    assert cache_bytes == 4096 * row_bytes


def test_native_graph_uses_update_and_fused_attention_without_concat():
    family_dir = Path(__file__).resolve().parents[4] / (
        "python/tensorrt_model_connect/families/stablelm"
    )
    builder = (family_dir / "native_decoder_builder.py").read_text()
    graph_ops = (family_dir / "graph_ops.py").read_text()

    assert "add_native_kv_cache_attention_from_rows" in builder
    assert "add_kv_cache_update" in graph_ops
    assert "add_attention_v2" in graph_ops
    assert "attention.decomposable = False" in graph_ops
    assert "add_concatenation([cache_" not in builder


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"model_type": "stablelm2"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 2049}, {}, "divisible"),
        ({"num_key_value_heads": 8}, {}, "requires MHA"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({"tie_word_embeddings": True}, {}, "untied"),
        ({}, {"partial_rotary_factor": 0.5}, "partial_rotary_factor=0.25"),
        ({}, {"use_qkv_bias": False}, "use_qkv_bias=true"),
        ({}, {"use_parallel_residual": True}, "sequential residuals"),
        ({}, {"rope_scaling": {"type": "linear"}}, "unsupported"),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_updates, reason):
    decision = native_kv_architecture_capability(_config(raw_updates=raw_updates, **overrides))

    assert decision.applicable
    assert not decision.eligible
    assert reason in decision.reason
    assert not prefer_native_default(_config(raw_updates=raw_updates, **overrides))


def test_foreign_model_type_is_not_applicable():
    decision = native_kv_architecture_capability(_config(model_type="llama"))

    assert not decision.applicable
    assert not decision.eligible


def test_missing_context_limit_does_not_use_the_parser_default():
    config = _config()
    config.raw.pop("max_position_embeddings")

    decision = native_kv_architecture_capability(config)

    assert not decision.eligible
    assert "max_position_embeddings must be explicit" in decision.reason


@pytest.mark.parametrize(
    ("kwargs", "raw_updates", "reason"),
    [
        ({"precision": "fp32"}, {}, "FP16"),
        ({"precision": "bf16"}, {}, "FP16"),
        ({"max_cache_length": 4095}, {}, "max_cache_length"),
        ({"parallel_enabled": True}, {}, "tensor parallel"),
        ({"dynamic_kv_cache": True}, {}, "fixed physical"),
        ({"quantized": True}, {}, "quantized"),
        ({"debug_layer_outputs": True}, {}, "debug"),
        ({}, {"_fp32_layers": ["layer.0"]}, "FP32 layer"),
        ({}, {"_decoder_engine_layout": "dual_profile"}, "split"),
        ({}, {"_rtx_build_requested": True}, "standard TensorRT"),
    ],
)
def test_unqualified_build_modes_use_the_legacy_route(kwargs, raw_updates, reason):
    decision = native_kv_build_capability(
        _config(raw_updates=raw_updates),
        **kwargs,
    )

    assert not decision.eligible
    assert reason in decision.reason


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def _small_config(*, role: str = "prefill") -> ModelConfig:
    return _config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=128,
        raw_updates={"_decoder_engine_role": role},
    )


def _weights(config: ModelConfig) -> dict[str, object]:
    hidden = config.hidden_size
    attention = config.num_attention_heads * resolved_head_dim(config)
    kv_attention = config.num_key_value_heads * resolved_head_dim(config)
    mlp = config.intermediate_size
    weights: dict[str, object] = {
        "embedding": _Tensor((config.vocab_size, hidden)),
        "final_norm": _Tensor((hidden,)),
        "final_norm_beta": _Tensor((hidden,)),
        "w_out": _Tensor((hidden, config.vocab_size)),
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_mlp_size": mlp,
    }
    for name, shape in (
        ("input_norm", (hidden,)),
        ("input_norm_beta", (hidden,)),
        ("w_q", (hidden, attention)),
        ("w_k", (hidden, kv_attention)),
        ("w_v", (hidden, kv_attention)),
        ("q_bias", (attention,)),
        ("k_bias", (kv_attention,)),
        ("v_bias", (kv_attention,)),
        ("w_o", (attention, hidden)),
        ("post_attn_norm", (hidden,)),
        ("post_attn_norm_beta", (hidden,)),
        ("w_gate", (hidden, mlp)),
        ("w_up", (hidden, mlp)),
        ("w_down", (mlp, hidden)),
    ):
        weights[f"layer.0.{name}"] = _Tensor(shape)
    return weights


def test_weight_contract_rejects_missing_shape_and_foreign_weights():
    config = _small_config()
    weights = _weights(config)
    validate_native_kv_weights(config, weights)

    missing = dict(weights)
    missing.pop("layer.0.w_k")
    with pytest.raises(ValueError, match="missing.*w_k"):
        validate_native_kv_weights(config, missing)

    wrong_shape = dict(weights)
    wrong_shape["layer.0.w_q"] = _Tensor((63, 64))
    with pytest.raises(ValueError, match="must have shape"):
        validate_native_kv_weights(config, wrong_shape)

    foreign = dict(weights)
    foreign["layer.0.o_bias"] = _Tensor((64,))
    with pytest.raises(ValueError, match="unsupported mapped weights"):
        validate_native_kv_weights(config, foreign)


def test_plugin_builds_only_the_requested_native_split_role(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module("tensorrt_model_connect.families.stablelm.plugin")
    config = _small_config(role="decode")
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"plan"

    monkeypatch.setattr(plugin_module, "build_native_decoder_engine", _build)
    result = plugin_module.plugin.build_engine(
        config,
        _weights(config),
        128,
        precision="fp16",
    )

    assert result == b"plan"
    assert captured["kwargs"]["profile_mode"] == "decode"
    assert captured["kwargs"]["precision"] == "fp16"
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_explicit_non_native_options_preserve_the_legacy_builder(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module("tensorrt_model_connect.families.stablelm.plugin")
    config = _small_config(role="decode")
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"legacy-plan"

    monkeypatch.setattr(plugin_module, "build_standard_decoder_engine", _build)
    result = plugin_module.plugin.build_engine(
        config,
        _weights(config),
        64,
        precision="fp16",
    )

    assert result == b"legacy-plan"
    assert captured["args"][2] == 64
    assert plugin_module.plugin.get_bundle_config_overrides(config) is None
