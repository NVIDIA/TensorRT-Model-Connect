# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for Llama's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pytest

from tensorrt_model_connect.models.llama.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
)
from tensorrt_model_connect.models.llama.config import ModelConfig
from tensorrt_model_connect.models.llama.native_kv_contract import (
    validate_native_kv_weights,
)


_LLAMA3_ROPE = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _config(
    *,
    raw_updates: dict | None = None,
    llama3_rope: bool = True,
    **overrides,
) -> ModelConfig:
    values = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000.0,
        "max_position_embeddings": 131072,
        "hidden_act": "silu",
        "_head_dim": 0,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "rope_scaling": dict(_LLAMA3_ROPE) if llama3_rope else None,
    }
    raw.update(raw_updates or {})
    values["raw"] = raw
    return ModelConfig(**values)


@pytest.mark.parametrize(
    (
        "hidden",
        "mlp",
        "layers",
        "heads",
        "kv_heads",
        "context",
    ),
    [
        (4096, 14336, 32, 32, 8, 131072),
        (5120, 13824, 40, 40, 40, 4096),
        (8192, 28672, 80, 64, 8, 131072),
    ],
    ids=("llama-8b-shape", "llama-13b-shape", "llama-70b-shape"),
)
def test_dense_llama_sizes_share_one_native_contract(
    hidden, mlp, layers, heads, kv_heads, context,
):
    config = _config(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=context,
    )

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(config, context)

    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert resolved_head_dim(config) == 128
    assert row_bytes == 2 * layers * kv_heads * 128 * 2
    assert cache_bytes == context * row_bytes


def test_route_uses_architecture_not_checkpoint_identity():
    config = _config(
        raw_updates={
            "_model_dir": "/models/renamed-checkpoint",
            "name_or_path": "any-owner/any-llama",
            "checkpoint_sha256": "a" * 64,
        }
    )

    assert native_kv_architecture_capability(config).eligible
    assert prefer_native_default(config)


def test_explicit_head_dim_is_supported_when_hidden_width_is_decoupled():
    config = _config(
        hidden_size=3072,
        num_attention_heads=32,
        _head_dim=128,
    )

    assert resolved_head_dim(config) == 128
    assert native_kv_architecture_capability(config).eligible


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"model_type": "llama4"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 4100}, {}, "divisible"),
        ({"_head_dim": 64}, {}, "head_dim=128"),
        ({"num_key_value_heads": 6}, {}, "divisible"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({}, {"sliding_window": 4096}, "unsupported Llama fields"),
        ({}, {"num_experts": 8}, "unsupported Llama fields"),
        ({}, {"pretraining_tp": 2}, "pretraining_tp"),
        (
            {},
            {"layer_types": ["full_attention", "linear_attention"]},
            "hybrid",
        ),
        (
            {},
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "rope_type",
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
        ({"max_cache_length": 131071}, {}, "max_cache_length"),
        ({"parallel_enabled": True}, {}, "tensor parallel"),
        ({"dynamic_kv_cache": True}, {}, "fixed physical"),
        ({"quantized": True}, {}, "quantized"),
        ({"debug_layer_outputs": True}, {}, "debug"),
        ({}, {"_fp32_layers": ["layer.0"]}, "FP32 layer"),
        ({}, {"_decoder_engine_layout": "dual_profile"}, "split"),
        ({}, {"_rtx_build_requested": True}, "standard TensorRT"),
    ],
)
def test_unqualified_build_modes_fail_closed(kwargs, raw_updates, reason):
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
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=256,
        llama3_rope=False,
        raw_updates={"_decoder_engine_role": role},
    )


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
        weights[f"layer.0.{name}"] = _Tensor(shape)
    return weights


def test_weight_contract_rejects_missing_shape_and_bias():
    config = _small_config()
    weights = _weights(config)
    validate_native_kv_weights(config, weights)

    missing = dict(weights)
    missing.pop("layer.0.w_k")
    with pytest.raises(ValueError, match="missing.*w_k"):
        validate_native_kv_weights(config, missing)

    wrong_shape = dict(weights)
    wrong_shape["layer.0.w_q"] = _Tensor((127, 128))
    with pytest.raises(ValueError, match="must have shape"):
        validate_native_kv_weights(config, wrong_shape)

    biased = dict(weights)
    biased["layer.0.q_bias"] = _Tensor((128,))
    with pytest.raises(ValueError, match="bias"):
        validate_native_kv_weights(config, biased)


def test_plugin_builds_the_requested_split_role_directly(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.llama.model"
    )

    config = _small_config(role="prefill")
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"plan"

    monkeypatch.setattr(
        plugin_module,
        "build_dual_profile_decoder_engine",
        _build,
    )

    result = plugin_module.build_engine(
        config,
        _weights(config),
        256,
        precision="bf16",
    )

    assert result == b"plan"
    assert captured["kwargs"]["profile_mode"] == "prefill"
    assert captured["kwargs"]["native_kv_cache"] is True
    assert plugin_module.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_plugin_falls_back_for_explicit_legacy_build_options(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.llama.model"
    )

    config = _small_config(role="decode")
    config.raw["_native_kv_cache_metadata"] = {"stale": True}
    quant_ctx = object()
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"legacy-plan"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        _build,
    )

    result = plugin_module.build_engine(
        config,
        _weights(config),
        128,
        precision="fp16",
        quant_ctx=quant_ctx,
    )

    assert result == b"legacy-plan"
    assert captured["args"][2] == 128
    assert captured["kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["quant_ctx"] is quant_ctx
    assert plugin_module.get_bundle_config_overrides(config) is None


def test_plugin_falls_back_outside_the_native_architecture_contract(
    monkeypatch,
):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.llama.model"
    )
    config = _small_config()
    config._head_dim = 64
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"legacy-plan"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        _build,
    )

    assert not prefer_native_default(config)
    assert plugin_module.default_build_precision(config) == "fp32"
    assert plugin_module.default_max_cache_length(config) == 256
    assert plugin_module.build_engine(
        config,
        _weights(config),
        128,
        precision="fp16",
    ) == b"legacy-plan"
    assert captured["args"][2] == 128
    assert captured["kwargs"]["precision"] == "fp16"
