# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for InternLM2's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pytest

from tensorrt_model_connect.families.internlm.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
)
from tensorrt_model_connect.families.internlm.config import ModelConfig
from tensorrt_model_connect.families.internlm.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(
    *,
    raw_updates: dict | None = None,
    **overrides,
) -> ModelConfig:
    values = {
        "model_type": "internlm2",
        "architectures": ["InternLM2ForCausalLM"],
        "vocab_size": 92544,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 8192,
        "hidden_act": "silu",
        "_head_dim": 0,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "bias": False,
        "sliding_window": None,
        "rope_scaling": {"type": "dynamic", "factor": 1.0},
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
        "head_dim",
    ),
    [
        (2048, 8192, 24, 16, 8, 8192, 0),
        (4096, 14336, 32, 32, 8, 32768, 0),
        (6144, 16384, 48, 48, 8, 32768, 0),
    ],
    ids=("internlm2-1.8b", "internlm2-7b", "internlm2-20b"),
)
def test_dense_internlm2_sizes_share_one_native_contract(
    hidden, mlp, layers, heads, kv_heads, context, head_dim,
):
    config = _config(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=context,
        _head_dim=head_dim,
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


def test_official_math_plus_1_8b_uses_complete_8k_context_by_default():
    config = _config()

    assert config.max_position_embeddings == 8192
    assert native_kv_architecture_capability(config).eligible
    assert native_kv_build_capability(config).eligible
    assert native_kv_cache_geometry(config, 8192)[1] == 768 * 1024**2


def test_explicit_head_dim_supports_decoupled_internlm2_widths():
    config = _config(
        hidden_size=1024,
        num_attention_heads=8,
        _head_dim=128,
    )

    assert resolved_head_dim(config) == 128
    assert native_kv_architecture_capability(config).eligible


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"model_type": "internlm"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 2050}, {}, "divisible"),
        ({"_head_dim": 64}, {}, "head_dim=128"),
        ({"num_key_value_heads": 6}, {}, "divisible"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({}, {"bias": True}, "explicit bias=false"),
        ({}, {"sliding_window": 4096}, "unsupported InternLM fields"),
        ({}, {"num_experts": 8}, "unsupported InternLM fields"),
        ({}, {"pretraining_tp": 2}, "pretraining_tp"),
        (
            {},
            {"layer_types": ["full_attention", "linear_attention"]},
            "hybrid",
        ),
        (
            {},
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "default or dynamic-NTK",
        ),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_updates, reason):
    config = _config(raw_updates=raw_updates, **overrides)
    decision = native_kv_architecture_capability(config)

    assert decision.applicable
    assert not decision.eligible
    assert reason in decision.reason
    assert prefer_native_default(config)


def test_foreign_model_types_do_not_enter_internlm_routing():
    config = _config(model_type="llama")

    decision = native_kv_architecture_capability(config)

    assert not decision.applicable
    assert not decision.eligible
    assert not prefer_native_default(config)


def test_missing_bias_uses_internlm2s_bias_enabled_default_and_fails_closed():
    config = _config()
    config.raw.pop("bias")

    decision = native_kv_architecture_capability(config)

    assert decision.applicable
    assert not decision.eligible
    assert "explicit bias=false" in decision.reason


@pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
def test_dynamic_ntk_is_exact_within_fixed_official_capacity(factor):
    decision = native_kv_architecture_capability(
        _config(raw_updates={
            "rope_scaling": {"type": "dynamic", "factor": factor},
        })
    )

    assert decision.eligible, decision.reason


@pytest.mark.parametrize(
    ("kwargs", "raw_updates", "reason"),
    [
        ({"precision": "fp16"}, {}, "BF16"),
        ({"max_cache_length": 8191}, {}, "max_cache_length"),
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


def test_plugin_builds_only_the_requested_native_split_role(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internlm.plugin"
    )
    config = _small_config(role="prefill")
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"plan"

    monkeypatch.setattr(
        plugin_module,
        "build_native_decoder_engine",
        _build,
    )

    result = plugin_module.plugin.build_engine(
        config,
        _weights(config),
        256,
    )

    assert result == b"plan"
    assert captured["kwargs"]["profile_mode"] == "prefill"
    assert captured["kwargs"]["precision"] == "bf16"
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_plugin_never_falls_back_to_a_legacy_builder(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internlm.plugin"
    )
    config = _small_config(role="decode")
    called = False

    def _build(*args, **kwargs):
        nonlocal called
        called = True
        return b"unexpected"

    monkeypatch.setattr(
        plugin_module,
        "build_native_decoder_engine",
        _build,
    )

    with pytest.raises(ValueError, match="requires BF16"):
        plugin_module.plugin.build_engine(
            config,
            _weights(config),
            256,
            precision="fp16",
        )
    assert not called
    assert plugin_module.plugin.get_bundle_config_overrides(config) is None
