# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for GLM's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pytest

from tensorrt_model_connect.families.glm.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
    resolved_partial_rotary_factor,
)
from tensorrt_model_connect.families.glm.config import ModelConfig
from tensorrt_model_connect.families.glm.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(*, raw_updates: dict | None = None, **overrides) -> ModelConfig:
    values = {
        "model_type": "glm",
        "architectures": ["GlmForCausalLM"],
        "vocab_size": 151552,
        "hidden_size": 4096,
        "intermediate_size": 13696,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1.5625e-7,
        "rope_theta": 10000.0,
        "max_position_embeddings": 131072,
        "hidden_act": "silu",
        "_head_dim": 128,
    }
    values.update(overrides)
    raw = {
        "attention_bias": True,
        "partial_rotary_factor": 0.5,
        "rope_scaling": None,
        "_decoder_engine_layout": "split",
    }
    raw.update(raw_updates or {})
    values["raw"] = raw
    return ModelConfig(**values)


def test_official_glm_4_9b_uses_complete_128k_context() -> None:
    config = _config()

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(config, 131072)

    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert resolved_head_dim(config) == 128
    assert resolved_partial_rotary_factor(config) == 0.5
    assert row_bytes == 2 * 40 * 2 * 128 * 2
    assert cache_bytes == 5 * 1024**3


@pytest.mark.parametrize(
    ("hidden", "mlp", "layers", "heads", "kv_heads", "context"),
    [
        (2048, 6912, 24, 16, 2, 32768),
        (4096, 13696, 40, 32, 2, 131072),
        (5120, 17920, 48, 40, 4, 262144),
    ],
)
def test_dense_glm_sizes_share_one_native_contract(
    hidden,
    mlp,
    layers,
    heads,
    kv_heads,
    context,
) -> None:
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
    assert row_bytes == 2 * layers * kv_heads * 128 * 2
    assert cache_bytes == context * row_bytes


def test_route_uses_architecture_not_checkpoint_identity() -> None:
    config = _config(
        raw_updates={
            "_model_dir": "/models/renamed-checkpoint",
            "name_or_path": "any-owner/any-glm",
            "checkpoint_sha256": "f" * 64,
        }
    )

    assert native_kv_architecture_capability(config).eligible
    assert prefer_native_default(config)


def test_partial_rope_defaults_to_hf_glm_value() -> None:
    defaulted = _config(raw_updates={"partial_rotary_factor": None})
    nested = _config(
        raw_updates={
            "partial_rotary_factor": None,
            "rope_parameters": {
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
            },
            "rope_scaling": None,
        }
    )

    assert resolved_partial_rotary_factor(defaulted) == 0.5
    assert resolved_partial_rotary_factor(nested) == 0.5
    assert native_kv_architecture_capability(defaulted).eligible
    assert native_kv_architecture_capability(nested).eligible


def test_explicit_hf_head_dim_overrides_hidden_divided_by_heads() -> None:
    config = _config(
        hidden_size=3584,
        num_attention_heads=28,
        _head_dim=0,
        raw_updates={"head_dim": 128},
    )

    assert resolved_head_dim(config) == 128
    assert native_kv_architecture_capability(config).eligible


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({"_head_dim": 64}, {}, "head_dim=128"),
        ({"num_key_value_heads": 6}, {}, "divisible"),
        ({}, {"attention_bias": False}, "Q/K/V attention biases"),
        ({}, {"num_experts": 8}, "unsupported GLM fields"),
        ({}, {"partial_rotary_factor": 1.0}, "partial_rotary_factor=0.5"),
        ({}, {"interleaved_rope": False}, "interleaved_rope=true"),
        (
            {},
            {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
            "unscaled default RoPE",
        ),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_updates, reason) -> None:
    decision = native_kv_architecture_capability(_config(raw_updates=raw_updates, **overrides))

    assert decision.applicable
    assert not decision.eligible
    assert reason in decision.reason
    assert prefer_native_default(_config(raw_updates=raw_updates, **overrides))


def test_non_glm_family_is_not_routed() -> None:
    config = _config(model_type="qwen3")

    decision = native_kv_architecture_capability(config)
    assert not decision.applicable
    assert not decision.eligible
    assert not prefer_native_default(config)


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
def test_unqualified_build_modes_fail_closed(
    kwargs,
    raw_updates,
    reason,
) -> None:
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
        ("q_bias", (attention,)),
        ("w_k", (hidden, kv_attention)),
        ("k_bias", (kv_attention,)),
        ("w_v", (hidden, kv_attention)),
        ("v_bias", (kv_attention,)),
        ("w_o", (attention, hidden)),
        ("post_attn_norm", (hidden,)),
        ("w_gate", (hidden, mlp)),
        ("w_up", (hidden, mlp)),
        ("w_down", (mlp, hidden)),
    ):
        weights[f"layer.0.{name}"] = _Tensor(shape)
    return weights


def test_weight_contract_requires_qkv_biases_and_exact_shapes() -> None:
    config = _small_config()
    weights = _weights(config)
    validate_native_kv_weights(config, weights)

    missing = dict(weights)
    missing.pop("layer.0.k_bias")
    with pytest.raises(ValueError, match="missing.*k_bias"):
        validate_native_kv_weights(config, missing)

    wrong_shape = dict(weights)
    wrong_shape["layer.0.w_q"] = _Tensor((127, 128))
    with pytest.raises(ValueError, match="must have shape"):
        validate_native_kv_weights(config, wrong_shape)

    q_norm = dict(weights)
    q_norm["layer.0.q_norm"] = _Tensor((128,))
    with pytest.raises(ValueError, match="unsupported weights"):
        validate_native_kv_weights(config, q_norm)


def test_plugin_defaults_to_bf16_and_full_context() -> None:
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module("tensorrt_model_connect.families.glm.plugin")
    config = _small_config()

    assert plugin_module.plugin.default_build_precision(config) == "bf16"
    assert plugin_module.plugin.default_max_cache_length(config) == 256
    assert not hasattr(plugin_module, "build_standard_decoder_engine")
    assert not hasattr(plugin_module, "build_dual_profile_tp_decoder_engine")


def test_plugin_builds_requested_native_split_role(monkeypatch) -> None:
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module("tensorrt_model_connect.families.glm.plugin")
    config = _small_config(role="decode")
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
        precision="bf16",
    )

    assert result == b"plan"
    assert captured["kwargs"]["profile_mode"] == "decode"
    assert "native_kv_cache" not in captured["kwargs"]
    assert "attention_mask" not in captured["kwargs"]
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"precision": "fp16"}, "BF16"),
        ({"max_cache_length": 128}, "max_cache_length"),
        ({"quant_ctx": object()}, "quantized"),
        ({"debug_layer_outputs": True}, "debug"),
    ],
)
def test_plugin_rejects_unsupported_build_without_fallback(
    monkeypatch,
    kwargs,
    reason,
) -> None:
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module("tensorrt_model_connect.families.glm.plugin")
    config = _small_config(role="decode")
    called = False

    def _build(*args, **build_kwargs):
        nonlocal called
        called = True
        return b"unexpected"

    monkeypatch.setattr(
        plugin_module,
        "build_native_decoder_engine",
        _build,
    )
    call_kwargs = {
        "max_cache_length": 256,
        "precision": "bf16",
        **kwargs,
    }

    with pytest.raises(ValueError, match=reason):
        plugin_module.plugin.build_engine(
            config,
            _weights(config),
            **call_kwargs,
        )
    assert not called
