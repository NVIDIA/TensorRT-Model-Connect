# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for Qwen3's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pytest

from tensorrt_model_connect.families.qwen.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
)
from tensorrt_model_connect.families.qwen.config import ModelConfig
from tensorrt_model_connect.families.qwen.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(*, raw_updates: dict | None = None, **overrides) -> ModelConfig:
    values = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 40960,
        "hidden_act": "silu",
        "_head_dim": 128,
    }
    values.update(overrides)
    raw = {
        "attention_bias": False,
        "sliding_window": None,
        "use_sliding_window": False,
        "rope_scaling": None,
        "_decoder_engine_layout": "split",
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
        (1024, 3072, 28, 16, 8, 40960),
        (2560, 9728, 36, 32, 8, 262144),
        (4096, 12288, 36, 32, 8, 40960),
    ],
    ids=("qwen3-0.6b", "qwen3-4b-shape", "qwen3-8b-shape"),
)
def test_dense_qwen3_sizes_share_one_native_contract(
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
    assert row_bytes == 2 * layers * kv_heads * 128 * 2
    assert cache_bytes == context * row_bytes


def test_route_uses_architecture_not_checkpoint_identity():
    config = _config(
        raw_updates={
            "_model_dir": "/models/renamed-checkpoint",
            "name_or_path": "any-owner/any-qwen3",
            "checkpoint_sha256": "f" * 64,
        }
    )

    assert native_kv_architecture_capability(config).eligible
    assert prefer_native_default(config)


def test_explicit_hf_head_dim_overrides_hidden_divided_by_heads():
    config = _config(
        hidden_size=3584,
        num_attention_heads=32,
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
        ({}, {"use_sliding_window": True}, "unsupported Qwen3 fields"),
        ({}, {"num_experts": 8}, "unsupported Qwen3 fields"),
        (
            {},
            {"layer_types": ["full_attention", "linear_attention"]},
            "hybrid",
        ),
        (
            {},
            {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
            "unscaled default RoPE",
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
        ({"max_cache_length": 40959}, {}, "max_cache_length"),
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
        ("q_norm", (attention,)),
        ("k_norm", (kv_attention,)),
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
        "tensorrt_model_connect.families.qwen.plugin"
    )

    config = _small_config(role="decode")
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"plan"

    monkeypatch.setattr(
        plugin_module,
        "build_dual_profile_decoder_engine",
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
    assert captured["kwargs"]["native_kv_cache"] is True
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }
