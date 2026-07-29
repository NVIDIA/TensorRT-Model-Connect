# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for Granite's TensorRT native KV-cache path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path

import pytest

from tensorrt_model_connect.families.granite.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
)
from tensorrt_model_connect.families.granite.config import ModelConfig
from tensorrt_model_connect.families.granite.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(**overrides) -> ModelConfig:
    values = {
        "model_type": "granite",
        "architectures": ["GraniteForCausalLM"],
        "vocab_size": 49152,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 5_000_000.0,
        "max_position_embeddings": 131072,
        "hidden_act": "silu",
        "_head_dim": 0,
        "raw": {
            "_decoder_engine_layout": "split",
            "attention_bias": False,
            "mlp_bias": False,
            "rope_scaling": None,
        },
    }
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    ("hidden", "mlp", "layers", "heads", "kv_heads", "head_dim"),
    [
        (2048, 8192, 40, 32, 8, 64),
        (4096, 12800, 40, 32, 8, 128),
    ],
    ids=("granite-3.1-2b", "granite-3.1-8b"),
)
def test_dense_granite_sizes_share_the_full_context_contract(
    hidden,
    mlp,
    layers,
    heads,
    kv_heads,
    head_dim,
):
    config = _config(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
    )

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(config, 131072)

    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert resolved_head_dim(config) == head_dim
    assert row_bytes == 2 * layers * kv_heads * head_dim * 2
    assert cache_bytes == 131072 * row_bytes


@pytest.mark.parametrize(
    ("overrides", "raw_update", "reason"),
    [
        ({"model_type": "granite_moe"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 2112}, {}, "head_dim"),
        ({"num_key_value_heads": 6}, {}, "divisible"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({}, {"sliding_window": 4096}, "unsupported Granite fields"),
        ({}, {"num_experts": 8}, "unsupported Granite fields"),
        ({}, {"attention_bias": True}, "unsupported Granite fields"),
        (
            {},
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "rope_type",
        ),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_update, reason):
    config = _config(**overrides)
    config.raw.update(raw_update)

    decision = native_kv_architecture_capability(config)

    assert decision.applicable
    assert not decision.eligible
    assert prefer_native_default(config)
    assert reason in decision.reason


@pytest.mark.parametrize(
    ("kwargs", "raw_update", "reason"),
    [
        ({"precision": "bf16"}, {}, "FP16"),
        ({"max_cache_length": 131071}, {}, "max_cache_length"),
        ({"parallel_enabled": True}, {}, "tensor parallel"),
        ({"dynamic_kv_cache": True}, {}, "fixed physical"),
        ({"quantized": True}, {}, "quantized"),
        ({"debug_layer_outputs": True}, {}, "debug"),
        ({}, {"_decoder_engine_layout": "dual_profile"}, "split"),
    ],
)
def test_unqualified_build_modes_fail_closed(kwargs, raw_update, reason):
    config = _config()
    config.raw.update(raw_update)

    decision = native_kv_build_capability(config, **kwargs)

    assert not decision.eligible
    assert reason in decision.reason


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def _small_config(*, role: str = "prefill") -> ModelConfig:
    config = _config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=256,
    )
    config.raw["_decoder_engine_role"] = role
    return config


def _weights(config: ModelConfig) -> dict[str, object]:
    hidden = config.hidden_size
    head_dim = resolved_head_dim(config)
    attention = config.num_attention_heads * head_dim
    kv_attention = config.num_key_value_heads * head_dim
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
    wrong_shape["layer.0.w_q"] = _Tensor((63, 64))
    with pytest.raises(ValueError, match="must have shape"):
        validate_native_kv_weights(config, wrong_shape)

    biased = dict(weights)
    biased["layer.0.q_bias"] = _Tensor((64,))
    with pytest.raises(ValueError, match="bias"):
        validate_native_kv_weights(config, biased)


def test_plugin_builds_native_split_roles_and_never_falls_back(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module("tensorrt_model_connect.families.granite.plugin")
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

    assert plugin_module.plugin.default_build_precision(config) == "fp16"
    assert plugin_module.plugin.default_max_cache_length(config) == 256
    assert (
        plugin_module.plugin.build_engine(
            config,
            _weights(config),
            256,
            precision="fp16",
        )
        == b"plan"
    )
    assert captured["kwargs"]["profile_mode"] == "decode"
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }

    with pytest.raises(NotImplementedError, match="only.*native KV-cache"):
        plugin_module.plugin.build_engine(
            config,
            _weights(config),
            256,
            precision="bf16",
        )


def test_manifest_uses_family_defaults_for_precision_and_full_context():
    manifest_path = Path(__file__).parent / "manifests" / "granite-3.1-2b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "precision" not in manifest
    assert "max_cache_length" not in manifest
