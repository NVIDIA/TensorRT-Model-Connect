# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for GPT-NeoX's TensorRT native KV path."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
from pathlib import Path

import pytest

from tensorrt_model_connect.families.gpt_neox.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
    resolved_rotary_dim,
)
from tensorrt_model_connect.families.gpt_neox.config import ModelConfig
from tensorrt_model_connect.families.gpt_neox.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(
    *,
    raw_updates: dict | None = None,
    **overrides,
) -> ModelConfig:
    values = {
        "model_type": "gpt_neox",
        "architectures": ["GPTNeoXForCausalLM"],
        "vocab_size": 50304,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
        "hidden_act": "gelu",
        "tie_word_embeddings": False,
        "_head_dim": 0,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "rotary_pct": 0.25,
        "use_parallel_residual": True,
    }
    raw.update(raw_updates or {})
    values["raw"] = raw
    return ModelConfig(**values)


@pytest.mark.parametrize(
    ("hidden", "mlp", "layers", "heads", "head_dim"),
    [
        (512, 2048, 6, 8, 64),
        (2048, 8192, 24, 16, 128),
        (2560, 10240, 32, 32, 80),
        (4096, 16384, 32, 32, 128),
        (5120, 20480, 36, 40, 128),
    ],
    ids=("pythia-70m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"),
)
def test_supported_pythia_sizes_share_one_native_contract(
    hidden,
    mlp,
    layers,
    heads,
    head_dim,
):
    config = _config(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
    )

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(config, 2048)

    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert resolved_head_dim(config) == head_dim
    assert resolved_rotary_dim(config) == int(head_dim * 0.25)
    assert row_bytes == 2 * layers * heads * head_dim * 2
    assert cache_bytes == 2048 * row_bytes


def test_pythia_70m_uses_its_complete_2048_context_by_default():
    config = _config()

    assert config.max_position_embeddings == 2048
    assert native_kv_architecture_capability(config).eligible
    assert native_kv_build_capability(config).eligible
    assert native_kv_cache_geometry(config, 2048) == (
        12288,
        24 * 1024**2,
    )


def test_missing_context_limit_is_not_replaced_by_a_hidden_default():
    config = ModelConfig.from_json(
        """{
          "model_type": "gpt_neox",
          "architectures": ["GPTNeoXForCausalLM"],
          "vocab_size": 32,
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 1,
          "num_attention_heads": 1,
          "hidden_act": "gelu"
        }"""
    )

    assert config.max_position_embeddings == 0
    decision = native_kv_architecture_capability(config)
    assert not decision.eligible
    assert "max_position_embeddings must be positive" in decision.reason


def test_native_graph_has_no_concat_or_decomposable_fallback():
    family_dir = Path(__file__).resolve().parents[4] / (
        "python/tensorrt_model_connect/families/gpt_neox"
    )
    builder = (family_dir / "native_decoder_builder.py").read_text()
    graph_ops = (family_dir / "graph_ops.py").read_text()

    assert "add_native_kv_cache_attention_from_rows" in builder
    assert "add_kv_cache_update" in graph_ops
    assert "add_attention_v2" in graph_ops
    assert "attention.decomposable = False" in graph_ops
    assert "add_concatenation([cache_" not in builder


def test_native_graph_keeps_the_residual_stream_in_fp32():
    family_dir = Path(__file__).resolve().parents[4] / (
        "python/tensorrt_model_connect/families/gpt_neox"
    )
    builder = (family_dir / "native_decoder_builder.py").read_text()
    tree = ast.parse(builder)
    matmul = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "matmul"
    )
    matmul_source = ast.unparse(matmul)
    norm_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "norm_multi"
    ]

    assert "hidden_state = network.add_cast(" in builder
    assert "attn_out = network.add_cast(" in builder
    assert "mlp_out = network.add_cast(" in builder
    assert "physical KV cache remain FP16" in builder
    assert "if lhs.dtype != work_trt_dtype" in matmul_source
    assert "network.add_cast(lhs, work_trt_dtype)" in matmul_source
    assert len(norm_calls) == 4
    assert all(ast.unparse(call.args[-1]) == "np.float32" for call in norm_calls)


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"model_type": "gptneox"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 513}, {}, "divisible"),
        ({"max_position_embeddings": 0}, {}, "must be positive"),
        (
            {
                "hidden_size": 2048,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
            },
            {},
            "no larger than 128",
        ),
        ({"num_key_value_heads": 4}, {}, "requires MHA"),
        ({"hidden_act": "silu"}, {}, "hidden_act"),
        ({"tie_word_embeddings": True}, {}, "untied"),
        ({}, {"rotary_pct": 0.01}, "even rotary dimension"),
        ({}, {"rotary_pct": 1.25}, "in (0, 1]"),
        ({}, {"rope_scaling": {"type": "linear", "factor": 2.0}}, "unsupported"),
        ({}, {"use_parallel_residual": "yes"}, "boolean"),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_updates, reason):
    config = _config(raw_updates=raw_updates, **overrides)
    decision = native_kv_architecture_capability(config)

    assert decision.applicable
    assert not decision.eligible
    assert reason in decision.reason
    assert prefer_native_default(config)


def test_foreign_model_types_do_not_enter_gpt_neox_routing():
    config = _config(model_type="llama")

    decision = native_kv_architecture_capability(config)

    assert not decision.applicable
    assert not decision.eligible
    assert not prefer_native_default(config)


@pytest.mark.parametrize(
    ("kwargs", "raw_updates", "reason"),
    [
        ({"precision": "fp32"}, {}, "FP16"),
        ({"precision": "bf16"}, {}, "FP16"),
        ({"max_cache_length": 2047}, {}, "max_cache_length"),
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
    mlp = config.intermediate_size
    weights: dict[str, object] = {
        "embedding": _Tensor((config.vocab_size, hidden)),
        "final_norm": _Tensor((hidden,)),
        "final_norm_beta": _Tensor((hidden,)),
        "w_out": _Tensor((hidden, config.vocab_size)),
        "_attention_size": attention,
        "_kv_attention_size": attention,
        "_mlp_size": mlp,
    }
    for name, shape in (
        ("input_norm", (hidden,)),
        ("input_norm_beta", (hidden,)),
        ("w_q", (hidden, attention)),
        ("w_k", (hidden, attention)),
        ("w_v", (hidden, attention)),
        ("q_bias", (attention,)),
        ("k_bias", (attention,)),
        ("v_bias", (attention,)),
        ("w_o", (attention, hidden)),
        ("o_bias", (hidden,)),
        ("post_attn_norm", (hidden,)),
        ("post_attn_norm_beta", (hidden,)),
        ("w_fc1", (hidden, mlp)),
        ("fc1_bias", (mlp,)),
        ("w_fc2", (mlp, hidden)),
        ("fc2_bias", (hidden,)),
    ):
        weights[f"layer.0.{name}"] = _Tensor(shape)
    return weights


def test_weight_contract_rejects_missing_wrong_shape_and_foreign_weights():
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
    foreign["layer.0.q_norm"] = _Tensor((64,))
    with pytest.raises(ValueError, match="unsupported mapped weights"):
        validate_native_kv_weights(config, foreign)


def test_plugin_builds_only_the_requested_native_split_role(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.gpt_neox.plugin"
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
        128,
    )

    assert result == b"plan"
    assert captured["kwargs"]["profile_mode"] == "prefill"
    assert captured["kwargs"]["precision"] == "fp16"
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_plugin_never_falls_back_to_a_legacy_builder(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.gpt_neox.plugin"
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

    with pytest.raises(ValueError, match="requires FP16"):
        plugin_module.plugin.build_engine(
            config,
            _weights(config),
            128,
            precision="fp32",
        )
    assert not called
    assert plugin_module.plugin.get_bundle_config_overrides(config) is None
