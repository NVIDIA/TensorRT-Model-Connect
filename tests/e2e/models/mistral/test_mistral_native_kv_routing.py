# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for Mistral's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import pytest

from tensorrt_model_connect.families.mistral.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
)
from tensorrt_model_connect.families.mistral.config import ModelConfig
from tensorrt_model_connect.families.mistral.native_kv_contract import (
    validate_native_kv_weights,
)


def _config(
    *,
    raw_updates: dict | None = None,
    **overrides,
) -> ModelConfig:
    values = {
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"],
        "vocab_size": 32768,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "_head_dim": 0,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "sliding_window": None,
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
        (3072, 8640, 34, 32, 8, 8192, 128),
        (4096, 14336, 32, 32, 8, 32768, 0),
        (5120, 14336, 40, 32, 8, 131072, 128),
    ],
    ids=("riva-4b-shape", "mistral-7b-shape", "mistral-nemo-12b-shape"),
)
def test_dense_mistral_sizes_share_one_native_contract(
    hidden,
    mlp,
    layers,
    heads,
    kv_heads,
    context,
    head_dim,
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


def test_official_v03_uses_complete_32k_context_by_default():
    config = _config()

    assert config.max_position_embeddings == 32768
    assert native_kv_architecture_capability(config).eligible
    assert native_kv_build_capability(config).eligible
    assert native_kv_cache_geometry(config, 32768)[1] == 4 * 1024**3


def test_family_exposes_only_the_native_kv_implementation():
    repo_root = Path(__file__).resolve().parents[4]
    family_dir = repo_root / "python" / "tensorrt_model_connect" / "families" / "mistral"
    runtime_dir = repo_root / "src" / "runtime" / "models" / "mistral"

    for legacy_builder in (
        "debug_runner.py",
        "default_decoder.py",
        "default_dual_profile_decoder.py",
        "graph_blocks.py",
        "standard_decoder_builder.py",
    ):
        assert not (family_dir / legacy_builder).exists()

    graph_ops = (family_dir / "graph_ops.py").read_text()
    runtime_plugin = (runtime_dir / "plugin.cpp").read_text()
    runtime_helpers = (runtime_dir / "plugin_helpers.h").read_text() + (
        runtime_dir / "plugin_helpers.cpp"
    ).read_text()

    for legacy_graph_op in (
        "add_bias_sum",
        "add_rms_norm_per_head",
        "compute_alibi_slopes",
        "add_layer_norm_native",
        "make_rope_table_half_dim",
        "add_2d_mask_to_4d",
        "add_alibi_mask_4d",
        "add_attention_core",
        "_scalar_constant_for_trt_dtype",
        "add_tanh_softcap",
        "_repeat_kv_heads_4d",
        "_add_attention_core_with_logit_softcap",
        "add_attention_from_rows",
        "add_decoder_attention_ffi",
        "add_tvm_ffi_kernel",
    ):
        assert f"def {legacy_graph_op}(" not in graph_ops

    for legacy_runtime_path in (
        "DualProfileModules",
        "load_dual_profile_modules",
        "create_dual_profile_modules",
        "load_ffi_kernels_from_bundle",
        "tvm_ffi_module_loader",
        "TRTMC_HAS_TVM_FFI",
        "kernel_manifest.json",
        "dynamic_kv_profile_rows",
    ):
        assert legacy_runtime_path not in runtime_plugin + runtime_helpers

    for strict_profile_contract in (
        "modules.modules.front().profile_idx != 0",
        "module->optimization_profile_count() != 1",
        "module->profile_idx() != 0",
        "profile.min != 1 || profile.opt != 1 || profile.max != 1",
        "profile.max <= 1",
        "profile.max > cache_capacity",
    ):
        assert strict_profile_contract in runtime_plugin


def test_explicit_head_dim_supports_decoupled_mistral_widths():
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
        ({"model_type": "mistral3"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 4100}, {}, "divisible"),
        ({"_head_dim": 64}, {}, "head_dim=128"),
        ({"num_key_value_heads": 6}, {}, "divisible"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({}, {"sliding_window": 4096}, "unsupported Mistral fields"),
        ({}, {"num_experts": 8}, "unsupported Mistral fields"),
        ({}, {"pretraining_tp": 2}, "pretraining_tp"),
        (
            {},
            {"layer_types": ["full_attention", "linear_attention"]},
            "hybrid",
        ),
        (
            {},
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "unscaled default RoPE",
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


def test_v01_sliding_window_checkpoint_fails_closed():
    config = _config(
        rope_theta=10_000.0,
        vocab_size=32000,
        raw_updates={"sliding_window": 4096},
    )
    decision = native_kv_architecture_capability(config)

    assert not decision.eligible
    assert "sliding_window" in decision.reason
    assert prefer_native_default(config)


def test_foreign_model_types_do_not_enter_mistral_routing():
    config = _config(model_type="llama")

    decision = native_kv_architecture_capability(config)

    assert not decision.applicable
    assert not decision.eligible
    assert not prefer_native_default(config)


@pytest.mark.parametrize(
    ("kwargs", "raw_updates", "reason"),
    [
        ({"precision": "fp16"}, {}, "BF16"),
        ({"max_cache_length": 32767}, {}, "max_cache_length"),
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
    plugin_module = importlib.import_module("tensorrt_model_connect.families.mistral.plugin")
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
    plugin_module = importlib.import_module("tensorrt_model_connect.families.mistral.plugin")
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
