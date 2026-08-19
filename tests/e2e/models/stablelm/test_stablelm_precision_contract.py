# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StableLM precision boundaries required by continuation parity."""

from __future__ import annotations

import importlib

import pytest


pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="stablelm",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        raw={"partial_rotary_factor": 0.25},
    )


@pytest.mark.parametrize(
    ("precision", "expected_fp32_accumulation"),
    (("fp16", True), ("bf16", False), ("fp32", False)),
)
def test_stablelm_attention_accumulation_matches_reference_contract(
    monkeypatch,
    precision: str,
    expected_fp32_accumulation: bool,
) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.plugin"
    )
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured.update(kwargs)
        return b"engine"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        fake_build,
    )

    plan = plugin_module.StableLMPlugin().build_engine(
        _config(),
        WeightDict(),
        max_cache_length=64,
        precision=precision,
    )

    assert plan == b"engine"
    assert (
        captured["fp32_attention_accumulation"]
        is expected_fp32_accumulation
    )


@pytest.mark.parametrize(
    ("precision", "fp32_layers"),
    (
        ("fp16", ()),
        ("bf16", (1,)),
        ("bf16", ()),
    ),
)
def test_precision_boundaries_support_asymmetric_split_engines(
    precision: str,
    fp32_layers: tuple[int, ...],
) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.plugin"
    )
    config = _config()
    config.raw["_resolved_build_precision"] = precision
    config.raw["_fp32_layers"] = list(fp32_layers)

    assert plugin_module.StableLMPlugin().supports_split_decoder_roles(config)


def test_fp16_prefill_keeps_the_dynamic_graph_homogeneous(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.plugin"
    )
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured.update(kwargs)
        return b"engine"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        fake_build,
    )
    config = _config()
    config.raw["_decoder_engine_role"] = "prefill"

    plugin_module.StableLMPlugin().build_engine(
        config,
        WeightDict(),
        max_cache_length=64,
        precision="fp16",
    )

    assert captured["fp32_attention_accumulation"] is False


def test_prefill_ignores_decode_only_fp32_layers(monkeypatch) -> None:
    builder_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.default_decoder"
    )
    called = False

    def fake_dual_build(*args, **kwargs):
        nonlocal called
        called = True
        return b"prefill-engine"

    monkeypatch.setattr(
        builder_module,
        "build_dual_profile_decoder_engine",
        fake_dual_build,
    )
    config = _config()
    config.raw["_decoder_engine_role"] = "prefill"
    config.raw["_fp32_layers"] = [1]

    plan = builder_module.build_standard_decoder_engine(
        config,
        WeightDict(),
        max_cache_length=64,
        precision="fp16",
        fp32_attention_accumulation=False,
    )

    assert plan == b"prefill-engine"
    assert called


def test_dual_profile_rejects_fp32_precision_boundary() -> None:
    builder_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.default_decoder"
    )
    config = _config()
    config.raw["_decoder_engine_role"] = "dual_profile"

    with pytest.raises(NotImplementedError, match="FP32 precision boundaries"):
        builder_module.build_standard_decoder_engine(
            config,
            WeightDict(),
            max_cache_length=64,
            precision="fp16",
            fp32_attention_accumulation=True,
        )
