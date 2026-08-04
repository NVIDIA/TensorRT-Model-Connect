# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed routing tests for DeepSeek-OCR native KV."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.deepseek_ocr.build_routing import (
    native_kv_cache_bytes,
    validate_native_kv_build,
)
from tensorrt_model_connect.parallel_config import ParallelConfig


def _config(**raw_overrides) -> SimpleNamespace:
    raw = {
        "_decoder_engine_layout": "split",
        "language_config": {"use_mla": False},
        **raw_overrides,
    }
    return SimpleNamespace(
        model_type="deepseek_vl_v2",
        raw=raw,
        vocab_size=129280,
        hidden_size=1280,
        intermediate_size=6848,
        num_hidden_layers=12,
        num_attention_heads=10,
        num_key_value_heads=10,
        head_dim=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def _validate(config: SimpleNamespace, **overrides) -> None:
    validate_native_kv_build(
        config,
        precision=overrides.get("precision", "bf16"),
        max_cache_length=overrides.get("max_cache_length", 8192),
        parallel=overrides.get("parallel", ParallelConfig()),
        quantized=overrides.get("quantized", False),
        debug_layer_outputs=overrides.get("debug_layer_outputs", False),
    )


def test_official_context_uses_one_full_bf16_cache() -> None:
    config = _config()
    _validate(config)
    assert native_kv_cache_bytes(config, 8192) == 503316480


def test_tp_cache_geometry_is_rank_local() -> None:
    config = _config()
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0)
    _validate(config, parallel=parallel)
    assert native_kv_cache_bytes(config, 8192, tp_size=2) == 251658240


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision": "fp16"}, "requires BF16"),
        ({"max_cache_length": 4096}, "max_position_embeddings"),
        ({"quantized": True}, "quantized"),
        ({"debug_layer_outputs": True}, "debug layer outputs"),
    ],
)
def test_unsupported_builds_fail_closed(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_config(), **kwargs)


def test_runtime_sized_and_legacy_layout_requests_fail_closed() -> None:
    with pytest.raises(ValueError, match="fixed full-context capacity"):
        _validate(_config(_runtime_dynamic_kv_requested=True))
    with pytest.raises(ValueError, match="split prefill/decode"):
        _validate(_config(_decoder_engine_layout="single"))


def test_deepseek_ocr_runtime_reads_boolean_vision_contract() -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "src/runtime/models/deepseek_ocr/plugin.cpp"
    ).read_text(encoding="utf-8")
    assert 'extract_json_bool(ctx.config_json, "has_vision_engine", false)' in source
    assert 'extract_json_int(ctx.config_json, "has_vision_engine"' not in source
    assert "declared_in_config || plan != nullptr" in source
