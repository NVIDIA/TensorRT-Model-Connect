# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StableLM precision boundaries required by continuation parity."""

from __future__ import annotations

import importlib

import numpy as np
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


@pytest.mark.parametrize(
    ("precision", "fp32_layers", "expected"),
    (
        ("fp16", frozenset({1}), True),
        ("fp16", frozenset({0}), False),
        ("fp16", frozenset(), False),
        ("bf16", frozenset({1}), False),
        ("fp32", frozenset({1}), False),
    ),
)
def test_only_fp16_terminal_precision_boundary_stabilizes_the_lm_head(
    precision: str, fp32_layers: frozenset[int], expected: bool,
) -> None:
    builder_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.default_decoder"
    )

    assert builder_module._use_fp32_lm_head_accumulation(
        precision, num_layers=2, fp32_layers=fp32_layers
    ) is expected


@pytest.mark.parametrize(
    ("fp32_accumulation", "expected_matmul_dtype", "expected_casts"),
    (
        (False, np.float16, 1),
        (True, np.float32, 3),
    ),
)
def test_decode_lm_head_stabilizes_accumulation_without_changing_output_precision(
    monkeypatch,
    fp32_accumulation: bool,
    expected_matmul_dtype,
    expected_casts: int,
) -> None:
    builder_module = importlib.import_module(
        "tensorrt_model_connect.families.stablelm.default_decoder"
    )
    trt = builder_module.trt

    class FakeTensor:
        def __init__(self, dtype) -> None:
            self.dtype = dtype

    class FakeCast:
        def __init__(self, tensor: FakeTensor) -> None:
            self.tensor = tensor

        def get_output(self, index: int) -> FakeTensor:
            assert index == 0
            return self.tensor

    class FakeNetwork:
        def __init__(self) -> None:
            self.casts: list[tuple[object, object]] = []

        def add_cast(self, tensor: FakeTensor, dtype) -> FakeCast:
            self.casts.append((tensor.dtype, dtype))
            return FakeCast(FakeTensor(dtype))

    captured: dict[str, object] = {}

    def fake_matmul(network, lhs, lhs_width, rhs_width, rhs_weights, *, dtype):
        del network, lhs_width, rhs_width
        captured["matmul_input_dtype"] = lhs.dtype
        captured["matmul_dtype"] = dtype
        captured["matmul_weights"] = rhs_weights.copy()
        return FakeTensor(lhs.dtype)

    def fake_bias(network, inp, width, bias, *, dtype):
        del network, width, bias
        captured["bias_input_dtype"] = inp.dtype
        captured["bias_dtype"] = dtype
        return FakeTensor(inp.dtype)

    monkeypatch.setattr(
        builder_module.graph_ops, "add_matmul_rhs_constant", fake_matmul)
    monkeypatch.setattr(builder_module.graph_ops, "add_bias_sum", fake_bias)

    network = FakeNetwork()
    source_weights = np.linspace(0.1, 3.2, 32, dtype=np.float32).reshape(4, 8)
    output = builder_module._add_lm_head(
        network,
        FakeTensor(trt.float16),
        WeightDict(w_out=source_weights),
        hidden_size=4,
        out_vocab=8,
        work_np_dtype=np.float16,
        work_trt_dtype=trt.float16,
        fp32_accumulation=fp32_accumulation,
    )

    expected_compute_dtype = trt.float32 if fp32_accumulation else trt.float16
    assert captured["matmul_input_dtype"] == expected_compute_dtype
    assert captured["matmul_dtype"] == expected_matmul_dtype
    assert captured["bias_input_dtype"] == expected_compute_dtype
    assert captured["bias_dtype"] == expected_matmul_dtype
    expected_weights = (
        source_weights.astype(np.float16).astype(np.float32)
        if fp32_accumulation
        else source_weights
    )
    np.testing.assert_array_equal(captured["matmul_weights"], expected_weights)
    assert network.casts == (
        [
            (trt.float16, trt.float32),
            (trt.float32, trt.float16),
            (trt.float16, trt.float32),
        ]
        if fp32_accumulation
        else [(trt.float16, trt.float32)]
    )
    assert len(network.casts) == expected_casts
    assert output.dtype == trt.float32


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
