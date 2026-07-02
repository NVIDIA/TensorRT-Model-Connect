# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantization framework abstractions.

Trace: ARCH-QUANT-001, UD-QUANT-FRAMEWORK
Intent: Validate quantization format registry, scale map JSON round-trip, and format protocol compliance
Preconditions: Quantization formats (FP8, INT8, INT4, NVFP4, W4A8) are registered
Postconditions: All formats are discoverable, scale maps survive JSON serialization, and formats expose wrap_matmul
"""
import json
import numpy as np
import pytest

from tensorrt_model_connect.quantization import get_format, list_formats, QuantScaleMap, LayerScales
from tensorrt_model_connect.quantization.plan import QuantPlan, canonicalize_quant_format
from tensorrt_model_connect.quantization.formats import QuantFormat
from tensorrt_model_connect.quantization.profile import QuantProfile


class TestFormatRegistry:
    def test_all_formats_registered(self):
        names = list_formats()
        assert "fp8" in names
        assert "int8_sq" in names
        assert "int4_awq" in names
        assert "nvfp4" in names
        assert "w4a8" in names

    def test_get_format_returns_protocol(self):
        fmt = get_format("fp8")
        assert isinstance(fmt, QuantFormat)
        assert fmt.name == "fp8"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_format("nonexistent")


class TestScaleMapJsonRoundTrip:
    def test_roundtrip(self):
        original = QuantScaleMap(scales={
            "layer.0.w_q": LayerScales(input_scale=0.042, weight_scale=0.051),
            "layer.1.w_k": LayerScales(input_scale=0.1, weight_scale=0.2, block_size=128),
        })
        restored = QuantScaleMap.from_json(original.to_json())
        assert len(restored.scales) == 2
        assert abs(restored.scales["layer.0.w_q"].input_scale - 0.042) < 1e-6
        assert restored.scales["layer.1.w_k"].block_size == 128

    def test_dynamic_flag(self):
        m = QuantScaleMap(scales={}, dynamic=True)
        restored = QuantScaleMap.from_json(m.to_json())
        assert restored.dynamic is True

    def test_family_scoped_keys_resolve_by_suffix(self):
        m = QuantScaleMap(scales={
            "unit_family/layer.0.w_q": LayerScales(input_scale=0.25, weight_scale=0.5),
        })
        entry = m.get("layer.0.w_q")
        assert entry is not None
        assert abs(entry.input_scale - 0.25) < 1e-6


class TestQuantPlan:
    def test_aliases_canonicalize(self):
        assert canonicalize_quant_format("int8") == "int8_sq"
        assert canonicalize_quant_format("int4") == "int4_awq"
        assert canonicalize_quant_format("fp8") == "fp8"

    def test_plan_infers_scale_source(self):
        plan = QuantPlan.from_build_args(
            precision="bf16",
            quantize="int8",
            quant_scales=None,
            quant_calibration_samples=32,
        )
        assert plan.quant_format == "int8_sq"
        assert plan.scale_source == "modelopt"
        assert plan.calibration_samples == 32

    def test_plan_uses_precomputed_source(self):
        plan = QuantPlan.from_build_args(
            precision="fp16",
            quantize="fp8",
            quant_scales="/tmp/scales.json",
        )
        assert plan.scale_source == "precomputed"
        assert plan.scale_artifact == "/tmp/scales.json"


class TestQuantFormatProtocol:
    def test_all_formats_have_wrap_matmul(self):
        for name in list_formats():
            fmt = get_format(name)
            assert hasattr(fmt, "wrap_matmul"), f"{name} missing wrap_matmul"

    def test_all_formats_have_wrap_conv2d(self):
        for name in list_formats():
            fmt = get_format(name)
            assert hasattr(fmt, "wrap_conv2d"), f"{name} missing wrap_conv2d"


class TestQuantContextGraphOpsOwnership:
    def test_plain_matmul_uses_injected_family_graph_ops(self):
        from tensorrt_model_connect.quantization.context import QuantContext

        calls = []

        class FakeGraphOps:
            @staticmethod
            def add_matmul_rhs_constant(*args, **kwargs):
                calls.append((args, kwargs))
                return "family-owned-matmul"

        ctx = QuantContext(
            profile=QuantProfile(
                format=get_format("fp8"),
                scale_map=QuantScaleMap(),
            ),
            graph_ops=FakeGraphOps,
        )

        result = ctx.maybe_quantized_matmul(
            object(),
            object(),
            lhs_width=2,
            rhs_width=3,
            rhs_weights=np.zeros((2, 3), dtype=np.float32),
            weight_name="layer.0.w_q",
        )

        assert result == "family-owned-matmul"
        assert calls

    def test_context_rejects_matmul_without_family_graph_ops(self):
        from tensorrt_model_connect.quantization.context import QuantContext

        ctx = QuantContext(
            profile=QuantProfile(
                format=get_format("fp8"),
                scale_map=QuantScaleMap(),
            ),
        )

        with pytest.raises(RuntimeError, match="family graph_ops"):
            ctx.maybe_quantized_matmul(
                object(),
                object(),
                lhs_width=2,
                rhs_width=3,
                rhs_weights=np.zeros((2, 3), dtype=np.float32),
                weight_name="layer.0.w_q",
            )


class TestPreQuantizedCheckpointProvider:
    def test_detect_awq_format(self, tmp_path):
        """AWQ path reached (not NotImplementedError)."""
        from tensorrt_model_connect.quantization.scale_providers import PreQuantizedCheckpointProvider
        from tensorrt_model_connect.config import ModelConfig

        config = ModelConfig.from_json(json.dumps({
            "model_type": "prequantized_decoder",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            }
        }))
        provider = PreQuantizedCheckpointProvider()
        # Should reach AWQ extraction (not NotImplementedError)
        # Will return empty scales since tmp_path has no safetensors,
        # but must NOT raise NotImplementedError
        result = provider.acquire_scales(str(tmp_path), config, get_format("int4_awq"), [])
        assert isinstance(result, QuantScaleMap)
        assert len(result.scales) == 0  # no safetensors files present


class _FakeAdapter:
    def map_layer_name(self, layer_name: str) -> str | None:
        if layer_name == "skip.this":
            return None
        return f"unit_family/{layer_name}"


class TestModelOptScaleMapping:
    def test_family_adapter_maps_layer_names(self):
        from tensorrt_model_connect.quantization.scale_providers import ModelOptCalibrationProvider

        provider = ModelOptCalibrationProvider()
        state_dict = {
            "layer.0.w_q.input_quantizer._amax": np.array(44.8, dtype=np.float32),
            "layer.0.w_q.weight_quantizer._amax": np.array(22.4, dtype=np.float32),
            "skip.this.input_quantizer._amax": np.array(1.0, dtype=np.float32),
            "skip.this.weight_quantizer._amax": np.array(1.0, dtype=np.float32),
        }

        scale_map = provider._build_scale_map(
            state_dict,
            adapter=_FakeAdapter(),
            exclude_re=None,
            exclude_patterns=[],
            maxbound=448.0,
        )

        assert "unit_family/layer.0.w_q" in scale_map.scales
        assert "unit_family/skip.this" not in scale_map.scales
        assert abs(scale_map.scales["unit_family/layer.0.w_q"].input_scale - 0.1) < 1e-6

    def test_family_adapter_exclude_patterns_apply_after_mapping(self):
        from tensorrt_model_connect.quantization.scale_providers import ModelOptCalibrationProvider

        provider = ModelOptCalibrationProvider()
        state_dict = {
            "model.layers.0.self_attn.q_proj.input_quantizer._amax": np.array(44.8, dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight_quantizer._amax": np.array(22.4, dtype=np.float32),
            "model.layers.0.self_attn.o_proj.input_quantizer._amax": np.array(44.8, dtype=np.float32),
            "model.layers.0.self_attn.o_proj.weight_quantizer._amax": np.array(22.4, dtype=np.float32),
        }

        from tensorrt_model_connect.quantization.adapters import StandardDecoderCalibrationAdapter
        scale_map = provider._build_scale_map(
            state_dict,
            adapter=StandardDecoderCalibrationAdapter(family="unit_family"),
            exclude_re=None,
            exclude_patterns=["layer.*.w_o"],
            maxbound=448.0,
        )

        assert "unit_family/layer.0.w_q" in scale_map.scales
        assert "unit_family/layer.0.w_o" not in scale_map.scales

    def test_standard_decoder_adapter_maps_standard_decoder_names(self):
        from tensorrt_model_connect.quantization.adapters import StandardDecoderCalibrationAdapter

        adapter = StandardDecoderCalibrationAdapter(family="unit_family")

        assert adapter.map_layer_name("model.layers.0.self_attn.q_proj") == "unit_family/layer.0.w_q"
        assert adapter.map_layer_name("model.layers.12.self_attn.o_proj") == "unit_family/layer.12.w_o"
        assert adapter.map_layer_name("model.layers.7.mlp.gate_proj") == "unit_family/layer.7.w_gate"
        assert adapter.map_layer_name("model.layers.7.mlp.up_proj") == "unit_family/layer.7.w_up"
        assert adapter.map_layer_name("model.layers.7.mlp.down_proj") == "unit_family/layer.7.w_down"
        assert adapter.map_layer_name("model.layers.0.input_layernorm") is None


class TestQuantProfileExclusions:
    def test_family_scoped_weight_name_matches_builder_local_exclude_pattern(self):
        profile = QuantProfile(
            format=get_format("fp8"),
            scale_map=QuantScaleMap(scales={
                "unit_family/layer.0.w_o": LayerScales(input_scale=0.25, weight_scale=0.5),
            }),
            exclude_patterns=["layer.*.w_o"],
        )

        assert profile.should_quantize("unit_family/layer.0.w_o") is False
