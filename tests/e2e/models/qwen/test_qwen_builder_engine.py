# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Qwen family plugin.

Trace: ARCH-FAM-001, UD-FAM-QWEN-01
Intent: Validate the Qwen family plugin weight loading and standard decoder key mapping with RMSNorm, SwiGLU MLP, and RoPE.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="Qwen builder tests require TensorRT")

from tests.builder.family_plugin_tester import FamilyPluginTester, TinyModelSpec
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


def test_qwen3_rope_table_matches_hf_fp32_through_full_capacity():
    from tensorrt_model_connect.families.qwen import graph_ops

    head_dim = 128
    rope_theta = 1_000_000.0
    capacity = 40960
    positions = np.asarray(
        [0, 1, 2048, 8192, 32768, capacity - 1],
        dtype=np.int64,
    )
    dims = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = np.float32(1.0) / np.power(
        np.float32(rope_theta),
        dims / np.float32(head_dim),
    )
    reference_angles = (
        positions.astype(np.float32)[:, None] * inv_freq[None, :]
    )

    actual_cos = graph_ops.make_rope_table_half_dim(
        capacity, head_dim, rope_theta, True)
    actual_sin = graph_ops.make_rope_table_half_dim(
        capacity, head_dim, rope_theta, False)
    np.testing.assert_allclose(
        actual_cos[positions], np.cos(reference_angles), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_sin[positions], np.sin(reference_angles), rtol=1e-6, atol=1e-6)


class QwenPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.qwen"
    model_type = "qwen3"
    # Native attention is intentionally non-decomposable. Use a production
    # fused-MHA head geometry instead of the shared toy D=4 fixture.
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        max_cache_length=16,
    )

class TestQwenEngine(FamilyPluginTestMixin):
    tester_class = QwenPluginTester

    @staticmethod
    def _assert_native_cache_contract(engine, tester):
        import tensorrt as trt

        inputs = {
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        }
        assert "attention_mask" not in inputs
        assert {"cache_write_indices", "key_value_lengths"} <= inputs

        expected_cache_shape = (
            1,
            tester.spec.num_key_value_heads,
            tester.spec.max_cache_length,
            tester.spec.head_dim,
        )
        for layer_idx in range(tester.spec.num_hidden_layers):
            for stem in ("k", "v"):
                cache_name = f"cache_{stem}_{layer_idx}"
                present_name = f"present_{stem}_{layer_idx}"
                assert tuple(engine.get_tensor_shape(cache_name)) == expected_cache_shape
                assert tuple(engine.get_tensor_shape(present_name)) == expected_cache_shape
                assert engine.get_aliased_input_tensor(present_name) == cache_name

    @pytest.mark.trt
    @pytest.mark.gpu
    def test_native_kv_cache_shapes_and_aliases(self, tester, tmp_path):
        import tensorrt as trt
        from tensorrt_model_connect.families.qwen.standard_decoder_builder import (
            build_standard_decoder_engine,
        )

        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        plan = build_standard_decoder_engine(
            config,
            weights,
            tester.spec.max_cache_length,
            precision="bf16",
            verbose=False,
            native_kv_cache=True,
        )
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
        assert engine is not None
        self._assert_native_cache_contract(engine, tester)

    @pytest.mark.parametrize("role", ["prefill", "decode"])
    @pytest.mark.trt
    @pytest.mark.gpu
    def test_split_engines_share_native_cache_contract(
        self, tester, tmp_path, role,
    ):
        import tensorrt as trt
        from tensorrt_model_connect.families.qwen.standard_decoder_builder import (
            build_standard_decoder_engine,
        )

        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        config.raw["_decoder_engine_role"] = role
        plan = build_standard_decoder_engine(
            config,
            weights,
            tester.spec.max_cache_length,
            precision="bf16",
            verbose=False,
            native_kv_cache=True,
        )
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
        assert engine is not None
        assert engine.num_optimization_profiles == 1
        self._assert_native_cache_contract(engine, tester)


@pytest.mark.unit
def test_native_kv_cache_is_qwen3_owned(monkeypatch):
    import importlib

    from tensorrt_model_connect.families.qwen.config import ModelConfig

    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen.plugin")
    native_modes = []

    def fake_build(*_args, native_kv_cache=False, **_kwargs):
        native_modes.append(native_kv_cache)
        return b"plan"

    monkeypatch.setattr(
        plugin_module, "build_standard_decoder_engine", fake_build)
    for model_type in ("qwen2", "qwen3"):
        config = ModelConfig.create_tiny(model_type)
        assert plugin_module.plugin.build_engine(
            config, {}, max_cache_length=16, precision="bf16") == b"plan"

    prototype = ModelConfig(
        model_type="qwen3",
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        _head_dim=128,
        max_position_embeddings=40960,
    )
    assert plugin_module.plugin.build_engine(
        prototype, {}, max_cache_length=40960, precision="bf16") == b"plan"

    assert native_modes == [False, False, True]


def _qwen_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.families.qwen.dual_profile_decoder_tp_builder",
        reason="TensorRT is required for Qwen TP builder tests",
    )


def _quant_ctx(format_name: str):
    from tensorrt_model_connect.quantization import get_format, QuantScaleMap
    from tensorrt_model_connect.quantization.context import QuantContext
    from tensorrt_model_connect.quantization.profile import QuantProfile

    return QuantContext(
        profile=QuantProfile(
            format=get_format(format_name),
            scale_map=QuantScaleMap(dynamic=True),
        )
    )


def test_qwen_tp_quantization_allows_fp8():
    qwen_tp_builder = _qwen_tp_builder_module()

    qwen_tp_builder._validate_tp_quantization(_quant_ctx("fp8"))


def test_qwen_tp_quantization_rejects_non_fp8():
    qwen_tp_builder = _qwen_tp_builder_module()

    with pytest.raises(ValueError, match="supports fp8 only"):
        qwen_tp_builder._validate_tp_quantization(_quant_ctx("int8_sq"))


def test_qwen_plugin_advertises_fp8_parallel_quantization_only():
    from tensorrt_model_connect.families.qwen import plugin

    assert plugin.supports_parallel_quantization("fp8")
    assert not plugin.supports_parallel_quantization("int8_sq")
    assert not plugin.supports_parallel_quantization(None)


def test_engine_builder_parallel_quantization_gate_allows_qwen_fp8():
    from tensorrt_model_connect.engine_builder import (
        _plugin_supports_parallel_quantization,
    )
    from tensorrt_model_connect.families.qwen import plugin

    assert _plugin_supports_parallel_quantization(plugin, _quant_ctx("fp8"))


def test_engine_builder_parallel_quantization_gate_rejects_default_plugin():
    from tensorrt_model_connect.engine_builder import (
        _plugin_supports_parallel_quantization,
    )

    class PluginWithoutParallelQuantization:
        pass

    assert not _plugin_supports_parallel_quantization(
        PluginWithoutParallelQuantization(),
        _quant_ctx("fp8"),
    )
