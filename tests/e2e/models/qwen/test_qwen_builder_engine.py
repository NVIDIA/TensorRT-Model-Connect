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

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin
from tests.builder.conftest import requires_trt


class QwenPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.qwen"
    model_type = "qwen3"


class TestQwenEngine(FamilyPluginTestMixin):
    tester_class = QwenPluginTester


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


@requires_trt
def test_qwen_dynamic_kv_split_prefill_profile():
    """Qwen split prefill accepts a runtime-sized external KV buffer."""
    import tensorrt as trt

    # Load the public family plugin before importing its implementation modules.
    # The package lazily re-exports plugin symbols and otherwise a direct builder
    # import can observe a partially initialized plugin module.
    from tensorrt_model_connect.families.qwen import plugin as qwen_plugin
    from tensorrt_model_connect.families.qwen.checkpoint_mapper import WeightDict
    from tensorrt_model_connect.families.qwen.config import ModelConfig

    hidden, vocab, num_layers = 16, 32, 2
    num_heads = 4
    max_cache = 4
    rng = np.random.RandomState(7)
    weights = WeightDict()
    weights["embedding"] = rng.randn(vocab, hidden).astype(np.float32)
    for layer in range(num_layers):
        prefix = f"layer.{layer}"
        weights[f"{prefix}.input_norm"] = rng.randn(hidden).astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = rng.randn(hidden).astype(np.float32)
        for name in ("w_q", "w_k", "w_v", "w_o"):
            weights[f"{prefix}.{name}"] = rng.randn(hidden, hidden).astype(np.float32)
        weights[f"{prefix}.w_gate"] = rng.randn(hidden, hidden * 2).astype(np.float32)
        weights[f"{prefix}.w_up"] = rng.randn(hidden, hidden * 2).astype(np.float32)
        weights[f"{prefix}.w_down"] = rng.randn(hidden * 2, hidden).astype(np.float32)
    weights["final_norm"] = rng.randn(hidden).astype(np.float32)
    weights["w_out"] = rng.randn(hidden, vocab).astype(np.float32)
    weights["_attention_size"] = hidden
    weights["_mlp_size"] = hidden * 2

    config = ModelConfig(
        hidden_size=hidden,
        vocab_size=vocab,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    config.raw["dynamic_kv_cache"] = True
    config.raw["_dynamic_kv_profile_rows"] = [2, 4]
    config.raw["_decoder_engine_role"] = "prefill"

    plan = qwen_plugin.build_engine(config, weights, max_cache)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)

    assert engine is not None
    assert engine.num_optimization_profiles == 1
    assert tuple(engine.get_tensor_shape("token_id")) == (-1,)
    assert tuple(engine.get_tensor_shape("cache_k_0")) == (-1, hidden)
    assert engine.get_tensor_profile_shape("cache_k_0", 0) == [
        (1, hidden),
        (1, hidden),
        (max_cache, hidden),
    ]
    assert engine.get_tensor_profile_shape("attention_mask", 0) == [
        (1, 2),
        (max_cache, max_cache + 1),
        (max_cache, max_cache * 2),
    ]
