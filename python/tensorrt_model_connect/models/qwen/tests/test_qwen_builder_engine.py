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


class QwenPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.models.qwen.model"
    # Keep the shared mixin on Qwen2's legacy dense-mask graph. Native Qwen3
    # has a stricter BF16, head-dimension, and split-engine contract below.
    model_type = "qwen2"


class TestQwenEngine(FamilyPluginTestMixin):
    tester_class = QwenPluginTester


class Qwen3NativePluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.models.qwen.model"
    model_type = "qwen3"
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=16,
        max_cache_length=16,
    )

    def get_config_dict(self):
        config = super().get_config_dict()
        config.update(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_act": "silu",
                "head_dim": self.spec.head_dim,
            }
        )
        return config

    def make_hf_tensors(self):
        tensors = super().make_hf_tensors()
        for layer_idx in range(self.spec.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            tensors[f"{prefix}.q_norm.weight"] = np.ones(self.spec.head_dim, dtype=np.float32)
            tensors[f"{prefix}.k_norm.weight"] = np.ones(self.spec.head_dim, dtype=np.float32)
        return tensors


@pytest.mark.parametrize(
    ("role", "profile_shapes"),
    [
        ("prefill", [(1,), (16,), (16,)]),
        ("decode", [(1,), (1,), (1,)]),
    ],
)
@pytest.mark.trt
@pytest.mark.gpu
def test_native_qwen3_split_role_engine_contract(tmp_path, role, profile_shapes):
    import tensorrt as trt

    tester = Qwen3NativePluginTester()
    config, weights, _ = tester.prepare_config_and_weights(tmp_path)
    config.raw["_decoder_engine_role"] = role
    plan = tester.get_plugin().build_engine(
        config,
        weights,
        tester.spec.max_cache_length,
        precision="bf16",
        verbose=False,
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)

    assert engine is not None
    assert engine.num_optimization_profiles == 1
    assert [
        tuple(shape) for shape in engine.get_tensor_profile_shape("token_id", 0)
    ] == profile_shapes

    inputs = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    }
    outputs = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
    }
    assert inputs == {
        "token_id",
        "position_id",
        "cache_write_indices",
        "key_value_lengths",
        "cache_k_0",
        "cache_v_0",
    }
    assert outputs == {"logits", "present_k_0", "present_v_0"}
    assert tuple(engine.get_tensor_shape("logits")) == (
        1,
        tester.spec.vocab_size,
    )

    expected_cache_shape = (
        1,
        tester.spec.num_key_value_heads,
        tester.spec.max_cache_length,
        tester.spec.head_dim,
    )
    for stem in ("k", "v"):
        cache_name = f"cache_{stem}_0"
        present_name = f"present_{stem}_0"
        assert tuple(engine.get_tensor_shape(cache_name)) == expected_cache_shape
        assert tuple(engine.get_tensor_shape(present_name)) == expected_cache_shape
        assert engine.get_tensor_dtype(cache_name) == trt.bfloat16
        assert engine.get_aliased_input_tensor(present_name) == cache_name


def _qwen_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.models.qwen.dual_profile_decoder_tp_builder",
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


def test_qwen_model_advertises_fp8_parallel_quantization_only():
    from tensorrt_model_connect.models.qwen import model

    assert model.supports_parallel_quantization("fp8")
    assert not model.supports_parallel_quantization("int8_sq")
    assert not model.supports_parallel_quantization(None)
