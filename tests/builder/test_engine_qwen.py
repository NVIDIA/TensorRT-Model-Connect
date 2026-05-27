"""Engine tests for the Qwen family plugin.

Trace: ARCH-FAM-001, UD-FAM-QWEN-01
Intent: Validate the Qwen family plugin weight loading and standard decoder key mapping with RMSNorm, SwiGLU MLP, and RoPE.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
import pytest

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


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
