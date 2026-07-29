# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Granite family native-KV plugin.

Trace: ARCH-FAM-001, UD-FAM-GRANITE-01
Intent: Validate the Granite family plugin weight loading and standard decoder key mapping.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""

from __future__ import annotations

import pytest

from tests.builder.family_plugin_tester import FamilyPluginTester, TinyModelSpec
from tests.builder.family_plugin_test_mixin import (
    FamilyPluginTestMixin,
    requires_trt,
)


class GranitePluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.granite"
    model_type = "granite"
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=16,
        max_cache_length=16,
    )

    def get_config_dict(self) -> dict:
        config = super().get_config_dict()
        config.update(
            architectures=["GraniteForCausalLM"],
            attention_bias=False,
            hidden_act="silu",
            mlp_bias=False,
        )
        return config

    def expected_engine_input_names(self) -> set[str]:
        names = super().expected_engine_input_names()
        names.remove("attention_mask")
        names.update({"cache_write_indices", "key_value_lengths"})
        return names


def _deserialize(plan: bytes):
    import tensorrt as trt

    return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)


def _io_names(engine) -> tuple[set[str], set[str]]:
    import tensorrt as trt

    inputs: set[str] = set()
    outputs: set[str] = set()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        target = inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
        target.add(name)
    return inputs, outputs


class TestGraniteEngine(FamilyPluginTestMixin):
    tester_class = GranitePluginTester

    @staticmethod
    def _build_native_engine(tester, tmp_path, role: str = "decode") -> bytes:
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        config.raw["_decoder_engine_role"] = role
        return tester.get_plugin().build_engine(
            config,
            weights,
            tester.spec.max_cache_length,
            precision="fp16",
            verbose=False,
        )

    @requires_trt
    def test_build_engine_succeeds(self, tester, tmp_path):
        plan = self._build_native_engine(tester, tmp_path)
        assert isinstance(plan, bytes)
        assert plan

    @requires_trt
    def test_engine_io_tensor_names(self, tester, tmp_path):
        engine = _deserialize(self._build_native_engine(tester, tmp_path))
        assert engine is not None
        inputs, outputs = _io_names(engine)
        assert inputs == tester.expected_engine_input_names()
        assert outputs == tester.expected_engine_output_names()

    @requires_trt
    def test_engine_logits_output_shape(self, tester, tmp_path):
        import tensorrt as trt

        engine = _deserialize(self._build_native_engine(tester, tmp_path))
        assert engine is not None
        assert tuple(engine.get_tensor_shape("logits")) == (
            1,
            tester.spec.vocab_size,
        )
        assert engine.get_tensor_dtype("logits") == trt.float32

    @pytest.mark.parametrize("role", ["prefill", "decode"])
    @requires_trt
    def test_native_split_role_cache_alias_contract(
        self,
        tester,
        tmp_path,
        role,
    ):
        import tensorrt as trt

        engine = _deserialize(self._build_native_engine(tester, tmp_path, role))
        assert engine is not None
        assert engine.num_optimization_profiles == 1

        cache_shape = (
            1,
            tester.spec.num_key_value_heads,
            tester.spec.max_cache_length,
            tester.spec.head_dim,
        )
        for stem in ("k", "v"):
            cache = f"cache_{stem}_0"
            present = f"present_{stem}_0"
            assert tuple(engine.get_tensor_shape(cache)) == cache_shape
            assert tuple(engine.get_tensor_shape(present)) == cache_shape
            assert engine.get_tensor_dtype(cache) == trt.float16
            assert engine.get_aliased_input_tensor(present) == cache

        profile = engine.get_tensor_profile_shape("token_id", 0)
        assert tuple(profile[0]) == (1,)
        assert tuple(profile[2]) == (tester.spec.max_cache_length if role == "prefill" else 1,)
