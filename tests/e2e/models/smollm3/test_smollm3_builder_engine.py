# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the SmolLM3 family plugin.

Trace: ARCH-FAM-001, UD-FAM-SMOLLM3-01
Intent: Validate the SmolLM3 family plugin weight loading and standard decoder key mapping with RMSNorm and SwiGLU MLP.
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


class SmolLM3PluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.smollm3"
    model_type = "smollm3"


class NativeSmolLM3PluginTester(SmolLM3PluginTester):
    """Smallest production-shaped dense SmolLM3 accepted by native attention."""

    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=128,
        max_cache_length=128,
    )

    def get_config_dict(self) -> dict:
        config = super().get_config_dict()
        config.update(
            architectures=["SmolLM3ForCausalLM"],
            hidden_act="silu",
        )
        return config


def _deserialize(plan: bytes):
    import tensorrt as trt

    return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)


def _io_names(engine) -> tuple[set[str], set[str]]:
    import tensorrt as trt

    inputs: set[str] = set()
    outputs: set[str] = set()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        target = (
            inputs
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else outputs
        )
        target.add(name)
    return inputs, outputs


class TestSmolLM3Engine(FamilyPluginTestMixin):
    tester_class = SmolLM3PluginTester

    @staticmethod
    def _build_legacy_engine(tester, tmp_path) -> bytes:
        from tensorrt_model_connect.families.smollm3.standard_decoder_builder import (
            build_standard_decoder_engine,
        )

        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        return build_standard_decoder_engine(
            config,
            weights,
            tester.spec.max_cache_length,
            precision="fp32",
            verbose=False,
        )

    @requires_trt
    def test_build_engine_succeeds(self, tester, tmp_path):
        """Keep the generic dense-mask SmolLM3 builder smoke covered."""
        plan = self._build_legacy_engine(tester, tmp_path)
        assert isinstance(plan, bytes)
        assert plan

    @requires_trt
    def test_engine_io_tensor_names(self, tester, tmp_path):
        """Keep the legacy Python-builder/C++ tensor naming contract covered."""
        engine = _deserialize(self._build_legacy_engine(tester, tmp_path))
        assert engine is not None
        inputs, outputs = _io_names(engine)
        assert inputs == tester.expected_engine_input_names()
        assert outputs == tester.expected_engine_output_names()

    @requires_trt
    def test_engine_logits_output_shape(self, tester, tmp_path):
        """Keep the legacy single-row logits contract covered."""
        engine = _deserialize(self._build_legacy_engine(tester, tmp_path))
        assert engine is not None
        assert tuple(engine.get_tensor_shape("logits")) == (
            1,
            tester.spec.vocab_size,
        )

    @pytest.mark.parametrize("role", ["prefill", "decode"])
    @requires_trt
    def test_native_split_role_engine_contract(self, tmp_path, role):
        """Qualified BF16 split engines expose full-capacity aliased KV state."""
        import tensorrt as trt

        tester = NativeSmolLM3PluginTester()
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        config.raw["_decoder_engine_role"] = role

        plan = tester.get_plugin().build_engine(
            config,
            weights,
            tester.spec.max_cache_length,
            precision="bf16",
            verbose=False,
        )
        engine = _deserialize(plan)
        assert engine is not None
        assert engine.num_optimization_profiles == 1

        inputs, outputs = _io_names(engine)
        assert "attention_mask" not in inputs
        assert {"cache_write_indices", "key_value_lengths"} <= inputs
        assert {"cache_k_0", "cache_v_0"} <= inputs
        assert {"present_k_0", "present_v_0"} <= outputs

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
            assert engine.get_tensor_dtype(cache) == trt.bfloat16
            assert engine.get_aliased_input_tensor(present) == cache

        profile = engine.get_tensor_profile_shape("token_id", 0)
        assert tuple(profile[0]) == (1,)
        assert tuple(profile[2]) == (
            min(tester.spec.max_cache_length, 64) if role == "prefill" else 1,
        )

    @requires_trt
    def test_native_decode_records_explicit_attention_graph(self, tmp_path):
        """Native decode uses primitives and does not expose the old KVL Recipe."""
        from tensorrt_model_connect.tvm_ffi import graph_build
        from tensorrt_model_connect.tvm_ffi.graph_patch import load_snapshot

        tester = NativeSmolLM3PluginTester()
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        config.raw["_decoder_engine_role"] = "decode"
        snapshot_path = tmp_path / "smollm3-decode.graph.json"

        with pytest.raises(graph_build.GraphInspectionComplete):
            with graph_build.inspect_graph(
                snapshot_path,
                engine_role="decode",
                metadata={},
            ):
                with graph_build.engine_role("decode"):
                    tester.get_plugin().build_engine(
                        config,
                        weights,
                        tester.spec.max_cache_length,
                        precision="bf16",
                        verbose=False,
                    )

        snapshot = load_snapshot(snapshot_path)
        assert snapshot.metadata.get("graph_recipes", []) == []
        operations = [node.op for node in snapshot.nodes]
        assert sum("MATRIX_MULTIPLY" in operation for operation in operations) >= 2
        assert any("SOFTMAX" in operation for operation in operations)
        assert not any(operation.endswith("ATTENTION") for operation in operations)
