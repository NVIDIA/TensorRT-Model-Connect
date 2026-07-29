# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the native-KV InternLM2 family plugin.

InternLM2 uses non-standard HF key names and a group-interleaved fused QKV
projection (attention.wqkv.weight). The tester overrides make_hf_tensors()
to produce the correct synthetic weight layout.

Trace: ARCH-FAM-001, UD-FAM-INTERNLM-01
Intent: Validate the InternLM2 family plugin weight loading including group-interleaved fused QKV splitting and non-standard HF key names (tok_embeddings, attention.wqkv, output.weight).
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Fused QKV is split correctly and the engine exposes the native KV alias contract.
"""
import importlib

import numpy as np
import pytest

from tests.builder.family_plugin_tester import FamilyPluginTester, TinyModelSpec
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class InternLMPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.internlm"
    model_type = "internlm2"
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        max_cache_length=128,
    )

    def get_plugin(self):
        try:
            module = importlib.import_module(f"{self.plugin_module}.plugin")
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"Cannot import {self.plugin_module}: {exc}")
        return module.plugin

    def get_config_dict(self) -> dict:
        config = super().get_config_dict()
        config.update(
            {
                "architectures": ["InternLM2ForCausalLM"],
                "hidden_act": "silu",
                "bias": False,
                "rope_scaling": {"type": "dynamic", "factor": 1.0},
                "_decoder_engine_layout": "split",
                "_decoder_engine_role": "decode",
            }
        )
        return config

    def expected_engine_input_names(self) -> set[str]:
        names = {
            "token_id",
            "position_id",
            "cache_write_indices",
            "key_value_lengths",
        }
        for layer in range(self.spec.num_hidden_layers):
            names.add(f"cache_k_{layer}")
            names.add(f"cache_v_{layer}")
        return names

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic InternLM2 weight layout with fused group-interleaved QKV.

        Intention:
            InternLM2 uses different HF key names than the standard decoder
            (model.tok_embeddings instead of model.embed_tokens, output.weight
            instead of lm_head.weight, etc.) and stores QKV as a single
            group-interleaved tensor (attention.wqkv.weight).

        Setup:
            Build synthetic tensors matching InternLM2's checkpoint layout:
            - model.tok_embeddings.weight [vocab, hidden]
            - model.layers.{i}.attention_norm.weight [hidden]
            - model.layers.{i}.ffn_norm.weight [hidden]
            - model.layers.{i}.attention.wqkv.weight [q_dim + 2*kv_dim, hidden]
              (group-interleaved: per KV group, Q heads then K then V)
            - model.layers.{i}.attention.wo.weight [hidden, hidden]
            - model.layers.{i}.feed_forward.{w1,w3,w2}.weight (gate, up, down)
            - model.norm.weight [hidden]
            - output.weight [vocab, hidden]
        """
        s = self.spec

        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.tok_embeddings.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            t[f"{p}.attention_norm.weight"] = rand(s.hidden_size)
            t[f"{p}.ffn_norm.weight"] = rand(s.hidden_size)

            # Build group-interleaved fused QKV:
            # For each KV group g: [Q_heads_in_group, K_head, V_head]
            group_size = s.num_attention_heads // s.num_key_value_heads
            rows_per_group = group_size * s.head_dim + 2 * s.head_dim
            total_qkv = s.num_key_value_heads * rows_per_group
            wqkv = rand(total_qkv, s.hidden_size)
            t[f"{p}.attention.wqkv.weight"] = wqkv

            t[f"{p}.attention.wo.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.feed_forward.w1.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.feed_forward.w3.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.feed_forward.w2.weight"] = rand(
                s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["output.weight"] = rand(s.vocab_size, s.hidden_size)
        return t


class TestInternLMEngine(FamilyPluginTestMixin):
    tester_class = InternLMPluginTester

    @pytest.mark.parametrize("role", ["prefill", "decode"])
    @pytest.mark.gpu
    @pytest.mark.trt
    def test_native_split_role_builds_and_deserializes(
        self, tester, tmp_path, role,
    ):
        trt = pytest.importorskip("tensorrt")
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        config.raw["_decoder_engine_role"] = role

        plan = tester.get_plugin().build_engine(
            config,
            weights,
            tester.spec.max_position_embeddings,
        )

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        assert engine.num_optimization_profiles == 1
        token_profile = tuple(
            tuple(shape)
            for shape in engine.get_tensor_profile_shape("token_id", 0)
        )
        position_profile = tuple(
            tuple(shape)
            for shape in engine.get_tensor_profile_shape("position_id", 0)
        )
        expected_profile = (
            ((1,), (64,), (tester.spec.max_position_embeddings,))
            if role == "prefill"
            else ((1,), (1,), (1,))
        )
        assert token_profile == expected_profile
        assert position_profile == expected_profile
        assert tuple(engine.get_tensor_shape("cache_k_0")) == (
            1,
            tester.spec.num_key_value_heads,
            tester.spec.max_position_embeddings,
            tester.spec.head_dim,
        )
        assert tuple(engine.get_tensor_shape("present_k_0")) == tuple(
            engine.get_tensor_shape("cache_k_0"))
        assert tuple(engine.get_tensor_shape("logits")) == (
            1, tester.spec.vocab_size)
