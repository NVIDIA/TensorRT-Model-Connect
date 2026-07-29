# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the native-KV Mistral family plugin.

Trace: ARCH-FAM-001, UD-FAM-MISTRAL-01
Intent: Validate dense Mistral weight mapping and the split native-KV engine I/O contract.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All native decoder weights are present and the engine exposes user-owned KV cache aliases.
"""
import importlib

import pytest

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin
from tests.builder.family_plugin_tester import TinyModelSpec


class MistralPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.mistral"
    model_type = "mistral"
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
                "architectures": ["MistralForCausalLM"],
                "hidden_act": "silu",
                "sliding_window": None,
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


class TestMistralEngine(FamilyPluginTestMixin):
    tester_class = MistralPluginTester
