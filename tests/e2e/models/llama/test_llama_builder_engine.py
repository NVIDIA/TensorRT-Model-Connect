# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the LLaMA family plugin.

Trace: ARCH-FAM-001, UD-FAM-LLAMA-01
Intent: Validate the LLaMA family plugin weight loading and standard decoder key mapping with RMSNorm and SwiGLU MLP.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class LlamaPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.llama"
    model_type = "llama"


class TestLlamaEngine(FamilyPluginTestMixin):
    tester_class = LlamaPluginTester
