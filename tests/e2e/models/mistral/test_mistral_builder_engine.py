# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Mistral family plugin.

Trace: ARCH-FAM-001, UD-FAM-MISTRAL-01
Intent: Validate the Mistral family plugin weight loading and standard decoder key mapping with RMSNorm, SwiGLU MLP, and sliding window attention config.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class MistralPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.mistral"
    model_type = "mistral"


class TestMistralEngine(FamilyPluginTestMixin):
    tester_class = MistralPluginTester
