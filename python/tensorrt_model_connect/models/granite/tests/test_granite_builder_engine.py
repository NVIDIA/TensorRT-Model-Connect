# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Granite family plugin.

Trace: ARCH-FAM-001, UD-FAM-GRANITE-01
Intent: Validate the Granite family plugin weight loading and standard decoder key mapping.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
from tensorrt_model_connect.models.granite.tests._family_plugin_tester import (
    FamilyPluginTester,
)
from tensorrt_model_connect.models.granite.tests._family_plugin_test_mixin import (
    FamilyPluginTestMixin,
)


class GranitePluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.models.granite.model"
    model_type = "granite"


class TestGraniteEngine(FamilyPluginTestMixin):
    tester_class = GranitePluginTester
