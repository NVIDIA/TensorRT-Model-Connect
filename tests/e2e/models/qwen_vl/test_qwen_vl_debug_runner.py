# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect

import pytest


RUNNER_MODULES = (
    "tensorrt_model_connect.families.qwen_vl.vl_debug_runner",
    "tests.e2e.models.qwen_vl.e2e_plugins.runners.vl_debug_runner",
)


class FakeEngine:
    def __init__(self, declared, profile_shapes=()):
        self.declared = declared
        self.profile_shapes = profile_shapes
        self.profile_requests: list[tuple[str, int]] = []

    def get_tensor_shape(self, name):
        assert name == "input_embed"
        return self.declared

    def get_tensor_profile_shape(self, name, profile_index):
        self.profile_requests.append((name, profile_index))
        return self.profile_shapes


@pytest.mark.parametrize("module_name", RUNNER_MODULES)
def test_profile_min_shape_resolves_dynamic_decode_input(module_name) -> None:
    module = importlib.import_module(module_name)
    engine = FakeEngine(
        (-1, 2048),
        ((1, 2048), (1, 2048), (1, 2048)),
    )

    assert module._profile_min_shape(engine, "input_embed", 1) == (1, 2048)
    assert engine.profile_requests == [("input_embed", 1)]


@pytest.mark.parametrize("module_name", RUNNER_MODULES)
def test_profile_min_shape_preserves_static_input(module_name) -> None:
    module = importlib.import_module(module_name)
    engine = FakeEngine((1, 2048))

    assert module._profile_min_shape(engine, "input_embed", 0) == (1, 2048)
    assert engine.profile_requests == []


@pytest.mark.parametrize("module_name", RUNNER_MODULES)
def test_profile_min_shape_rejects_unresolved_profile(module_name) -> None:
    module = importlib.import_module(module_name)
    engine = FakeEngine(
        (-1, 2048),
        ((-1, 2048), (-1, 2048), (-1, 2048)),
    )

    with pytest.raises(RuntimeError, match="Invalid optimization profile shape"):
        module._profile_min_shape(engine, "input_embed", 0)


@pytest.mark.parametrize("module_name", RUNNER_MODULES)
def test_debug_runner_binds_mrope_and_fails_closed(module_name) -> None:
    module = importlib.import_module(module_name)
    init_source = inspect.getsource(module.TrtRunner.__init__)
    step_source = inspect.getsource(module.TrtRunner.step)

    assert '"mrope_position_ids"' in init_source
    assert "self._h_mrope_position_ids.fill(position_id)" in step_source
    assert '"mrope_position_ids", self._d_mrope_position_ids' in step_source
    assert "if not self.context.execute_async_v3(stream):" in step_source
    assert "TensorRT Qwen-VL debug decoder execution failed" in step_source
