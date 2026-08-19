# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest


RUNNER_MODULES = (
    "tensorrt_model_connect.models.qwen_vl.vl_debug_runner",
    "tensorrt_model_connect.models.qwen_vl.tests.e2e_plugins.runners.vl_debug_runner",
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


def test_owner_debug_runner_exposes_complete_diff_vl_surface() -> None:
    module = importlib.import_module(RUNNER_MODULES[0])

    required = (
        "TrtRunner",
        "VLTrtRunner",
        "VisionTrtRunner",
        "load_config_from_bundle",
        "load_engine_from_bundle",
        "load_preprocessor_config_from_bundle",
        "load_section_from_bundle",
        "load_vision_engine_from_bundle",
        "preprocess_image_inputs_for_trt",
    )
    assert all(callable(getattr(module, name, None)) for name in required)
    assert callable(module.VisionTrtRunner)
    assert callable(module.VisionTrtRunner.encode)
    assert callable(module.VLTrtRunner.encode_image)
    assert callable(module.VLTrtRunner.generate_vl)


def test_owner_preprocess_returns_named_qwen_merge_group_input(monkeypatch) -> None:
    module = importlib.import_module(RUNNER_MODULES[0])
    pixel_values = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    captured = {}

    def fake_merge_group(image_path, **kwargs):
        captured["image_path"] = image_path
        captured["kwargs"] = kwargs
        return pixel_values

    monkeypatch.setattr(module, "_preprocess_merge_group_chw", fake_merge_group)

    inputs = module.preprocess_image_inputs_for_trt(
        "image.png",
        fixed_image_size=2,
        temporal_patch_size=2,
        patch_size=1,
        merge_size=2,
    )

    assert set(inputs) == {"pixel_values"}
    assert inputs["pixel_values"] is pixel_values
    assert captured == {
        "image_path": "image.png",
        "kwargs": {
            "fixed_image_size": 2,
            "temporal_patch_size": 2,
            "patch_size": 1,
            "merge_size": 2,
        },
    }


def test_owner_vl_runner_substitutes_image_features_and_stops_on_eos() -> None:
    module = importlib.import_module(RUNNER_MODULES[0])

    class FakeTextRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict]] = []

        def step(self, token_id, **kwargs):
            self.calls.append((token_id, kwargs))
            logits = np.zeros(8, dtype=np.float32)
            logits[5 if len(self.calls) == 2 else 7] = 1.0
            return {"logits": logits}

    runner = module.VLTrtRunner.__new__(module.VLTrtRunner)
    runner.image_token_id = 99
    runner.config = {"eos_token_id": 7}
    runner.text_runner = FakeTextRunner()
    image_features = np.arange(8, dtype=np.float32).reshape(2, 4)

    output_ids = runner.generate_vl([1, 99], image_features, max_new_tokens=4)

    assert output_ids == [1, 99, 5, 7]
    assert runner.text_runner.calls[0] == (
        1,
        {"input_embed": None, "use_input_embed": 0.0},
    )
    image_call = runner.text_runner.calls[1]
    assert image_call[0] == 99
    assert image_call[1]["use_input_embed"] == 1.0
    np.testing.assert_array_equal(
        image_call[1]["input_embed"], image_features[:1]
    )
    assert runner.text_runner.calls[2] == (5, {})
