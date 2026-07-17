# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.sam3 import (
    native_plugin_builder,
    tracker_memory_ffi_builder,
)


def _global(policy: str, batch_size: int, digit: str) -> str:
    return f"trtmc.sam3.tracker_memory.{policy}.b{batch_size}.fixed.{digit * 20}"


@pytest.mark.parametrize(
    ("policy", "batch_size", "hard_mask"),
    [("soft", 1, False), ("soft", 2, False), ("hard", 1, True), ("hard", 2, True)],
)
def test_memory_globals_bind_policy_batch_and_content_digest(
    policy: str,
    batch_size: int,
    hard_mask: bool,
) -> None:
    tracker_memory_ffi_builder._validate_global_name(
        _global(policy, batch_size, "a"),
        batch_size=batch_size,
        hard_mask=hard_mask,
    )

    with pytest.raises(ValueError, match=f"{policy} B{batch_size} fixed AOTI"):
        tracker_memory_ffi_builder._validate_global_name(
            _global("hard" if policy == "soft" else "soft", batch_size, "a"),
            batch_size=batch_size,
            hard_mask=hard_mask,
        )
    with pytest.raises(ValueError, match=f"{policy} B{batch_size} fixed AOTI"):
        tracker_memory_ffi_builder._validate_global_name(
            _global(policy, 2 if batch_size == 1 else 1, "a"),
            batch_size=batch_size,
            hard_mask=hard_mask,
        )


def test_memory_plugin_uses_tensorrt_11_creator_api() -> None:
    calls: list[tuple[str, str, str]] = []
    creator = object()

    class _Registry:
        def get_creator(self, plugin_type: str, version: str, namespace: str):
            calls.append((plugin_type, version, namespace))
            return creator

    trt = SimpleNamespace(get_plugin_registry=lambda: _Registry())
    assert tracker_memory_ffi_builder._plugin_creator(trt) is creator
    assert calls == [("Sam3TrackerMemoryFfi", "2", "")]


def test_memory_plans_load_dso_then_build_canonical_soft_hard_b1_b2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_path = Path("/tmp/libtrtmc_sam3_tracker_step_native_plugin.so")
    globals_by_variant = {
        ("soft", 1): _global("soft", 1, "1"),
        ("soft", 2): _global("soft", 2, "2"),
        ("hard", 1): _global("hard", 1, "3"),
        ("hard", 2): _global("hard", 2, "4"),
    }
    spec = tracker_memory_ffi_builder.TrackerMemoryPlanSpec(
        plugin_library=plugin_path,
        soft_global_b1=globals_by_variant[("soft", 1)],
        soft_global_b2=globals_by_variant[("soft", 2)],
        hard_global_b1=globals_by_variant[("hard", 1)],
        hard_global_b2=globals_by_variant[("hard", 2)],
    )
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        native_plugin_builder,
        "load_native_plugin",
        lambda path: events.append(("load", path)),
    )

    def fake_build(
        global_name: str,
        *,
        batch_size: int,
        hard_mask: bool,
        verbose: bool,
    ) -> bytes:
        events.append(("build", global_name, batch_size, hard_mask, verbose))
        return f"{'hard' if hard_mask else 'soft'}-b{batch_size}".encode()

    monkeypatch.setattr(
        tracker_memory_ffi_builder,
        "_build_tracker_memory_ffi_plan",
        fake_build,
    )
    plans = tracker_memory_ffi_builder.build_sam3_tracker_memory_ffi_plans(
        spec,
        verbose=True,
    )

    assert events == [
        ("load", plugin_path),
        ("build", globals_by_variant[("soft", 1)], 1, False, True),
        ("build", globals_by_variant[("soft", 2)], 2, False, True),
        ("build", globals_by_variant[("hard", 1)], 1, True, True),
        ("build", globals_by_variant[("hard", 2)], 2, True, True),
    ]
    assert plans == {
        "sam3_tracker_memory_engine_plan": b"soft-b1",
        "sam3_tracker_memory_batch2_engine_plan": b"soft-b2",
        "sam3_tracker_hard_memory_engine_plan": b"hard-b1",
        "sam3_tracker_hard_memory_batch2_engine_plan": b"hard-b2",
    }


def test_memory_wrapper_preserves_runtime_io_and_fixed_plugin_contract() -> None:
    build_source = inspect.getsource(tracker_memory_ffi_builder._build_tracker_memory_ffi_plan)
    for name in (
        "tracker_feature_2",
        "final_mask",
        "owned_tracker_mask",
        "object_score_logits",
        "suppress_area_shrinkage",
    ):
        assert f'"{name}"' in build_source
    assert "(1, 256, 72, 72)" in build_source
    assert "mask_size = 1008 if hard_mask else 288" in build_source
    assert "(batch_size, 1, mask_size, mask_size)" in build_source
    assert "(batch_size, 1)" in build_source
    assert "_constant_zero_suppression" in build_source
    assert "[feature, mask, score, suppression]" in build_source
    assert "network.add_plugin_v2" in build_source

    output_source = inspect.getsource(tracker_memory_ffi_builder._add_output_contract)
    assert '"new_memory_features"' in output_source
    assert '"new_memory_position"' in output_source
    slice_source = inspect.getsource(tracker_memory_ffi_builder._slice_plane)
    assert "(1, _SPATIAL_TOKENS, 1, _MEMORY_CHANNELS)" in slice_source
    assert "(_SPATIAL_TOKENS, 1, _MEMORY_CHANNELS)" in slice_source
    assert "(1, 2, _SPATIAL_TOKENS, _MEMORY_CHANNELS)" in slice_source
    assert "(2, _SPATIAL_TOKENS, _MEMORY_CHANNELS)" in slice_source


def test_memory_native_bridge_is_stream_aware_and_generation_safe() -> None:
    source_dir = Path(tracker_memory_ffi_builder.__file__).with_name("native_plugins")
    plugin = (source_dir / "sam3_tracker_memory_ffi_plugin.cpp").read_text(encoding="utf-8")
    bridge = (source_dir / "sam3_tracker_memory_aoti_bridge.cpp").read_text(encoding="utf-8")
    cmake = (source_dir / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "TVMFFIEnvSetStream" in plugin
    assert "FfiStreamScope" in plugin
    assert "return resolve_kernel() ? 0 : -1;" in plugin
    assert "descriptor_shapes_valid" in plugin
    assert "plugin->resolve_kernel()" in plugin
    assert "TVMFFIErrorMoveFromRaised" in plugin
    assert "cudaStreamSynchronize" not in plugin + bridge

    assert "trtmc_sam3_tracker_memory_register_package" in bridge
    assert "TVMFFIFunctionSetGlobal(&name, function, 1)" in bridge
    assert "std::vector<std::unique_ptr<Entry>>& retained_entries()" in bridge
    assert "static auto* value = new std::vector<std::unique_ptr<Entry>>" in bridge
    assert "constexpr std::size_t kAotiRunnerCount = 2" in bridge
    assert 'AotiLoader>(package_path, "model", false, kAotiRunnerCount' in bridge
    assert "loader_for_device" in bridge
    assert "validate_tensor_contract" in bridge
    assert "cudaMemcpyAsync" in bridge
    assert "cudaMemcpyDeviceToDevice" in bridge
    assert "run_mutex" not in bridge
    assert "onnx" not in (plugin + bridge).lower()

    assert "sam3_tracker_memory_aoti_bridge.cpp" in cmake
    assert "sam3_tracker_memory_ffi_plugin.cpp" in cmake
    assert "sam3_tracker_memory_ffi_shape_bridge" in cmake
    assert "sam3_tracker_memory_aoti_registration_generations" in cmake
