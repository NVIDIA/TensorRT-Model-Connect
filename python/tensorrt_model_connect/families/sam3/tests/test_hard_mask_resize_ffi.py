# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tensorrt_model_connect.families.sam3 import (
    hard_mask_resize_aoti_exporter as exporter,
    hard_mask_resize_ffi_builder as builder,
    native_plugin_builder,
)


def _global(batch_size: int, digit: str) -> str:
    return f"trtmc.sam3.tracker_memory.resize.b{batch_size}.fixed.{digit * 20}"


@pytest.mark.parametrize("error", [float("nan"), float("inf"), float("-inf"), -1.0, 2.1e-5])
def test_resize_smoke_error_rejects_non_finite_and_out_of_range(error: float) -> None:
    assert not exporter._valid_smoke_error(error)


@pytest.mark.parametrize("error", [0.0, 1.0e-6, 2.0e-5])
def test_resize_smoke_error_accepts_finite_tolerance(error: float) -> None:
    assert exporter._valid_smoke_error(error)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resize_global_binds_batch_and_content_digest(batch_size: int) -> None:
    builder._validate_global_name(_global(batch_size, "a"), batch_size=batch_size)
    with pytest.raises(ValueError, match=f"B{batch_size} AOTI package"):
        builder._validate_global_name(
            _global(2 if batch_size == 1 else 1, "a"), batch_size=batch_size
        )


def test_resize_plans_load_dso_then_build_b1_b2(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = Path("/tmp/libtrtmc_sam3_tracker_step_native_plugin.so")
    spec = builder.HardMaskResizePlanSpec(
        plugin_library=plugin,
        global_name_b1=_global(1, "1"),
        global_name_b2=_global(2, "2"),
    )
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        native_plugin_builder,
        "load_native_plugin",
        lambda path: events.append(("load", path)),
    )

    def fake_build(global_name: str, *, batch_size: int, verbose: bool) -> bytes:
        events.append(("build", global_name, batch_size, verbose))
        return f"resize-b{batch_size}".encode()

    monkeypatch.setattr(builder, "_build_hard_mask_resize_plan", fake_build)
    plans = builder.build_sam3_hard_mask_resize_ffi_plans(spec, verbose=True)
    assert events == [
        ("load", plugin),
        ("build", spec.global_name_b1, 1, True),
        ("build", spec.global_name_b2, 2, True),
    ]
    assert plans == {
        builder.HARD_MASK_RESIZE_SECTION: b"resize-b1",
        builder.HARD_MASK_RESIZE_BATCH2_SECTION: b"resize-b2",
    }


def test_resize_contract_uses_exact_pytorch_kernel_and_native_output_branch() -> None:
    module_source = inspect.getsource(exporter._make_module)
    assert "torch.nn.functional.interpolate" in module_source
    assert 'mode="bilinear"' in module_source
    assert "align_corners=False" in module_source
    assert "onnx" not in (module_source + inspect.getsource(builder)).lower()

    plan_source = inspect.getsource(builder._build_hard_mask_resize_plan)
    assert '"tracker_mask"' in plan_source
    assert "(batch_size, 1, 288, 288)" in plan_source
    assert '"resized_tracker_mask"' in plan_source
    source_dir = Path(builder.__file__).with_name("native_plugins")
    plugin_source = (source_dir / "sam3_tracker_memory_ffi_plugin.cpp").read_text()
    bridge_source = (source_dir / "sam3_tracker_memory_aoti_bridge.cpp").read_text()
    assert 'policy != "resize"' in plugin_source
    assert "resize_output_dimensions_valid" in plugin_source
    assert 'entry.policy == "resize"' in bridge_source
    assert "copy_resize_output" in bridge_source
    assert "{batch_size, 1, 1008, 1008}" in bridge_source
