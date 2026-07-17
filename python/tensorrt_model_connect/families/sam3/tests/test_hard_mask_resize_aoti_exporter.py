# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest

from tensorrt_model_connect.families.sam3 import hard_mask_resize_aoti_exporter as exporter


@pytest.mark.parametrize("error", [float("nan"), float("inf"), float("-inf"), -1.0, 2.1e-5])
def test_resize_smoke_error_rejects_non_finite_and_out_of_range(error: float) -> None:
    assert not exporter._valid_smoke_error(error)


@pytest.mark.parametrize("error", [0.0, 1.0e-6, 2.0e-5])
def test_resize_smoke_error_accepts_finite_tolerance(error: float) -> None:
    assert exporter._valid_smoke_error(error)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resize_global_binds_batch_and_content_digest(batch_size: int) -> None:
    digest = "a" * 64
    assert exporter._global_name(batch_size, digest) == (
        f"trtmc.sam3.tracker_memory.resize.b{batch_size}.fixed.{digest[:20]}"
    )

    with pytest.raises(ValueError, match="Invalid SAM3 hard-mask resize package identity"):
        exporter._global_name(3, digest)
    with pytest.raises(ValueError, match="Invalid SAM3 hard-mask resize package identity"):
        exporter._global_name(batch_size, "not-a-digest")


def test_resize_golden_uses_exact_meta_pytorch_kernel() -> None:
    module_source = inspect.getsource(exporter._make_module)
    assert "torch.nn.functional.interpolate" in module_source
    assert "size=(_TRACKER_IMAGE_SIZE, _TRACKER_IMAGE_SIZE)" in module_source
    assert 'mode="bilinear"' in module_source
    assert "align_corners=False" in module_source
    assert ".contiguous()" in module_source
    assert "onnx" not in module_source.lower()


def test_resize_golden_compiles_and_validates_b1_b2_packages() -> None:
    compile_source = inspect.getsource(exporter._compile_and_validate)
    assert "torch.export.export" in compile_source
    assert "aoti_compile_and_package" in compile_source
    assert "aoti_load_package" in compile_source
    assert "maximum_absolute_error" in compile_source
    assert "_valid_smoke_error" in compile_source

    export_source = inspect.getsource(exporter.export_sam3_hard_mask_resize_aoti)
    assert "for batch_size in (1, 2)" in export_source
    assert "_compile_and_validate" in export_source
