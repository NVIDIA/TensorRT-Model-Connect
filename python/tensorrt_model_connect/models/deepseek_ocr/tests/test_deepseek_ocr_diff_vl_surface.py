# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tensorrt_model_connect.models.deepseek_ocr import vl_debug_runner


DIFF_VL_SURFACE = {
    "TrtRunner",
    "VLTrtRunner",
    "VisionTrtRunner",
    "load_config_from_bundle",
    "load_engine_from_bundle",
    "load_preprocessor_config_from_bundle",
    "load_section_from_bundle",
    "load_vision_engine_from_bundle",
    "preprocess_image_inputs_for_trt",
}


def test_owner_debug_runner_exposes_complete_diff_vl_surface() -> None:
    assert all(callable(getattr(vl_debug_runner, name, None)) for name in DIFF_VL_SURFACE)


def test_owner_simple_chw_preprocess_matches_deepseek_ocr_contract(
    tmp_path: Path,
) -> None:
    pixels = np.array(
        [
            [[255, 128, 0], [0, 64, 255]],
            [[32, 96, 160], [224, 192, 16]],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "input.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)

    inputs = vl_debug_runner.preprocess_image_inputs_for_trt(
        str(image_path),
        preprocessor_type="simple_chw",
        fixed_image_size=2,
        temporal_patch_size=1,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        patch_size=16,
        merge_size=1,
        interpolation="nearest",
    )

    assert set(inputs) == {"pixel_values"}
    assert inputs["pixel_values"].shape == (3, 2, 2)
    assert inputs["pixel_values"].dtype == np.float32
    np.testing.assert_allclose(
        inputs["pixel_values"],
        pixels.astype(np.float32).transpose(2, 0, 1) / 255.0,
        rtol=0.0,
        atol=0.0,
    )
