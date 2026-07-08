# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image static build-geometry regression tests."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.qwen_image.config import ModelConfig
from tensorrt_model_connect.families.qwen_image.plugin import (
    _apply_static_image_geometry,
)


def _bundle_config() -> dict:
    return {
        "image": {
            "default_height": 1024,
            "default_width": 1024,
            "min_height": 256,
            "min_width": 256,
            "max_height": 2048,
            "max_width": 2048,
            "height_alignment": 16,
            "width_alignment": 16,
        },
        "vae": {"spatial_scale_factor": 8},
        "denoiser": {"patch_size": 2},
    }


def test_qwen_image_build_flags_drive_dit_vae_and_bundle_geometry() -> None:
    config = ModelConfig.create_tiny(
        "qwen_image",
        image_height=512,
        image_width=768,
    )
    bundle = _bundle_config()

    geometry = _apply_static_image_geometry(config, bundle)

    assert geometry == (512, 768, 64, 96, 32, 48)
    assert bundle["image"]["default_height"] == 512
    assert bundle["image"]["default_width"] == 768


def test_qwen_image_build_geometry_defaults_to_bundle_size() -> None:
    config = ModelConfig.create_tiny("qwen_image")
    bundle = _bundle_config()

    assert _apply_static_image_geometry(config, bundle) == (
        1024,
        1024,
        128,
        128,
        64,
        64,
    )


@pytest.mark.parametrize(
    ("height", "width", "message"),
    [
        (240, 512, "image_height must be in"),
        (512, 2050, "image_width must be in"),
        (510, 512, "image_height must be divisible"),
    ],
)
def test_qwen_image_build_geometry_rejects_unbuildable_shapes(
    height: int,
    width: int,
    message: str,
) -> None:
    config = ModelConfig.create_tiny(
        "qwen_image",
        image_height=height,
        image_width=width,
    )

    with pytest.raises(ValueError, match=message):
        _apply_static_image_geometry(config, _bundle_config())
