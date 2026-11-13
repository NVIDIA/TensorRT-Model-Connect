# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config resolution and build-request gates for the YOLOS family."""

from __future__ import annotations

import pytest

from families.yolos import config as config_module


def test_manifest_test_images_resolve_to_committed_assets():
    from families.yolos.tests.test_e2e import CASES, _asset

    for _, _, case in CASES.values():
        assert _asset(case["test_image"]).is_file()


def _raw(**overrides) -> dict:
    raw = {
        "model_type": "yolos",
        "architectures": ["YolosForObjectDetection"],
        "image_size": [512, 864],
        "patch_size": 16,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "layer_norm_eps": 1e-12,
        "num_detection_tokens": 100,
        "use_mid_position_embeddings": True,
    }
    raw.update(overrides)
    return raw


def test_resolves_the_native_non_square_image_size():
    """YOLOS states image_size as [height, width], not a single edge."""
    resolved = config_module.resolve(_raw())

    assert resolved["image_height"] == 512
    assert resolved["image_width"] == 864
    assert resolved["num_patches"] == (512 // 16) * (864 // 16)
    assert resolved["head_dim"] == 384 // 6


def test_a_square_image_size_is_accepted():
    resolved = config_module.resolve(_raw(image_size=512))

    assert resolved["image_height"] == 512
    assert resolved["image_width"] == 512


def test_the_token_count_covers_class_patches_and_detections():
    resolved = config_module.resolve(_raw())
    tokens = 1 + resolved["num_patches"] + resolved["num_detection_tokens"]

    # The checkpoint's position embeddings are sized for exactly this.
    assert tokens == 1829


def test_an_image_size_that_is_not_whole_patches_is_refused():
    with pytest.raises(ValueError, match="whole number of patches"):
        config_module.resolve(_raw(image_size=[513, 864]))


def test_heads_that_do_not_divide_the_width_are_refused():
    with pytest.raises(ValueError, match="divide"):
        config_module.resolve(_raw(num_attention_heads=5))


def test_a_missing_image_size_is_refused():
    raw = _raw()
    del raw["image_size"]
    with pytest.raises(ValueError, match="image_size"):
        config_module.resolve(raw)


def test_config_requires_a_model_type():
    with pytest.raises(ValueError, match="model_type"):
        config_module.ModelConfig.from_json("{}")


def test_config_reads_the_architecture():
    config = config_module.ModelConfig.from_json(
        '{"model_type": "yolos", "architectures": ["YolosForObjectDetection"]}'
    )

    assert config.model_type == "yolos"
    assert config.architecture == "YolosForObjectDetection"
