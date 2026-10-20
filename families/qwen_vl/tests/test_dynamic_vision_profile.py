# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default Qwen-VL builds keep the 448-square image budget and reserve its tokens."""

import json

import pytest

pytest.importorskip("tensorrt")

from families.qwen_vl.config import ModelConfig  # noqa: E402
from families.qwen_vl.model import (  # noqa: E402
    _QwenVLModel,
    _default_sequence_length,
    _dynamic_vision_profile,
)


def _qwen25_config(options=None):
    raw = {"model_type": "qwen2_5_vl",
           "vision_config": {"patch_size": 14, "spatial_merge_size": 2}}
    if options is not None:
        raw["_family_build_options"] = {"qwen_vl_vision": options}
    return ModelConfig(model_type="qwen2_5_vl", hidden_size=2048,
                       max_position_embeddings=32768, raw=raw)


@pytest.mark.parametrize("processor_maximum", [12845056, 6422528])
def test_default_dynamic_profile_keeps_the_448_square_budget(tmp_path, processor_maximum):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "min_pixels": 3136, "max_pixels": processor_maximum}))
    config = _qwen25_config()
    assert _dynamic_vision_profile(str(tmp_path), config) == (True, 3136, 200704, 200704)
    assert _QwenVLModel().get_vl_config(config)["max_pixels"] == 200704
    # 256 image tokens at the 448-square budget, plus 256 text tokens.
    assert _default_sequence_length(str(tmp_path), config) == 512
    config.raw["_family_build_options"] = {"qwen_vl_vision": {"dynamic_resolution": False}}
    assert not _dynamic_vision_profile(str(tmp_path), config)[0]
    assert not _QwenVLModel().get_vl_config(config)["dynamic_image_resolution"]
    assert _default_sequence_length(str(tmp_path), config) == 256


@pytest.mark.parametrize("maximum,capacity", [(12845056, 16640), (6422528, 8448)])
def test_explicit_max_pixels_selects_checkpoint_geometry_and_capacity(tmp_path, maximum, capacity):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "min_pixels": 3136, "max_pixels": maximum}))
    config = _qwen25_config({"max_pixels": maximum})
    assert _dynamic_vision_profile(str(tmp_path), config) == (True, 3136, 200704, maximum)
    assert _QwenVLModel().get_vl_config(config)["max_pixels"] == maximum
    assert _default_sequence_length(str(tmp_path), config) == capacity


def test_explicit_dynamic_profile_with_image_dimensions_reserves_image_tokens(tmp_path):
    config = _qwen25_config({"dynamic_resolution": True, "image_height": 448})
    assert _dynamic_vision_profile(str(tmp_path), config)[0]
    assert _default_sequence_length(str(tmp_path), config) == 512


def test_legacy_qwen2_keeps_the_main_default_profile(tmp_path):
    config = ModelConfig(model_type="qwen2_vl", raw={"model_type": "qwen2_vl"})
    assert _dynamic_vision_profile(str(tmp_path), config) == (True, 3136, 200704, 200704)
    assert _default_sequence_length(str(tmp_path), config) == 512


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen3_vl"])
@pytest.mark.parametrize("dimension", ["image_height", "image_width"])
def test_explicit_image_dimensions_keep_the_fixed_profile(tmp_path, model_type, dimension):
    vision = {"patch_size": 16, "spatial_merge_size": 2}
    if model_type == "qwen3_vl":
        vision["deepstack_visual_indexes"] = [5, 11, 17]
    config = ModelConfig(model_type=model_type, raw={
        "model_type": model_type,
        "vision_config": vision,
        "_family_build_options": {"qwen_vl_vision": {dimension: 448}},
    })
    assert not _dynamic_vision_profile(str(tmp_path), config)[0]
    assert not _QwenVLModel().get_vl_config(config)["dynamic_image_resolution"]
    assert _default_sequence_length(str(tmp_path), config) == 256
