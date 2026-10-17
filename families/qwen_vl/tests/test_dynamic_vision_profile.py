# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary Qwen2.5 builds retain Native geometry and sufficient image capacity."""

import json

import pytest

pytest.importorskip("tensorrt")

from families.qwen_vl.config import ModelConfig  # noqa: E402
from families.qwen_vl.model import (  # noqa: E402
    _QwenVLModel,
    _default_sequence_length,
    _dynamic_vision_profile,
)


@pytest.mark.parametrize("maximum,capacity", [(12845056, 16640), (6422528, 8448)])
def test_default_dynamic_profile_uses_checkpoint_pixel_limits(tmp_path, maximum, capacity):
    config = ModelConfig(model_type="qwen2_5_vl", hidden_size=2048, max_position_embeddings=32768,
                         raw={"model_type": "qwen2_5_vl",
                              "vision_config": {"patch_size": 14, "spatial_merge_size": 2}})
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "min_pixels": 3136, "max_pixels": maximum}))
    assert _dynamic_vision_profile(str(tmp_path), config) == (True, 3136, 200704, maximum)
    assert _QwenVLModel().get_vl_config(config)["dynamic_image_resolution"]
    assert _default_sequence_length(str(tmp_path), config) == capacity
    config.raw["_family_build_options"] = {"qwen_vl_vision": {"dynamic_resolution": False}}
    assert not _dynamic_vision_profile(str(tmp_path), config)[0]
    assert not _QwenVLModel().get_vl_config(config)["dynamic_image_resolution"]
    assert _default_sequence_length(str(tmp_path), config) == 256


def test_legacy_qwen2_keeps_its_existing_default(tmp_path):
    config = ModelConfig(model_type="qwen2_vl", raw={"model_type": "qwen2_vl"})
    assert not _dynamic_vision_profile(str(tmp_path), config)[0]
    assert _default_sequence_length(str(tmp_path), config) == 256
