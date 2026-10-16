# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")
from families.internvl.model import _InternVLModel


def _config(model_dir=None):
    raw = {"vision_config": {"patch_size": 14}}
    if model_dir is not None:
        raw["_model_dir"] = str(model_dir)
    return SimpleNamespace(raw=raw, hidden_size=16)


def test_runtime_tiles_images_like_internvl_processor():
    # InternVLProcessor crops every image to 1..12 448 px tiles plus a
    # thumbnail (GotOcr2ImageProcessor defaults).
    runtime = _InternVLModel().get_vl_config(_config())
    assert runtime["preprocessor_type"] == "dynamic_tiles_chw"
    assert (runtime["min_image_tiles"], runtime["max_image_tiles"]) == (1, 12)
    assert runtime["use_thumbnail"] is True
    assert runtime["fixed_image_size"] == 448
    assert runtime["num_image_pad_tokens"] == 256


def test_tile_limits_follow_the_checkpoint_processor(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"min_patches": 2, "max_patches": 6, "crop_to_patches": False})
    )
    runtime = _InternVLModel().get_vl_config(_config(tmp_path))
    assert (runtime["min_image_tiles"], runtime["max_image_tiles"]) == (2, 6)
