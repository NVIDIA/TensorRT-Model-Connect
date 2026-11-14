# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
from families.qwen_vl.model import _dynamic_vision_profile, _QwenVLModel


@pytest.mark.parametrize("qwen3", [False, True])
def test_bundle_normalization_matches_the_checkpoint_processor(qwen3):
    vision = {"patch_size": 14, "spatial_merge_size": 2}
    if qwen3:
        vision["deepstack_visual_indexes"] = [5, 11, 17]
    config = SimpleNamespace(raw={"vision_config": vision}, hidden_size=16)
    runtime = _QwenVLModel().get_vl_config(config)
    black_white = np.asarray([[0.0] * 3, [1.0] * 3])
    actual = (black_white - runtime["image_mean"]) / runtime["image_std"]
    if qwen3:
        np.testing.assert_allclose(actual, [[-1.0] * 3, [1.0] * 3])
    else:
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073])
        std = np.asarray([0.26862954, 0.26130258, 0.27577711])
        np.testing.assert_allclose(actual, (black_white - mean) / std)


@pytest.mark.parametrize("qwen3", [False, True])
def test_profile_preserves_smart_resize_within_the_existing_token_budget(tmp_path, qwen3):
    import json

    vision = {"patch_size": 16 if qwen3 else 14, "spatial_merge_size": 2}
    if qwen3:
        vision["deepstack_visual_indexes"] = [5, 11, 17]
    config = SimpleNamespace(raw={"vision_config": vision}, hidden_size=16)
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    }))
    enabled, minimum, optimum, maximum = _dynamic_vision_profile(str(tmp_path), config)
    assert enabled and minimum == 65536
    # Same image-token budget as the previous 448-square bundles.
    tokens = 196 if qwen3 else 256
    assert optimum == maximum == tokens * 4 * vision["patch_size"]**2
    runtime = _QwenVLModel().get_vl_config(config)
    assert runtime["preprocessor_type"] == (
        "qwen3_smart_resize_patchify" if qwen3 else "qwen_smart_resize_patchify")
    assert runtime["num_image_pad_tokens"] == 0
    if qwen3:
        assert runtime["vision_position_grid_size"] == 48
    config.raw["_family_build_options"] = {"qwen_vl_vision": {"dynamic_resolution": False}}
    assert not _dynamic_vision_profile(str(tmp_path), config)[0]
    assert _QwenVLModel().get_vl_config(config)["preprocessor_type"] == "merge_group_chw"
