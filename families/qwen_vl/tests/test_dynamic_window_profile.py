# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dynamic vision profile must admit upstream smart-resize window padding."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")


@pytest.mark.parametrize("height,width", [(2, 384), (384, 2), (4, 256), (32, 32), (10, 102)])
def test_profile_admits_upstream_padding_for_thin_and_regular_images(height, width):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
    from families.qwen_vl.qwen_vl_vision_builder import _max_window_padded_rows

    # The 192:1 cases are valid under Qwen's <=200 aspect limit and the
    # default 448-square pixel budget. They need 3072 padded rows, exceeding
    # the old profile's 2048-row bound.
    geometry = SimpleNamespace(window_size=112, patch_size=14,
                               spatial_merge_size=2, spatial_merge_unit=4)
    _, cumulative = Qwen2_5_VisionTransformerPretrainedModel.get_window_index(
        geometry, torch.tensor([[1, height, width]]))
    windows = len(torch.unique_consecutive(torch.tensor(cumulative))) - 1
    actual_padding = windows * 64
    assert height * width <= 1024
    assert actual_padding <= _max_window_padded_rows(1024, 4, 4)
    if min(height, width) == 2:
        assert actual_padding == 3072


@pytest.mark.parametrize("height,width,minimum", [
    (100, 20000, 65536), (20000, 100, 65536), (32, 6400, 65536), (6400, 32, 65536),
    (320, 427, 65536), (427, 320, 65536), (32, 2016, 65536), (32, 32, 65536),
    (32, 6200, 448**2),
])
def test_qwen3_profile_admits_upstream_resize_at_the_aspect_limit(height, width, minimum):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from families.qwen_vl.qwen_vl_vision_builder import _qwen3_max_profile_patches

    resized_h, resized_w = smart_resize(
        height, width, factor=32, min_pixels=minimum, max_pixels=448**2,
    )
    patches = resized_h * resized_w // 16**2
    assert patches <= _qwen3_max_profile_patches(minimum, 448**2, 16, 2)
    if max(height, width) / min(height, width) == 200 or minimum == 448**2:
        assert patches > 784  # The old nominal-area profile rejected this valid shape.
