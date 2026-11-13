# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the birefnet builders that need no checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.birefnet import config, swin_builder


def test_shifted_window_mask_blocks_only_across_regions():
    mask = swin_builder.shifted_window_mask(14, 14, 7, 3)
    assert mask.shape == (4, 49, 49)
    # Within a region the bias is zero; across regions it is -100.
    assert set(np.unique(mask)).issubset({0.0, -100.0})
    assert (mask == 0.0).any() and (mask == -100.0).any()
    # The diagonal is always within-region, so it is never blocked.
    for window in range(mask.shape[0]):
        assert np.allclose(np.diag(mask[window]), 0.0)


def test_unshifted_stage_needs_no_mask():
    # With no shift every window is one region, so a mask would be all zeros.
    mask = swin_builder.shifted_window_mask(14, 14, 7, 0)
    assert np.allclose(mask, 0.0)


def test_relative_position_bias_matches_the_gather():
    heads, area = 3, 49
    table = np.arange(169 * heads, dtype=np.float32).reshape(169, heads)
    index = np.arange(area * area, dtype=np.int64).reshape(area, area) % 169
    bias = swin_builder.relative_position_bias(table, index, heads, area)
    assert bias.shape == (heads, area, area)
    assert bias[0, 0, 0] == table[index[0, 0], 0]
    assert bias[2, 5, 7] == table[index[5, 7], 2]


def test_resolve_rejects_a_misaligned_size(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        config.resolve(tmp_path, image_size=1000)


def test_check_weights_rejects_a_larger_backbone():
    weights = {"bb.patch_embed.proj.weight": np.zeros((128, 3, 4, 4), dtype=np.float32)}
    with pytest.raises(NotImplementedError):
        config.check_weights(weights)


def test_check_weights_requires_the_optional_paths():
    # The checkpoint turns dec_ipt and out_ref on; a build without them would
    # silently drop whole branches instead of failing.
    weights = {"bb.patch_embed.proj.weight": np.zeros((96, 3, 4, 4), dtype=np.float32)}
    with pytest.raises(NotImplementedError):
        config.check_weights(weights)


def test_manifest_threshold_is_declared():
    path = ("families/birefnet/tests/manifests/birefnet-lite.json")
    case = json.loads(open(path, encoding="utf-8").read())["testcases"][0]
    assert 0.0 < float(case["min_mask_iou"]) <= 1.0
