# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config resolution for the Depth Anything family."""

from __future__ import annotations

import pytest

from families.depth_anything import config as config_module
from families.depth_anything.model import _count_encoder_layers


def _raw(**overrides) -> dict:
    raw = {
        "model_type": "depth_anything",
        "architectures": ["DepthAnythingForDepthEstimation"],
        "backbone_config": {
            "model_type": "dinov2",
            "hidden_size": 384,
            "num_attention_heads": 6,
            "image_size": 518,
            "patch_size": 14,
            "out_indices": [3, 6, 9, 12],
        },
        "reassemble_factors": [4, 2, 1, 0.5],
        "neck_hidden_sizes": [48, 96, 192, 384],
        "fusion_hidden_size": 64,
        "head_hidden_size": 32,
        "head_in_index": -1,
        "reassemble_hidden_size": 384,
    }
    raw.update(overrides)
    return raw


def test_resolves_a_backbone_that_states_only_geometry():
    """The checkpoint omits num_hidden_layers and layer_norm_eps entirely."""
    resolved = config_module.resolve(_raw())

    assert resolved["hidden_size"] == 384
    assert resolved["head_dim"] == 64
    assert resolved["patch_grid"] == 518 // 14
    # transformers' Dinov2Config supplies both of these.
    assert resolved["num_hidden_layers"] == 12
    assert resolved["layer_norm_eps"] == pytest.approx(1e-6)


def test_out_indices_are_converted_to_encoder_layer_indices():
    """out_indices are 1-based stage numbers over hidden states, not layers."""
    resolved = config_module.resolve(_raw())

    assert resolved["out_layer_indices"] == [2, 5, 8, 11]


def test_relative_depth_defaults_carry_through():
    resolved = config_module.resolve(_raw())

    assert resolved["max_depth"] == pytest.approx(1.0)


def test_a_metric_checkpoint_is_refused_rather_than_guessed():
    with pytest.raises(NotImplementedError, match="relative"):
        config_module.resolve(_raw(depth_estimation_type="metric"))


def test_an_image_size_that_is_not_whole_patches_is_refused():
    raw = _raw()
    raw["backbone_config"]["image_size"] = 519
    with pytest.raises(ValueError, match="whole number of patches"):
        config_module.resolve(raw)


def test_a_missing_backbone_is_refused():
    raw = _raw()
    del raw["backbone_config"]
    with pytest.raises(ValueError, match="backbone_config"):
        config_module.resolve(raw)


def test_layer_count_is_read_from_the_checkpoint():
    tensors = {f"backbone.encoder.layer.{index}.norm1.weight": None for index in range(6)}
    assert _count_encoder_layers(tensors) == 6


def test_a_checkpoint_with_gaps_in_its_layers_is_refused():
    tensors = {"backbone.encoder.layer.0.norm1.weight": None,
               "backbone.encoder.layer.2.norm1.weight": None}
    with pytest.raises(ValueError, match="contiguous"):
        _count_encoder_layers(tensors)
