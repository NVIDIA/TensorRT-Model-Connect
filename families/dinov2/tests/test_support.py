# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for DINOv2."""

from families.dinov2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_token_and_pooled_task_is_offered():
    support = describe(ModelMetadata(config={"model_type": "dinov2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_token_and_pooled_features",)
    assert support.default_task == "image_to_token_and_pooled_features"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["Dinov2Model"]}, model_index={}))
    assert support is not None
    assert support.default_task == "image_to_token_and_pooled_features"


def test_the_registers_variant_is_not_claimed():
    # dinov2_with_registers is its own model_type: it adds a register_tokens
    # tensor and four extra rows between the class token and the patches, so
    # this family would build it with the wrong sequence layout.
    assert describe(ModelMetadata(config={"model_type": "dinov2_with_registers"},
                                  model_index={})) is None
    assert describe(ModelMetadata(config={"architectures": ["Dinov2WithRegistersModel"]},
                                  model_index={})) is None


def test_the_neighbouring_backbones_are_not_claimed():
    # dinov3 is a separate family in this repo, and the rest are ViTs with
    # neither LayerScale nor an interpolated position table.
    for model_type in ("dinov3_vit", "dinov3_convnext", "vit", "clip", "siglip"):
        assert describe(ModelMetadata(config={"model_type": model_type},
                                      model_index={})) is None


def test_the_backbone_head_is_not_claimed_as_a_different_task():
    # Dinov2ForImageClassification has a classifier on top; this family builds
    # the backbone only and must not silently answer for the head.
    support = describe(
        ModelMetadata(config={"architectures": ["Dinov2ForImageClassification"]},
                      model_index={}))
    assert support is None or support.tasks == ("image_to_token_and_pooled_features",)
