# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for the transformers Swin Transformer V2."""

from families.swinv2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_class_scores_are_offered():
    support = describe(ModelMetadata(config={"model_type": "swinv2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_class_scores",)
    assert support.default_task == "image_to_class_scores"
    # The checkpoint stores float32 and only float32 is qualified.
    assert support.default_precision == "fp32"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["Swinv2ForImageClassification"]}, model_index={}))
    assert support is not None
    assert support.default_task == "image_to_class_scores"


def test_swin_v1_and_the_swinv2_derived_models_are_not_claimed():
    # families/timm_swin owns the timm packaging of Swin V1. The transformers
    # Swin V1, Swin2SR (a super-resolution model built from SwinV2 blocks) and
    # the Donut encoder are their own model types with their own layouts.
    for model_type in ("swin", "timm_swin", "swin2sr", "donut-swin"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
    for architecture in ("SwinForImageClassification", "Swin2SRForImageSuperResolution"):
        assert describe(
            ModelMetadata(config={"architectures": [architecture]}, model_index={})) is None


def test_the_headless_and_masked_models_are_not_claimed_as_classifiers():
    # Swinv2Model has no classifier, Swinv2ForMaskedImageModeling has a pixel
    # decoder and Swinv2Backbone returns feature maps. None of them is claimed
    # by architecture; with model_type swinv2 the builder refuses them.
    for architecture in ("Swinv2Model", "Swinv2ForMaskedImageModeling", "Swinv2Backbone"):
        assert describe(
            ModelMetadata(config={"architectures": [architecture]}, model_index={})) is None
