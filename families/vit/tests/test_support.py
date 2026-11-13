# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for the transformers ViT."""

from families.vit.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_class_scores_are_offered():
    support = describe(ModelMetadata(config={"model_type": "vit"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_class_scores",)
    assert support.default_task == "image_to_class_scores"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["ViTForImageClassification"]}, model_index={}))
    assert support is not None
    assert support.default_task == "image_to_class_scores"


def test_the_timm_vit_spellings_are_not_claimed():
    # families/timm_vit covers the timm packaging of the same architecture and
    # claims its own model_types. This family covers the transformers one.
    # Claiming both would make discovery ambiguous for every timm checkpoint.
    for model_type in ("timm_vit", "timmvit", "vision_transformer",
                       "vit_base_patch16_224"):
        assert describe(ModelMetadata(config={"model_type": model_type},
                                      model_index={})) is None


def test_the_neighbouring_vits_are_not_claimed():
    # Each of these is its own model_type with its own layout: DINOv2 adds
    # LayerScale, SigLIP pools with a learned query, DeiT adds a distillation
    # token that shifts every position by one row.
    for model_type in ("dinov2", "siglip", "clip", "deit", "vit_mae", "vit_msn"):
        assert describe(ModelMetadata(config={"model_type": model_type},
                                      model_index={})) is None


def test_the_headless_and_masked_models_are_not_claimed_as_classifiers():
    # ViTModel has no classifier and ViTForMaskedImageModeling has a decoder
    # instead; neither produces class scores.
    for architecture in ("ViTModel", "ViTForMaskedImageModeling"):
        support = describe(
            ModelMetadata(config={"architectures": [architecture]}, model_index={}))
        assert support is None or support.tasks == ("image_to_class_scores",)
