# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for OneFormer."""

from families.oneformer.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_is_image_to_semantic_segmentation() -> None:
    """OneFormer advertises image_to_semantic_segmentation as its default and only Task."""
    support = describe(ModelMetadata(config={"model_type": "oneformer"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_semantic_segmentation",)
    assert support.default_task == "image_to_semantic_segmentation"


def test_architecture_identifies_the_family() -> None:
    """The universal-segmentation architecture alone selects the family."""
    config = {"architectures": ["OneFormerForUniversalSegmentation"]}
    support = describe(ModelMetadata(config=config, model_index={}))
    assert support is not None and support.default_task == "image_to_semantic_segmentation"


def test_related_segmentation_models_are_not_claimed() -> None:
    """MaskFormer, Mask2Former and SegFormer checkpoints belong elsewhere."""
    for model_type, architecture in (
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("mask2former", "Mask2FormerForUniversalSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
    ):
        metadata = ModelMetadata(
            config={"model_type": model_type, "architectures": [architecture]}, model_index={}
        )
        assert describe(metadata) is None


def test_missing_model_type_is_not_claimed() -> None:
    """A config without identity is not claimed."""
    assert describe(ModelMetadata(config={}, model_index={})) is None
