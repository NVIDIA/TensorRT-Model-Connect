# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and default task for Mask2Former."""

from families.mask2former.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_checkpoint_identity_selects_semantic_segmentation():
    support = describe(
        ModelMetadata(
            config={
                "model_type": "mask2former",
                "architectures": ["Mask2FormerForUniversalSegmentation"],
            },
            model_index={},
        )
    )
    assert support is not None
    assert support.tasks == ("image_to_semantic_segmentation",)
    assert support.default_task == "image_to_semantic_segmentation"
    assert support.default_precision == "fp32"


def test_related_segmentation_checkpoints_are_not_claimed():
    for config in (
        {"model_type": "maskformer", "architectures": ["MaskFormerForInstanceSegmentation"]},
        {"model_type": "oneformer", "architectures": ["OneFormerForUniversalSegmentation"]},
        {"model_type": "segformer", "architectures": ["SegformerForSemanticSegmentation"]},
    ):
        assert describe(ModelMetadata(config=config, model_index={})) is None
