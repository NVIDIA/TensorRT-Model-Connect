# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and default task for ViTPose."""

from families.vitpose.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_checkpoint_identity_selects_keypoint_points():
    support = describe(
        ModelMetadata(
            config={"model_type": "vitpose", "architectures": ["VitPoseForPoseEstimation"]},
            model_index={},
        )
    )
    assert support is not None
    assert support.tasks == ("image_text_to_points",)
    assert support.default_task == "image_text_to_points"
    assert support.default_precision == "fp32"


def test_related_vision_checkpoints_are_not_claimed():
    for config in (
        {"model_type": "vitpose_backbone", "architectures": ["VitPoseBackbone"]},
        {"model_type": "vit", "architectures": ["ViTForImageClassification"]},
        {"model_type": "vitmatte", "architectures": ["VitMatteForImageMatting"]},
        {"model_type": "dinov2", "architectures": ["Dinov2Model"]},
    ):
        assert describe(ModelMetadata(config=config, model_index={})) is None
