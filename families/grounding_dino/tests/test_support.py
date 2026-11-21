# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for Grounding DINO."""

from families.grounding_dino.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(ModelMetadata(config={"model_type": "grounding-dino"}, model_index={}))
    assert support is not None
    # The semantic Task SDK contract, not the fixed-label object_detection one:
    # the prompt defines the labels, so there is no class vocabulary to report.
    assert support.tasks == ("image_text_to_boxes",)
    assert support.default_task == "image_text_to_boxes"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["GroundingDinoForObjectDetection"]},
                      model_index={})
    )
    assert support is not None
    assert support.default_task == "image_text_to_boxes"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "detr"}, model_index={})) is None
