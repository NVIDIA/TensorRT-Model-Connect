# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for OWLv2."""

from families.owlv2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_text_prompted_detection_is_offered():
    support = describe(ModelMetadata(config={"model_type": "owlv2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_text_to_boxes",)
    assert support.default_task == "image_text_to_boxes"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["Owlv2ForObjectDetection"]}, model_index={})
    )
    assert support is not None
    assert support.default_task == "image_text_to_boxes"


def test_owlvit_is_not_claimed():
    # OWL-ViT is the previous generation: no objectness head, a different
    # image processor, and it pads differently. It needs its own family rather
    # than a shared one that is right for neither.
    assert describe(ModelMetadata(config={"model_type": "owlvit"}, model_index={})) is None
