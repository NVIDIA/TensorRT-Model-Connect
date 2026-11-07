# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free checkpoint ownership and semantic Task declarations."""

from families.lfm2_vl.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_checkpoint_has_one_owner_and_image_text_default():
    metadata = ModelMetadata({"model_type": "lfm2_vl"}, {})
    family, support = resolve_family(metadata)
    assert family == "lfm2_vl"
    assert support.default_task == "images_text_to_text"
    assert support.tasks == (
        "images_text_to_text",
        "text_conversation",
        "vision_language_generation",
    )


def test_text_only_checkpoint_does_not_select_vision_owner():
    assert describe(ModelMetadata({"model_type": "lfm2"}, {})) is None
