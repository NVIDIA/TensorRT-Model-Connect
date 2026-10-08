# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for YOLO11."""

from families.yolo11.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(ModelMetadata(config={}, model_index={}, files=("yolo11n.pt",)))
    assert support is not None
    assert support.tasks == ("image_to_boxes",)
    assert support.default_task == "image_to_boxes"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={}, model_index={}, files=("unrelated.pt",))) is None
