# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for DETR."""

from families.detr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_is_image_to_boxes() -> None:
    support = describe(ModelMetadata(config={"model_type": "detr"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_boxes",)
    assert support.default_task == "image_to_boxes"


def test_unrelated_model_type_is_not_claimed() -> None:
    assert describe(ModelMetadata(config={"model_type": "yolov8"}, model_index={})) is None


def test_missing_model_type_is_not_claimed() -> None:
    assert describe(ModelMetadata(config={}, model_index={})) is None
