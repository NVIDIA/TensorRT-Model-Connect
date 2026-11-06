# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for YOLOv5."""

from families.yolov5.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_only_task_is_image_to_boxes() -> None:
    support = describe(ModelMetadata(config={}, model_index={}, files=("yolov5n.pt",)))
    assert support is not None
    assert support.tasks == ("image_to_boxes",)
    assert support.default_task == "image_to_boxes"
