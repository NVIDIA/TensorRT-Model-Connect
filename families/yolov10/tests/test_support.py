# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for YOLOv10."""

from families.yolov10.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_only_task_is_image_to_boxes() -> None:
    config = {"model": "yolov10n.yaml", "task": "detect", "names": {"0": "person"}}
    support = describe(ModelMetadata(config=config, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_boxes",)
    assert support.default_task == "image_to_boxes"
