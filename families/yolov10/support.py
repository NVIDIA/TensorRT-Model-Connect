# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for yolov10.

The published YOLOv10 checkpoints carry no `model_type` and no `architectures`,
so identity comes from the exact root JSON shape this family owns: a `model`
naming a YOLOv10 configuration, a `task` of `detect`, and a non-empty `names`
map. That is the exact-shape route the contributor guide prescribes when an
upstream repository has no standard identity field; it never matches on a
repository name, and every clause has to hold.
"""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


_SUPPORT = FamilySupport(tasks=("object_detection",), default_task="object_detection")


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    config = metadata.config
    if not isinstance(config, dict):
        return None
    model = config.get("model")
    names = config.get("names")
    if config.get("task") != "detect":
        return None
    if not isinstance(model, str) or not model.startswith("yolov10"):
        return None
    if not isinstance(names, dict) or not names:
        return None
    return _SUPPORT
