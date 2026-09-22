# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact local checkpoint identity and public task owned by YOLOX."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


ARCHIVES = (
    "yolox_nano.pth",
    "yolox_tiny.pth",
    "yolox_s.pth",
    "yolox_m.pth",
    "yolox_l.pth",
    "yolox_x.pth",
    "yolox_darknet.pth",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if any(name in metadata.files for name in ARCHIVES):
        return FamilySupport(tasks=("object_detection",), default_task="object_detection")
    return None
