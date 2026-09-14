# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for YOLOv5."""

from tensorrt_model_connect.model_support import family_support


# A YOLOv5 release carries no config.json, so the archive itself is the
# identity. Each published width is named separately rather than matched by a
# prefix, so a directory holding some other yolov5-ish file is not claimed.
describe = family_support(
    required_files=("yolov5n.pt",),
    tasks=("object_detection",),
    default_task="object_detection",
)
