# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for DETR."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("detr",),
    architectures=("DetrForObjectDetection",),
    tasks=("object_detection",),
    default_task="object_detection",
)
