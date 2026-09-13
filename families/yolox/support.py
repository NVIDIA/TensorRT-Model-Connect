# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact local checkpoint identity and public task owned by YOLOX."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    required_files=("yolox_s.pth",),
    tasks=("object_detection",),
    default_task="object_detection",
)
