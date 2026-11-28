# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for ViTPose."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("vitpose",),
    architectures=("VitPoseForPoseEstimation",),
    tasks=("image_text_to_points",),
    default_task="image_text_to_points",
    default_precision="fp32",
)
