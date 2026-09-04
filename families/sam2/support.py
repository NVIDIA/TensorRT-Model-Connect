# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for sam2."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("sam2", "sam2_video_tracking", "sam2_bbox_video_tracking"),
    required_files=("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    tasks=("video_segmentation",),
    default_task="video_segmentation",
)
