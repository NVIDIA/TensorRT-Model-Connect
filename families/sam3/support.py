# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for sam3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("sam3", "sam3-video", "sam3_video"),
    tasks=("text_prompted_segmentation",),
    default_task="text_prompted_segmentation",
)
