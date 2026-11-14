# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for fast_foundation_stereo."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("fast-foundation-stereo", "fast_foundation_stereo", "foundation_stereo_lite"),
    required_files=("weights/23-36-37/model_best_bp2_serialize.pth",),
    tasks=("stereo_disparity",),
    default_task="stereo_disparity",
)
