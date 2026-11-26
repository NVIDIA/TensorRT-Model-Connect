# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the BEV-only execution of the official BEVFusion segmentation checkpoint."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("bevfusion_bev_seg",),
    required_files=("bevfusion-seg.pth",),
    tasks=("spatial_features_to_class_scores",),
    default_task="spatial_features_to_class_scores",
    default_precision="fp32",
)
