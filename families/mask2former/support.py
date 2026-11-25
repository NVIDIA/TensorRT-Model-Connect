# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for Mask2Former."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("mask2former",),
    architectures=("Mask2FormerForUniversalSegmentation",),
    tasks=("image_to_semantic_segmentation",),
    default_task="image_to_semantic_segmentation",
)
