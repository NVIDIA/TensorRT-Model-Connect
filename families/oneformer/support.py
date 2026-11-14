# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for OneFormer."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("oneformer",),
    architectures=("OneFormerForUniversalSegmentation",),
    tasks=("image_to_semantic_segmentation",),
    default_task="image_to_semantic_segmentation",
)
