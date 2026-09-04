# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for dinov3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("dinov3_vit", "dinov3_convnext", "vit_small_patch16_dinov3_qkvb"),
    architectures=("vit_small_patch16_dinov3_qkvb",),
    tasks=("image_features",),
    default_task="image_features",
)
