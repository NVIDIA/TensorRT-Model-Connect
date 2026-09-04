# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm_vit."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_vit", "timmvit", "vision_transformer", "vit_base_patch16_224"),
    architectures=("vit_base_patch16_224",),
    tasks=("classification",),
    default_task="classification",
)
