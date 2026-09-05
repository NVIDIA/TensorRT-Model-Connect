# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for z_image."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("z-image", "z_image", "zimage", "zimagepipeline"),
    pipeline_classes=("ZImagePipeline",),
    tasks=("image_generation",),
    default_task="image_generation",
)
