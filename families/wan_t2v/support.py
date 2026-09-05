# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for wan_t2v."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("wan", "wan2.1", "wan_t2v", "want2v"),
    pipeline_classes=("WanPipeline", "WanVideoToVideoPipeline"),
    tasks=("image_generation",),
    default_task="image_generation",
)
