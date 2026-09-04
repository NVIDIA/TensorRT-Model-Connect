# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for cosmos3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("cosmos3", "cosmos3_nano", "cosmos3-nano"),
    pipeline_classes=("Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline", "Cosmos3OmniModularPipeline"),
    tasks=("image_generation",),
    default_task="image_generation",
)
