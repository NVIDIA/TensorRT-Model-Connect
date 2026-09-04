# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for flux."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("flux", "flux.2"),
    pipeline_classes=("Flux2Pipeline", "FluxPipeline"),
    tasks=("image_generation", "image_generation_batch"),
    default_task="image_generation",
)
