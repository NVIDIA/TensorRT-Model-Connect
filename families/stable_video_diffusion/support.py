# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Stable Video Diffusion."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    pipeline_classes=("StableVideoDiffusionPipeline",),
    # The pipeline is conditioned on one initial image; it has no text encoder.
    tasks=("initial_image_text_to_video",),
    default_task="initial_image_text_to_video",
    # The published usage runs the denoiser and image encoder in FP16.
    default_precision="fp16",
)
