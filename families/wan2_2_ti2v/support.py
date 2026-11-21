# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for wan2_2_ti2v."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("ti2v",),
    pipeline_classes=("WanModel",),
    # One bundle serves text-to-video and, with its first-frame VAE encoder, image-to-video.
    # image_generation stays the default bundle mode for the existing video endpoint.
    tasks=("image_generation", "text_to_video", "initial_image_text_to_video"),
    default_task="image_generation",
)
