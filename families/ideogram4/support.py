# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ideogram4 owns its asymmetric, two-denoiser text-to-image pipeline."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    pipeline_classes=("Ideogram4Pipeline",),
    tasks=("text_to_image",),
    default_task="text_to_image",
    default_precision="bf16",
)
