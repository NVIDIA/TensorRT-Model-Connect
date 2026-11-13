# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for CLIP."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("clip",),
    architectures=("CLIPModel",),
    tasks=("image_to_embedding", "text_to_embedding", "text_image_to_relevance"),
    default_task="text_image_to_relevance",
)
