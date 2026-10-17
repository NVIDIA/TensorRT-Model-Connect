# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for dinov2."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("dinov2", "dinov2_with_registers"),
    tasks=("image_to_token_and_pooled_features",),
    default_task="image_to_token_and_pooled_features",
)
