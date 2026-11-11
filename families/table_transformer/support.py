# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity and semantic Task for Table Transformer."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("table-transformer",),
    architectures=("TableTransformerForObjectDetection",),
    tasks=("image_to_boxes",),
    default_task="image_to_boxes",
)
