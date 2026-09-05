# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for minimax_h3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("minimax-h3", "minimax_h3", "minimaxh3"),
    pipeline_classes=("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline"),
    tasks=("image_generation",),
    default_task="image_generation",
)
