# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for the transformers Swin Transformer V2."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("swinv2",),
    architectures=("Swinv2ForImageClassification",),
    tasks=("image_to_class_scores",),
    default_task="image_to_class_scores",
)
