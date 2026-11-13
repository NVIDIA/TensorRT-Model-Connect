# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for the transformers ViT."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("vit",),
    architectures=("ViTForImageClassification",),
    tasks=("image_to_class_scores",),
    default_task="image_to_class_scores",
    # fp16 is faster than fp32 on every GPU measured and passes its own E2E
    # gate (tests/test_e2e.py); --precision fp32 still builds the fp32 graph.
    default_precision="fp16",
)
