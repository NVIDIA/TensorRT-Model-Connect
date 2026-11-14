# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned discovery for MiniCPM-V 4.7 checkpoints."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("minicpmv4_7",),
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
    default_precision="bf16",
)
