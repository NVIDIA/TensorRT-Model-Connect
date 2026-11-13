# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-to-text translation; speech encoder, speech generation and vocoder are not advertised."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("seamless_m4t_v2",),
    tasks=("text_translation",),
    default_task="text_translation",
    default_precision="fp32",
)
