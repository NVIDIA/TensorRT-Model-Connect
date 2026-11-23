# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint identity and task support for MiniCPM-OCR."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("minicpm_ocr",),
    architectures=("MiniCPMOcrForConditionalGeneration",),
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
    default_precision="fp16",
)
