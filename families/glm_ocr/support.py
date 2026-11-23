# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for GLM-OCR."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("glm_ocr",),
    architectures=("GlmOcrForConditionalGeneration",),
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
    # The checkpoint stores its weights in BF16 and Transformers runs it in
    # BF16 by default; --precision fp32 builds the FP32 graph.
    default_precision="bf16",
)
