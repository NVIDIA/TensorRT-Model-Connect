# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Stable Diffusion 3 and 3.5."""

from tensorrt_model_connect.model_support import family_support


# The model card runs SD3.5 in BF16 (the denoiser is stored in BF16, the text
# encoders in FP16); that is also the family's default build.
describe = family_support(
    pipeline_classes=("StableDiffusion3Pipeline",),
    tasks=("text_to_image",),
    default_task="text_to_image",
    default_precision="bf16",
)
