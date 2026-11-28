# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Muse-Glimmer."""

from tensorrt_model_connect.model_support import FamilySupport, family_support


_text = family_support(
    model_types=("muse_glimmer", "muse_glimmer_text"),
    architectures=("MuseGlimmerForConditionalGeneration",),
    tasks=("text_generation",),
    default_task="text_generation",
    default_precision="fp16",
)


def describe(metadata):
    support = _text(metadata)
    if (support is not None and metadata.config.get("vision_config")
            and not metadata.config.get("quantization_config")
            and "processor_config.json" in metadata.files and "hf_quant_config.json" not in metadata.files):
        return FamilySupport(("text_generation", "images_text_to_text", "video_text_to_text"),
                             "text_generation", "fp16")
    return support
