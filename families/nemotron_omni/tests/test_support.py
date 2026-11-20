# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-visible exact checkpoint ownership and default semantic task."""

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_omni_identity_and_default_task():
    family, support = resolve_family(
        ModelMetadata(
            config={
                "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
                "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
            },
            model_index={},
        )
    )
    assert family == "nemotron_omni"
    assert support.default_task == "vision_language_generation"
    assert {
        "text_continuation",
        "images_text_to_text",
        "audio_text_to_text",
        "video_text_to_text",
        "image_audio_text_to_text",
        "audio_video_text_to_text",
    } <= set(support.tasks)
