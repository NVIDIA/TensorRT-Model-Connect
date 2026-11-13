# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_pika_h3_flex_checkpoint_has_one_owner():
    family, support = resolve_family(
        ModelMetadata({}, {"_class_name": "PikaH3FlexModularPipeline"})
    )
    assert family == "pika_h3_flex"
    assert support.tasks == ("text_to_audio_video",)
    assert support.default_precision == "bf16"
