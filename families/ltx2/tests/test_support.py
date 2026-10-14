# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_ltx2_pipeline_resolves_to_text_to_audio_video() -> None:
    family, support = resolve_family(ModelMetadata({}, {"_class_name": "LTX2Pipeline"}))

    assert family == "ltx2"
    assert support.default_task == "text_to_audio_video"
    assert support.default_precision == "bf16"
