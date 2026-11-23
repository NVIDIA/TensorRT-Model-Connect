# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_ideogram_pipeline_has_a_text_to_image_owner():
    family, support = resolve_family(
        ModelMetadata(config={}, model_index={"_class_name": "Ideogram4Pipeline"})
    )
    assert family == "ideogram4"
    assert support.tasks == ("text_to_image",)
    assert support.default_task == "text_to_image"
    assert support.default_precision == "bf16"
