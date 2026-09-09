# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_qwen_image_default_task_is_checkpoint_owned() -> None:
    _, generation = resolve_family(
        ModelMetadata({}, {"_class_name": "QwenImagePipeline"})
    )
    _, editing = resolve_family(
        ModelMetadata({}, {"_class_name": "QwenImageEditPipeline"})
    )

    assert generation.default_task == "image_generation"
    assert editing.default_task == "image_edit"
