# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_wan2_2_ti2v_default_task_is_checkpoint_owned() -> None:
    family, support = resolve_family(ModelMetadata({"model_type": "ti2v"}, {}))

    assert family == "wan2_2_ti2v"
    assert support.default_task == "image_generation"
    assert {"text_to_video", "initial_image_text_to_video"} <= set(support.tasks)
